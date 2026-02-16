import torch
import torch.nn as nn
import torch.distributed as dist
import os
from unittest.mock import Mock, patch
import numpy as np
from hybrid_loss import *

# ============================================================================
# Cell 1: Initialize distributed environment and create real process group
# ============================================================================

def init_distributed_for_testing():
    """Initialize distributed environment for testing"""
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend='gloo',  # Use gloo backend for better compatibility
            init_method='env://',
            world_size=1,
            rank=0
        )
        print("✓ Distributed environment initialized")
    
    # Create a new process group (simulating TP group)
    # In single process, we create a group containing all processes (only one)
    world_size = dist.get_world_size()
    ranks = list(range(world_size))
    
    # Create new group
    tp_group = dist.new_group(ranks=ranks)
    
    print(f"✓ Created TP process group, size={world_size}, rank={dist.get_rank(tp_group)}")
    return tp_group

# Initialize and get process group
tp_group = init_distributed_for_testing()

# ============================================================================
# Cell 2: Create token-id mapping for testing
# ============================================================================

# Parameters
k = 6  # k-mer length (changed from 3 to 6)
dna_start_id = 128
dna_vocab_size = 4100  # 4^6 = 4096 k-mers + 4 special tokens
dna_special_tokens = ["<dna>", "</dna>", "<oov>", "<s>"]

# Create k-mer to id mapping (simulating real tokenizer)
dna_id_to_token = {}

# Add special tokens
for i, token in enumerate(dna_special_tokens):
    token_id = dna_start_id + i
    dna_id_to_token[token_id] = token

# Add k-mers (simulating all possible 6-mers)
nucleotides = ['A', 'T', 'C', 'G']
kmer_start_id = dna_start_id + len(dna_special_tokens)

# Generate all 4^6 = 4096 k-mers
for i in range(4**k):
    # Generate k-mer sequence (simple simulation)
    seq = ""
    val = i
    for _ in range(k):
        seq = nucleotides[val % 4] + seq
        val //= 4
    
    token_id = kmer_start_id + i
    dna_id_to_token[token_id] = seq

print(f"\nCreated {len(dna_id_to_token)} DNA token mappings")
print(f"Special tokens: {dna_special_tokens}")
print(f"k-mer examples: AAAAAA->{kmer_start_id}, TTTTTT->{kmer_start_id+21}")

# ============================================================================
# Cell 3: Create HybridLoss instance
# ============================================================================

# Create HybridLoss - using real dist process group
loss_fn = HybridLoss(
    tp_pg=tp_group,  # Use real process group
    k=k,
    dna_start_id=dna_start_id,
    dna_vocab_size=dna_vocab_size,
    dna_special_tokens=dna_special_tokens,
    dna_id_to_token=dna_id_to_token,
    nl_weight=1.0,
    bp_weight=1.0,
    eps=1e-8
)

print("\n✓ HybridLoss created successfully!")
print(f"DNA k-mer range: [{loss_fn.dna_kmer_start_id}, {loss_fn.dna_kmer_end_id})")
print(f"Number of DNA k-mers: {loss_fn.num_dna_kmers}")

# Check nucleotide table
print(f"\nNucleotide table shape: {loss_fn._dna_nt_table.shape}")
print("First 5 k-mer nucleotide mappings:")
for i in range(5):
    kmer_id = loss_fn.dna_kmer_start_id + i
    tokens = dna_id_to_token.get(kmer_id, "unknown")
    nts = loss_fn._dna_nt_table[i].tolist()
    print(f"  ID {kmer_id} ({tokens}): {nts}")

# ============================================================================
# Cell 4: Test Valid Length Masking via Token Mask
# ============================================================================

print("\nTesting _bp_nll_sum_and_count function...")

# Create a small test case
test_batch_size = 3
test_vocab_local = dna_start_id + dna_vocab_size  

# Create test logits
test_logits = torch.randn(test_batch_size, test_vocab_local) * 0.1

# Create test labels (DNA k-mer IDs)
test_label_ids = torch.tensor([
    kmer_start_id,      # AAAAAA
    kmer_start_id + 1,  # AAAAAT
    kmer_start_id + 2   # AAAAAC
])

# Case 1: valid k-mers
test_valid_len = torch.tensor([6, 6, 6])  # Test different lengths

# Ensure cache is built
loss_fn._maybe_build_local_cache(test_logits.device, test_vocab_local)

# Calculate bp loss
bp_sum, bp_count = loss_fn._bp_nll_sum_and_count(
    test_logits, test_label_ids, test_valid_len
)

print(f"✓ _bp_nll_sum_and_count calculation completed")
print(f"  bp_sum: {bp_sum.item():.6f}")
print(f"  bp_count: {bp_count.item()}")

print(f"Expected mean CE loss: Log(test_vocab_local)/k = {np.log(test_vocab_local)/k}")
print(f"Actual mean CE loss: {bp_sum.item() / bp_count.item()}")

outputs = loss_fn(
    sharded_logits=test_logits,
    label_ids=test_label_ids.unsqueeze(0),
    label_mask=torch.ones_like(test_label_ids).unsqueeze(0),
    token_mask=test_valid_len.unsqueeze(0)
)

print(f"✓ HybridLoss forward calculation completed")
print(outputs['loss'] == bp_sum/bp_count)






# Case 2: invalid k-mers
test_valid_len = torch.tensor([6, 6, 1])  # Test different lengths

# Ensure cache is built
loss_fn._maybe_build_local_cache(test_logits.device, test_vocab_local)

# Calculate bp loss
bp_sum, bp_count = loss_fn._bp_nll_sum_and_count(
    test_logits, test_label_ids, test_valid_len
)

print(f"✓ _bp_nll_sum_and_count calculation completed")
print(f"  bp_sum: {bp_sum.item():.6f}")
print(f"  bp_count: {bp_count.item()}")

print(f"Expected mean CE loss: Log(test_vocab_local)/k = {np.log(test_vocab_local)/k}")
print(f"Actual mean CE loss: {bp_sum.item() / bp_count.item()}")

outputs = loss_fn(
    sharded_logits=test_logits,
    label_ids=test_label_ids.unsqueeze(0),
    label_mask=torch.ones_like(test_label_ids).unsqueeze(0),
    token_mask=test_valid_len.unsqueeze(0)
)

print(f"✓ HybridLoss forward calculation completed")
print(outputs['loss'] == bp_sum/bp_count)


# ============================================================================
# Cell 5: Token Mask in Real Example
# ============================================================================

from hybrid_tokenizer import HybridTokenizer

# Initialize tokenizer
tokenizer = HybridTokenizer(
    base_model="Qwen/Qwen2-0.5B",
    k=6
)

test_seq = 'The DNA sequence is <dna>TTT</dna> in this example.'
print(f"Input: '{test_seq}'")

inputs = tokenizer(
    test_seq,
    add_special_tokens=False,
    return_tensors="pt",
    padding=True,
    truncation=True,
    return_token_mask=True,
    max_length=None
)

print(f"input_ids: {inputs['input_ids']}")
print(f"attention_mask: {inputs['attention_mask']}")
print(f"token_mask: {inputs['token_mask']}")


# Clean up distributed environment
if dist.is_initialized():
    dist.destroy_process_group()