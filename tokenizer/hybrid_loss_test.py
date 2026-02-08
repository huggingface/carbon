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
# Cell 4: Create simulated input data
# ============================================================================

# Create a small batch for testing
batch_size = 2
seq_len = 8
vocab_local = dna_start_id + dna_vocab_size  # Local vocabulary size (to include all DNA tokens + NL token)

# Create sharded_logits (simulating model output)
torch.manual_seed(42)
sharded_logits = torch.randn(batch_size * seq_len, vocab_local) * 0.1

# Create label_ids (already shifted labels)
label_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)

# NL token IDs (assumed to be outside DNA vocabulary, less than dna_start_id)
nl_token_ids = [0, 0, 0, 0, 0]  # Using 0 for NL tokens

# DNA k-mer IDs (within DNA vocabulary)
dna_kmer_ids = [kmer_start_id, kmer_start_id + 1, kmer_start_id + 2, 
                kmer_start_id + 10, kmer_start_id + 20]

# Batch 0: [NL, NL, DNA-kmer, DNA-kmer, DNA-kmer, NL, padding, padding]
label_ids[0, 0] = nl_token_ids[0]  # NL
label_ids[0, 1] = nl_token_ids[1]  # NL
label_ids[0, 2] = dna_kmer_ids[0]  # DNA k-mer
label_ids[0, 3] = dna_kmer_ids[1]  # DNA k-mer
label_ids[0, 4] = dna_kmer_ids[2]  # DNA k-mer
label_ids[0, 5] = nl_token_ids[2]  # NL
label_ids[0, 6] = dna_start_id     # <dna> special token
label_ids[0, 7] = dna_start_id + 1 # </dna> special token

# Batch 1: [DNA-kmer, DNA-kmer, NL, DNA-kmer, NL, NL, DNA-kmer, padding]
label_ids[1, 0] = dna_kmer_ids[0]  # DNA k-mer
label_ids[1, 1] = dna_kmer_ids[1]  # DNA k-mer
label_ids[1, 2] = nl_token_ids[3]  # NL
label_ids[1, 3] = dna_kmer_ids[3]  # DNA k-mer
label_ids[1, 4] = nl_token_ids[4]  # NL
label_ids[1, 5] = nl_token_ids[0]  # NL
label_ids[1, 6] = dna_kmer_ids[4]  # DNA k-mer
label_ids[1, 7] = dna_start_id     # <dna> special token

# Create label_mask (1 indicates positions that should calculate loss)
label_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
label_mask[0, :6] = 1  # First 6 tokens should calculate loss
label_mask[1, :7] = 1  # First 7 tokens should calculate loss

# Create token_mask (MUST be aligned with label_ids, already shifted)
# -2: padding, -1: NL, 0: DNA special token, 1..k: DNA k-mer
token_mask = torch.full((batch_size, seq_len), -2)  # Initialize as padding

# Batch 0
token_mask[0, 0] = -1  # NL
token_mask[0, 1] = -1  # NL
token_mask[0, 2] = 6   # DNA k-mer, valid length 6 (changed from 3 to 6)
token_mask[0, 3] = 6   # DNA k-mer, valid length 6
token_mask[0, 4] = 4   # DNA k-mer, valid length 4 (tail A-padding)
token_mask[0, 5] = -1  # NL
token_mask[0, 6] = 0   # DNA special token (<dna>)
token_mask[0, 7] = 0   # DNA special token (</dna>)

# Batch 1
token_mask[1, 0] = 6   # DNA k-mer, valid length 6
token_mask[1, 1] = 2   # DNA k-mer, valid length 2 (very short fragment)
token_mask[1, 2] = -1  # NL
token_mask[1, 3] = 6   # DNA k-mer, valid length 6
token_mask[1, 4] = -1  # NL
token_mask[1, 5] = -1  # NL
token_mask[1, 6] = 6   # DNA k-mer, valid length 6
token_mask[1, 7] = 0   # DNA special token (<dna>)

print("\n✓ Simulated input data created successfully!")
print(f"sharded_logits shape: {sharded_logits.shape}")
print(f"label_ids shape: {label_ids.shape}")
print("\nlabel_mask:\n", label_mask)
print("\ntoken_mask:\n", token_mask)

# ============================================================================
# Cell 5: Test _assert_alignment checks
# ============================================================================

print("\nTesting _assert_alignment checks...")
try:
    loss_fn._assert_alignment(label_ids, label_mask, token_mask)
    print("✓ _assert_alignment check passed")
except Exception as e:
    print(f"✗ _assert_alignment check failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Cell 6: Test _bp_nll_sum_and_count function
# ============================================================================

print("\nTesting _bp_nll_sum_and_count function...")

# Create a small test case
test_batch_size = 3
test_vocab_local = vocab_local

# Create test logits
test_logits = torch.randn(test_batch_size, test_vocab_local) * 0.1

# Create test labels (DNA k-mer IDs)
test_label_ids = torch.tensor([
    kmer_start_id,      # AAAAAA
    kmer_start_id + 21, # TTTTTT
    kmer_start_id + 42  # CCCCCC
])

# Create valid lengths
test_valid_len = torch.tensor([6, 4, 2])  # Test different lengths

try:
    # Ensure cache is built
    loss_fn._maybe_build_local_cache(test_logits.device, test_vocab_local)

    # Calculate bp loss
    bp_sum, bp_count = loss_fn._bp_nll_sum_and_count(
        test_logits, test_label_ids, test_valid_len
    )

    print(f"✓ _bp_nll_sum_and_count calculation completed")
    print(f"  bp_sum: {bp_sum.item():.6f}")
    print(f"  bp_count: {bp_count.item()}")

    # Verify count is correct
    expected_count = test_valid_len.sum().item()  # 6+4+2=12
    print(f"  Expected bp_count: {expected_count}, Actual: {bp_count.item()}")

    if bp_count.item() == expected_count:
        print("  ✓ bp_count is correct")
    else:
        print(f"  ✗ bp_count error: expected {expected_count}, got {bp_count.item()}")
except Exception as e:
    print(f"✗ _bp_nll_sum_and_count calculation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Cell 7: Test complete HybridLoss forward
# ============================================================================

print("\nTesting complete HybridLoss forward...")

try:
    # Calculate loss
    outputs = loss_fn(
        sharded_logits=sharded_logits,
        label_ids=label_ids,
        label_mask=label_mask,
        token_mask=token_mask
    )

    print(f"✓ HybridLoss forward calculation completed")
    print(f"  Total loss: {outputs['loss'].item():.6f}")

    # Manually verify some counts
    nl_pos = (label_mask > 0) & (token_mask == -1)
    dna_pos = (label_mask > 0) & (token_mask >= 1) & (token_mask <= k)

    nl_count = nl_pos.sum().item()
    bp_count_manual = token_mask[dna_pos].sum().item() if dna_pos.any() else 0

    print(f"  NL token count: {nl_count}")
    print(f"  DNA base count: {bp_count_manual}")
    
    # Verify loss value is reasonable
    if outputs['loss'].item() > 0:
        print("  ✓ Loss is positive (as expected)")
    else:
        print("  ✗ Loss value is abnormal")
except Exception as e:
    print(f"✗ HybridLoss forward calculation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Cell 8: Test HybridLossWithZLoss
# ============================================================================

print("\nTesting HybridLossWithZLoss...")

try:
    # Create loss function with z-loss
    loss_fn_with_z = HybridLossWithZLoss(
        tp_pg=tp_group,  # Use same real process group
        k=k,
        dna_start_id=dna_start_id,
        dna_vocab_size=dna_vocab_size,
        dna_special_tokens=dna_special_tokens,
        dna_id_to_token=dna_id_to_token,
        nl_weight=1.0,
        bp_weight=1.0,
        eps=1e-8,
        z_loss_coefficient=0.1
    )

    # Calculate loss
    outputs_with_z = loss_fn_with_z(
        sharded_logits=sharded_logits,
        label_ids=label_ids,
        label_mask=label_mask,
        token_mask=token_mask
    )

    print(f"✓ HybridLossWithZLoss forward calculation completed")
    print(f"  Total loss: {outputs_with_z['loss'].item():.6f}")
    print(f"  z_loss: {outputs_with_z['z_loss'].item():.6f}")
    
    # Compare with regular loss if available
    if 'outputs' in locals():
        if outputs_with_z['loss'].item() > outputs['loss'].item():
            print("  ✓ z-loss increased total loss (as expected)")
        else:
            print("  ✗ z-loss did not increase total loss")
except Exception as e:
    print(f"✗ HybridLossWithZLoss test failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Cell 9: Test edge cases
# ============================================================================

print("\nTesting edge cases...")

# Test 1: Empty supervision positions
print("Test 1: Empty supervision positions...")
empty_label_mask = torch.zeros_like(label_mask)

try:
    outputs_empty = loss_fn(
        sharded_logits=sharded_logits,
        label_ids=label_ids,
        label_mask=empty_label_mask,
        token_mask=token_mask
    )
    print(f"  ✓ Empty supervision positions handled, loss: {outputs_empty['loss'].item():.6f}")
    if outputs_empty['loss'].item() == 0:
        print("  ✓ Loss is 0 for empty supervision (as expected)")
except Exception as e:
    print(f"  ✗ Empty supervision test failed: {e}")

# Test 2: Only NL tokens
print("\nTest 2: Only NL tokens...")
only_nl_label_ids = torch.full_like(label_ids, nl_token_ids[0])
only_nl_token_mask = torch.full_like(token_mask, -1)
# Keep label_mask unchanged

try:
    outputs_only_nl = loss_fn(
        sharded_logits=sharded_logits,
        label_ids=only_nl_label_ids,
        label_mask=label_mask,
        token_mask=only_nl_token_mask
    )
    print(f"  ✓ Only NL tokens test completed, loss: {outputs_only_nl['loss'].item():.6f}")
except Exception as e:
    print(f"  ✗ Only NL tokens test failed: {e}")

# Test 3: Only DNA tokens
print("\nTest 3: Only DNA tokens...")
only_dna_label_ids = torch.full_like(label_ids, dna_kmer_ids[0])
only_dna_token_mask = torch.full_like(token_mask, 6)
# Keep label_mask unchanged

try:
    outputs_only_dna = loss_fn(
        sharded_logits=sharded_logits,
        label_ids=only_dna_label_ids,
        label_mask=label_mask,
        token_mask=only_dna_token_mask
    )
    print(f"  ✓ Only DNA tokens test completed, loss: {outputs_only_dna['loss'].item():.6f}")
except Exception as e:
    print(f"  ✗ Only DNA tokens test failed: {e}")

# ============================================================================
# Cell 10: Test gradient computation
# ============================================================================

print("\nTesting gradient computation...")

try:
    # Enable gradient
    sharded_logits_with_grad = sharded_logits.clone().requires_grad_(True)

    # Calculate loss
    outputs_grad = loss_fn(
        sharded_logits=sharded_logits_with_grad,
        label_ids=label_ids,
        label_mask=label_mask,
        token_mask=token_mask
    )

    # Backward pass
    loss_value = outputs_grad['loss']
    loss_value.backward()

    print(f"✓ Gradient computation completed")
    print(f"  Loss value: {loss_value.item():.6f}")

    # Check if gradient exists
    if sharded_logits_with_grad.grad is not None:
        grad_norm = sharded_logits_with_grad.grad.norm().item()
        print(f"  Gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 0:
            print("  ✓ Gradient is non-zero (correct)")
        else:
            print("  ✗ Gradient is zero (might be problematic)")
    else:
        print("  ✗ No gradient")
except Exception as e:
    print(f"✗ Gradient computation failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Cell 11: Simplified test: Direct verification of core logic
# ============================================================================

print("\n" + "="*60)
print("Simplified test: Verify core logic")
print("="*60)

# Test 1: Verify DNA k-mer mapping
print("\nTest 1: Verify DNA k-mer mapping...")
test_kmer_id = kmer_start_id
if test_kmer_id in dna_id_to_token:
    token = dna_id_to_token[test_kmer_id]
    print(f"  K-mer ID {test_kmer_id} -> '{token}'")
    
    # Check if in nt_table
    idx = test_kmer_id - loss_fn.dna_kmer_start_id
    if 0 <= idx < loss_fn.num_dna_kmers:
        nts = loss_fn._dna_nt_table[idx].tolist()
        print(f"  Nucleotide mapping: {nts}")
        if all(n >= 0 for n in nts):
            print("  ✓ Nucleotide mapping is correct")
        else:
            print("  ✗ Nucleotide mapping contains -1 (invalid)")
    else:
        print(f"  ✗ ID not in k-mer range")
else:
    print(f"  ✗ ID not in mapping")

# Test 2: Manual BP loss calculation
print("\nTest 2: Manual BP loss calculation logic...")
# Create a simple test case
simple_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32).repeat(1, vocab_local)
simple_label = torch.tensor([kmer_start_id])
simple_valid_len = torch.tensor([6])

try:
    simple_bp_sum, simple_bp_count = loss_fn._bp_nll_sum_and_count(
        simple_logits, simple_label, simple_valid_len
    )
    print(f"  Simple case BP loss: {simple_bp_sum.item():.6f}")
    print(f"  BP count: {simple_bp_count.item()}")
    print(f"  ✓ BP loss calculation successful")
except Exception as e:
    print(f"  ✗ BP loss calculation failed: {e}")

# Test 3: Verify token_mask semantics
print("\nTest 3: Verify token_mask semantics...")
print("  token_mask value meanings:")
print("  -2: padding")
print("  -1: natural language token")
print("   0: DNA special/ignored token")
print(f"  1..{k}: DNA k-mer token (valid_len)")

# ============================================================================
# Cell 12: Cleanup and summary
# ============================================================================

# Clean up distributed environment
if dist.is_initialized():
    dist.destroy_process_group()
    print("\n✓ Distributed environment cleaned up")

print("\n" + "="*60)
print("Test Summary")
print("="*60)

print("""
Testing completed!

Main modifications:
1. Used real PyTorch distributed environment
2. Created real dist.new_group() as TP group
3. Directly used real process group for HybridLoss
4. All distributed calls are real, no mock needed

Key points:
1. HybridLoss requires real dist process group
2. sharded_cross_entropy internally calls dist.get_rank(group)
3. In single process testing, we need to initialize distributed environment
4. Created new_group() to simulate TP group

If issues persist, consider:
1. Special handling in nanotron's distributed module
2. Check if HybridLoss uses the correct dist module
3. May need to directly mock nanotron.distributed module
""")