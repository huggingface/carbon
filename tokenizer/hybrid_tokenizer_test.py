import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from hybrid_tokenizer import HybridTokenizer
from transformers import AutoTokenizer

# Initialize tokenizer
tokenizer = HybridTokenizer(
    base_model="Qwen/Qwen3-0.6B-Base",
    k=6
)

# Save and load tokenizer for consistency test
tokenizer.save_pretrained("hybrid_tokenizer")
loaded_tokenizer = HybridTokenizer.from_pretrained("hybrid_tokenizer")

base_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

print("=" * 80)
print("HybridTokenizer Test Script")
print("=" * 80)

# Test 0: Verify tokenizer consistency before and after loading
print("\n0. Test: Tokenizer Consistency Verification (Before/After Loading)")
print("-" * 60)

test_seq = '<dna>ATCGATCG</dna>'
print(f"Input: '{test_seq}'")

token_ids_original = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_loaded = loaded_tokenizer.encode(test_seq, return_token_mask=False)

print(f"Original Token IDs: {token_ids_original}")
print(f"Loaded Token IDs: {token_ids_loaded}")

if token_ids_original != token_ids_loaded:
    print("ERROR: Tokenizer output mismatch before and after loading!")
else:
    print("SUCCESS: Tokenizer output consistent before and after loading.")

# Test natural language only
test_seq = 'I am a test sentence of human language only.'
print(f"\nInput: '{test_seq}'")

token_ids_original = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_loaded = loaded_tokenizer.encode(test_seq, return_token_mask=False)
token_ids_base = base_tokenizer.encode(test_seq)

print(f"Original Token IDs: {token_ids_original}")
print(f"Loaded Token IDs: {token_ids_loaded}")
print(f"Base Tokenizer Token IDs: {token_ids_base}")

if token_ids_original != token_ids_loaded:
    print("ERROR: Tokenizer output mismatch before and after loading!")
else:
    print("SUCCESS: Tokenizer output consistent before and after loading.")
    
if token_ids_original != token_ids_base:
    print("ERROR: HybridTokenizer and Base Tokenizer output mismatch!")
else:
    print("SUCCESS: HybridTokenizer and Base Tokenizer output consistent.")

# Test 1: Only end tag without start tag
print("\n1. Test: Only End Tag Without Start Tag 'ATCG</dna>'")
print("-" * 60)

test_seq = 'ATCG</dna>'
print(f"Input: '{test_seq}'")

token_ids = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}") # Token Mask: [4, 0], 4 valid bases in ATCG and 2 padding bases AA
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 2: Only start tag without end tag
print("\n2. Test: Only Start Tag Without End Tag '<dna>ATCG'")
print("-" * 60)

test_seq = '<dna>ATCG'
print(f"Input: '{test_seq}'")

token_ids = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 3: Complete DNA region
print("\n3. Test: Complete DNA Region '<dna>ATCGATCG</dna>'")
print("-" * 60)

test_seq = '<dna>ATCGATCG</dna>'
print(f"Input: '{test_seq}'")

token_ids = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 4: Mixed text and DNA
print("\n4. Test: Mixed Text and DNA")
print("-" * 60)

test_seq = 'The DNA sequence is <dna>ATCG</dna> in this example.'
print(f"Input: '{test_seq}'")

token_ids = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 5: Multiple DNA regions
print("\n5. Test: Multiple DNA Regions")
print("-" * 60)

test_seq = 'First segment <dna>ATCG</dna> and second segment <dna>ATCGG</dna> and third segment <dna>ATCG'
print(f"Input: '{test_seq}'")

token_ids = tokenizer.encode(test_seq, return_token_mask=False, add_special_tokens=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True, add_special_tokens=False)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 6: Padding for non-k-multiple length DNA
print("\n6. Test: DNA Sequence with Non-k-multiple Length (Padding)")
print("-" * 60)

test_seq = '<dna>ATCGAT</dna>'
print(f"Input: '{test_seq}' (For 6-mer, ATCGAT is a complete token)")

token_ids = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 7: Short DNA sequence requiring padding
print("\n7. Test: Short DNA Sequence '<dna>AT</dna>' (Requires padding in 6-mer)")
print("-" * 60)

test_seq = '<dna>AT</dna>'
print(f"Input: '{test_seq}'")

token_ids = tokenizer.encode(test_seq, return_token_mask=False)
token_ids_with_mask, token_mask = tokenizer.encode(test_seq, return_token_mask=True)

print(f"Token IDs: {token_ids}")
print(f"Token IDs (with mask): {token_ids_with_mask}")
print(f"Token Mask: {token_mask}")
print(f"Decoded: '{tokenizer.decode(token_ids)}'")

# Test 8: Batch processing using __call__ method
print("\n8. Test: Batch Processing")
print("-" * 60)

test_seqs = [
    'ATCG</dna>',
    '<dna>ATCGATCG</dna>',
    'The DNA sequence is <dna>ATCG</dna>'
]

print("Input Sequences:")
for i, seq in enumerate(test_seqs):
    print(f"  {i+1}. '{seq}'")

result = tokenizer(
    test_seqs,
    padding=True,
    truncation=True,
    max_length=50,
    return_token_mask=True
)

print(f"\nBatch Results:")
print(f"  input_ids shape: {len(result['input_ids'])}x{len(result['input_ids'][0])}")
print(f"  attention_mask shape: {len(result['attention_mask'])}x{len(result['attention_mask'][0])}")
print(f"  token_mask shape: {len(result['token_mask'])}x{len(result['token_mask'][0])}")

# Show details for first sequence
print("\nFirst Sequence Details:")
print(f"  input_ids: {result['input_ids'][0]}")
print(f"  attention_mask: {result['attention_mask'][0]}")
print(f"  token_mask: {result['token_mask'][0]}")

# Decoding verification
print("\nDecoding Verification:")
for i, seq in enumerate(test_seqs):
    decoded = tokenizer.decode(result['input_ids'][i], skip_special_tokens=False)
    print(f"Sequence {i+1}:")
    print(f"  Original: '{seq}'")
    print(f"  Decoded: '{decoded}'")
    print()

# Test 9: Token mask value meaning verification
print("\n9. Test: Token Mask Value Meaning Verification")
print("=" * 60)

print("Token Mask Value Meanings:")
print("  -2: Padding token")
print("  -1: Natural language token")
print("   0: DNA special token (<dna>, </dna>, <oov>)")
print(f"   1-{tokenizer.k-1}: Partially padded k-mer token (number of valid bases, when k={tokenizer.k})")
print(f"   {tokenizer.k}: Full k-mer token (when k={tokenizer.k})")

# Create a test sequence with various situations
test_seq = "Start <dna>ATCGAT</dna> middle <dna>ATNCG</dna> end"
print(f"\nTest Sequence: '{test_seq}'")

result = tokenizer(
    test_seq,
    padding=False,
    truncation=False,
    return_token_mask=True,
    return_tensors=None
)

print(f"\nToken IDs: {result['input_ids']}")
print(f"Token Mask: {result['token_mask']}")

# Verify each token type
print("\nToken Detailed Analysis:")
for i, (token_id, mask_value) in enumerate(zip(result['input_ids'], result['token_mask'])):
    token = tokenizer.decode([token_id])  # Decode single token
    token = token.strip()
    
    if mask_value == -1:
        token_type = "Natural language token"
    elif mask_value == 0:
        token_type = "DNA special token"
    elif mask_value == tokenizer.k:
        token_type = f"Full k-mer token ({tokenizer.k}-mer)"
    elif 0 < mask_value < tokenizer.k:
        token_type = f"Partially padded k-mer token ({mask_value} valid bases)"
    elif mask_value == -2:
        token_type = "Padding token"
    else:
        token_type = f"Unknown type (value: {mask_value})"
    
    print(f"  [{i:2d}] Token ID: {token_id:6d}, Token: '{token:10s}', Mask: {mask_value:3d}, Type: {token_type}")

print("\n" + "=" * 80)
print("Test Completed!")
print("=" * 80)
