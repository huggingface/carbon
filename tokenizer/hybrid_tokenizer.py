import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import torch
from typing import List, Tuple, Dict, Optional, Union, Any
import itertools
import numpy as np
from transformers import AutoTokenizer


class HybridTokenizer:
    """
    Hybrid DNA Tokenizer that supports special processing for DNA regions.
    Provides an interface similar to Hugging Face tokenizers.
    """
    
    def __init__(
        self,
        base_model: str = "Qwen/Qwen3-0.6B-Base",
        k: int = 6,
        padding_side: str = "right",
        truncation_side: str = "right",
        model_max_length: int = 8192,
    ):
        """
        Initialize the Hybrid DNA Tokenizer.
        
        Args:
            base_model: Name of the base model
            k: DNA k-mer length
            padding_side: Padding direction ("left" or "right")
            truncation_side: Truncation direction ("left" or "right")
            model_max_length: Maximum model length
        """
        # Load base tokenizer
        self.base_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Basic attributes
        self.k = k
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.model_max_length = model_max_length
        
        # Inherit attributes from base tokenizer
        self.bos_token_id = self.base_tokenizer.bos_token_id
        self.eos_token_id = self.base_tokenizer.eos_token_id
        self.pad_token_id = self.base_tokenizer.pad_token_id
        self.unk_token_id = self.base_tokenizer.unk_token_id
        
        # Initialize DNA vocabulary
        self._init_dna_vocab()
        
        # Extend vocabulary
        self._extend_vocab()
    
    def _init_dna_vocab(self):
        """Initialize DNA vocabulary."""
        bases = ['A', 'T', 'C', 'G']
        
        # DNA special tokens
        self.dna_special_tokens = ["<dna>", "</dna>", "<oov>"]
        
        # Generate k-mer combinations
        kmers = [''.join(kmer) for kmer in itertools.product(bases, repeat=self.k)]
        kmers = kmers[:4096]  # Limit number
        
        # Create DNA vocabulary mappings
        self.dna_token_to_id = {}
        self.dna_id_to_token = {}
        
        # DNA token starting ID (after current vocabulary size)
        self.dna_start_id = len(self.base_tokenizer)
        
        # Add all DNA tokens
        all_dna_tokens = self.dna_special_tokens + kmers
        
        for i, token in enumerate(all_dna_tokens):
            token_id = self.dna_start_id + i
            self.dna_token_to_id[token] = token_id
            self.dna_id_to_token[token_id] = token
        
        self.dna_vocab_size = len(all_dna_tokens)
        
        # Set DNA special token IDs
        self.dna_begin_token_id = self.dna_token_to_id["<dna>"]
        self.dna_end_token_id = self.dna_token_to_id["</dna>"]
        self.oov_token_id = self.dna_token_to_id["<oov>"]
    
    def _extend_vocab(self):
        """Extend vocabulary to include DNA tokens."""
        # Get base vocabulary
        self.vocab = self.base_tokenizer.get_vocab().copy()
        
        # Add DNA tokens
        for token, token_id in self.dna_token_to_id.items():
            if token not in self.vocab:
                self.vocab[token] = token_id
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self):
        """Get vocabulary."""
        return self.vocab.copy()
    
    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert tokens to IDs."""
        if isinstance(tokens, str):
            # Try DNA vocabulary first
            if tokens in self.dna_token_to_id:
                return self.dna_token_to_id[tokens]
            # Then try base vocabulary
            return self.base_tokenizer.convert_tokens_to_ids(tokens)
        else:
            # Batch conversion
            return [self.convert_tokens_to_ids(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert IDs to tokens."""
        if isinstance(ids, (int, np.integer)):
            # Try DNA vocabulary first
            if ids in self.dna_id_to_token:
                return self.dna_id_to_token[ids]
            # Then try base vocabulary
            return self.base_tokenizer.convert_ids_to_tokens(ids)
        else:
            # Batch conversion
            return [self.convert_ids_to_tokens(id_) for id_ in ids]
    
    def _split_by_dna_tags(self, text: str) -> List[Tuple[str, bool]]:
        """
        Split text by DNA tags, handling all edge cases correctly.
        
        Returns:
            List[Tuple[str, bool]]: Each element is (text segment, is_dna_region)
        """
        segments = []
        
        i = 0
        n = len(text)
        
        while i < n:
            # Find next <dna> and </dna> tags from current position
            start_pos = text.find('<dna>', i)
            end_pos = text.find('</dna>', i)
            
            # Case 1: No more tags at all
            if start_pos == -1 and end_pos == -1:
                remaining = text[i:].strip()
                if remaining:
                    segments.append((remaining, False))
                break
            
            # Case 2: Only end tag exists
            if start_pos == -1 and end_pos != -1:
                # Everything from current position to end_pos+6 is DNA region
                dna_region = text[i:end_pos + 6].strip()
                if dna_region:
                    segments.append((dna_region, True))
                i = end_pos + 6
                continue
            
            # Case 3: Only start tag exists
            if start_pos != -1 and end_pos == -1:
                # Add normal text before the start tag (if any)
                if i < start_pos:
                    normal_text = text[i:start_pos].strip()
                    if normal_text:
                        segments.append((normal_text, False))
                
                # Everything from start_pos to end is DNA region
                dna_region = text[start_pos:].strip()
                if dna_region:
                    segments.append((dna_region, True))
                i = n  # 这里应该是break，但为了与后面的逻辑一致，我们设置i=n
                break
            
            # Case 4: Both tags exist, start tag comes first
            if start_pos < end_pos:
                # Add normal text before start tag
                if i < start_pos:
                    normal_text = text[i:start_pos].strip()
                    if normal_text:
                        segments.append((normal_text, False))
                
                # Find the matching end tag
                closing_pos = end_pos
                
                # Ensure this is the matching end tag (not a different one before start)
                dna_region = text[start_pos:closing_pos + 6].strip()
                if dna_region:
                    segments.append((dna_region, True))
                
                i = closing_pos + 6
            
            # Case 5: Both tags exist, end tag comes first (isolated end tag)
            else:
                # Everything from current position to end_pos+6 is DNA region
                dna_region = text[i:end_pos + 6].strip()
                if dna_region:
                    segments.append((dna_region, True))
                
                i = end_pos + 6
        
        return segments

    def _parse_dna_region(self, dna_region: str) -> Tuple[str, bool, bool]:
        """Parse DNA region, extract content and tag information."""
        # Handle special case where region might be just a tag (e.g., just "</dna>")
        if dna_region == '<dna>':
            return '', True, False
        elif dna_region == '</dna>':
            return '', False, True
        
        has_start = dna_region.startswith('<dna>')
        has_end = dna_region.endswith('</dna>')
        
        # Extract DNA content
        content = dna_region
        if has_start:
            content = content[5:]  # Remove <dna>
        if has_end:
            # Make sure we only remove </dna> if it's at the end
            # Could be content that contains </dna> elsewhere, but for our use case it should be at end
            if content.endswith('</dna>'):
                content = content[:-6]  # Remove </dna>
        
        return content.strip(), has_start, has_end
    
    def _process_dna_sequence(self, dna_seq: str) -> Dict:
        """Process DNA sequence, return k-mer tokens and other information."""
        k = self.k
        dna_seq = dna_seq.upper()
        
        kmer_tokens = []
        oov_positions = []
        valid_bases = set('ATCG')
        
        def _is_valid_kmer(kmer):
            return len(kmer) == k and all(base in valid_bases for base in kmer)
        
        # Process complete k-mers
        for i in range(0, len(dna_seq) - k + 1, k):
            kmer = dna_seq[i:i+k]
            if _is_valid_kmer(kmer):
                kmer_tokens.append(kmer)
            else:
                kmer_tokens.append("<oov>")
                oov_positions.append(len(kmer_tokens) - 1)
        
        # Process trailing part
        processed_length = len(kmer_tokens) * k
        padding_length = 0
        
        remaining_start = processed_length
        remaining = dna_seq[remaining_start:]
        
        if remaining:
            padding_needed = k - len(remaining)
            padded_remaining = remaining + 'A' * padding_needed
            
            if _is_valid_kmer(padded_remaining):
                kmer_tokens.append(padded_remaining)
            else:
                kmer_tokens.append("<oov>")
                oov_positions.append(len(kmer_tokens) - 1)
            
            padding_length = padding_needed
        
        return {
            "kmer_tokens": kmer_tokens,
            "padding_length": padding_length,
            "original_sequence": dna_seq,
            "oov_positions": oov_positions,
            "oov_count": len(oov_positions),
            "valid_length": len(remaining) if remaining else k  # 有效碱基数
        }
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_token_mask: bool = True,
        **kwargs
    ) -> Union[List[int], Tuple[List[int], List[int]]]:
        """
        Encode a single text.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens (e.g., BOS/EOS)
            truncation: Whether to truncate
            max_length: Maximum length
            return_token_mask: Whether to return token mask
            **kwargs: Other arguments
            
        Returns:
            List of token IDs, or tuple of (token_ids, token_mask) if return_token_mask=True
        """
        # Split text
        segments = self._split_by_dna_tags(text)
        
        token_ids = []
        token_mask = [] if return_token_mask else None
        
        # Add BOS token if needed
        if add_special_tokens and self.bos_token_id is not None:
            token_ids.append(self.bos_token_id)
            if return_token_mask:
                token_mask.append(-1)  # BOS token also treated as natural language
        
        # Process each segment
        for segment_content, is_dna in segments:
            if is_dna:
                # Parse DNA region
                dna_content, has_start_tag, has_end_tag = self._parse_dna_region(segment_content)
                
                # Add start token if there's a start tag
                if has_start_tag:
                    token_ids.append(self.dna_begin_token_id)
                    if return_token_mask:
                        token_mask.append(0)  # DNA special token
                
                # Process DNA sequence if there's content
                if dna_content:
                    result = self._process_dna_sequence(dna_content)
                    
                    # Add DNA kmer tokens
                    for idx, kmer_token in enumerate(result["kmer_tokens"]):
                        token_id = self.dna_token_to_id.get(kmer_token, self.oov_token_id)
                        token_ids.append(token_id)
                        
                        if return_token_mask:
                            if kmer_token == "<oov>":
                                token_mask.append(0)  # OOV special token
                            elif idx == len(result["kmer_tokens"]) - 1 and result["padding_length"] > 0:
                                # Last token with padding
                                valid_length = result["valid_length"]
                                token_mask.append(valid_length)
                            else:
                                # Regular full kmer token
                                token_mask.append(self.k)
                
                # Add end token if there's an end tag
                if has_end_tag:
                    token_ids.append(self.dna_end_token_id)
                    if return_token_mask:
                        token_mask.append(0)  # DNA special token
            else:
                # Use base tokenizer for normal text
                base_ids = self.base_tokenizer.encode(
                    segment_content,
                    add_special_tokens=False,
                    **kwargs
                )
                token_ids.extend(base_ids)
                if return_token_mask:
                    token_mask.extend([-1] * len(base_ids))  # Natural language tokens
        
        # Add EOS token if needed
        if add_special_tokens and self.eos_token_id is not None:
            token_ids.append(self.eos_token_id)
            if return_token_mask:
                token_mask.append(-1)  # EOS token also treated as natural language
        
        # Handle truncation
        if truncation:
            max_len = max_length or self.model_max_length
            if self.truncation_side == "left":
                if len(token_ids) > max_len:
                    token_ids = token_ids[-max_len:]
                    if return_token_mask:
                        token_mask = token_mask[-max_len:] if token_mask else None
            else:  # right
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                    if return_token_mask:
                        token_mask = token_mask[:max_len] if token_mask else None
        
        if return_token_mask:
            return token_ids, token_mask
        return token_ids
    
    def decode(
        self,
        token_ids: Union[int, List[int], "torch.Tensor", "np.ndarray"],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = True,
        **kwargs
    ) -> str:
        """Decode token IDs, preserving proper spacing."""
        # Convert token_ids to list
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.tolist()
        elif isinstance(token_ids, np.ndarray):
            ids = token_ids.tolist()
        elif isinstance(token_ids, (list, tuple)):
            ids = list(token_ids)
        else:
            ids = [token_ids]
        
        # Filter special tokens
        if skip_special_tokens:
            special_tokens_to_skip = [self.bos_token_id, self.eos_token_id, self.pad_token_id]
            ids = [tid for tid in ids if tid not in special_tokens_to_skip]
        
        text_parts = []
        i = 0
        
        # Helper: check if we need a space before next content
        def needs_space_before(parts, next_text=""):
            if not parts or not parts[-1]:
                return False
            last_char = parts[-1][-1]
            # If next_text is provided, check its first char too
            if next_text and next_text[0] != ' ':
                return last_char != ' '
            return last_char != ' '
        
        while i < len(ids):
            token_id = ids[i]
            
            if token_id == self.dna_begin_token_id:
                # Check if we need space before DNA region
                if needs_space_before(text_parts):
                    text_parts.append(' ')
                
                # Process DNA region
                dna_tokens = []
                i += 1
                
                while i < len(ids) and ids[i] != self.dna_end_token_id:
                    if ids[i] in self.dna_id_to_token:
                        dna_tokens.append(self.dna_id_to_token[ids[i]])
                    i += 1
                
                dna_sequence = ''.join(dna_tokens)
                
                if skip_special_tokens:
                    text_parts.append(dna_sequence)
                else:
                    text_parts.append(f"<dna>{dna_sequence}")
                    if i < len(ids) and ids[i] == self.dna_end_token_id:
                        text_parts.append("</dna>")
                        i += 1
                        
            elif token_id in self.dna_id_to_token or token_id == self.dna_end_token_id:
                # Handle standalone DNA tokens
                if not skip_special_tokens:
                    if needs_space_before(text_parts):
                        text_parts.append(' ')
                    
                    if token_id == self.dna_end_token_id:
                        text_parts.append("</dna>")
                    else:
                        text_parts.append(self.dna_id_to_token[token_id])
                i += 1
                
            else:
                # Collect text tokens
                regular_tokens = []
                while i < len(ids):
                    current_id = ids[i]
                    if (current_id == self.dna_begin_token_id or 
                        current_id == self.dna_end_token_id or 
                        current_id in self.dna_id_to_token):
                        break
                    regular_tokens.append(current_id)
                    i += 1
                
                if regular_tokens:
                    decoded_text = self.base_tokenizer.decode(
                        regular_tokens, 
                        clean_up_tokenization_spaces=False,
                        **kwargs
                    )
                    
                    # Check if we need space before this text
                    if needs_space_before(text_parts, decoded_text):
                        text_parts.append(' ')
                    
                    text_parts.append(decoded_text)
        
        result = ''.join(text_parts)
        
        if clean_up_tokenization_spaces:
            result = self.base_tokenizer.clean_up_tokenization(result)
        
        return result
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = False,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_mask: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main interface method, supports standard calling style.
        
        Returns:
            Dict with keys: input_ids, attention_mask, (token_mask if return_token_mask=True)
        """
        # Handle batch input
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        # Encode all texts
        all_input_ids = []
        all_token_masks = [] if return_token_mask else None
        
        for t in texts:
            if return_token_mask:
                input_ids, token_mask = self.encode(
                    t,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                    return_token_mask=True,
                    **kwargs
                )
                all_input_ids.append(input_ids)
                all_token_masks.append(token_mask)
            else:
                input_ids = self.encode(
                    t,
                    add_special_tokens=add_special_tokens,
                    truncation=truncation,
                    max_length=max_length,
                    return_token_mask=False,
                    **kwargs
                )
                all_input_ids.append(input_ids)
        
        # Determine actual max length to use
        if padding:
            # Longest sequence length
            target_length = max(len(ids) for ids in all_input_ids)
            if max_length is not None:
                # Use max_length if specified
                target_length = min(max_length, target_length)
            # Cannot exceed model_max_length
            if hasattr(self, 'model_max_length') and self.model_max_length:
                target_length = min(target_length, self.model_max_length)
        else:
            # No padding, no need for uniform length
            target_length = None
        
        # Handle padding and create attention masks
        padded_input_ids = []
        attention_masks = []
        padded_token_masks = [] if return_token_mask else None
        
        for idx, input_ids in enumerate(all_input_ids):
            current_len = len(input_ids)
            
            if padding and target_length is not None:
                # Need padding
                if current_len > target_length:
                    # Truncate if current length exceeds target
                    if self.truncation_side == "left":
                        input_ids = input_ids[-target_length:]
                        if return_token_mask:
                            token_mask = all_token_masks[idx][-target_length:] if all_token_masks else None
                    else:
                        input_ids = input_ids[:target_length]
                        if return_token_mask:
                            token_mask = all_token_masks[idx][:target_length] if all_token_masks else None
                    current_len = target_length
                    pad_len = 0
                else:
                    pad_len = target_length - current_len
                    if return_token_mask:
                        token_mask = all_token_masks[idx] if all_token_masks else None
                
                # Pad input_ids
                if pad_len > 0:
                    if self.padding_side == "left":
                        input_ids = [self.pad_token_id] * pad_len + input_ids
                        if return_token_mask:
                            # Use -2 for padding positions in token_mask (different from natural language -1)
                            token_mask = [-2] * pad_len + token_mask
                    else:
                        input_ids = input_ids + [self.pad_token_id] * pad_len
                        if return_token_mask:
                            token_mask = token_mask + [-2] * pad_len
                
                # Create attention mask
                # 0 for padding, 1 for actual content
                if self.padding_side == "left":
                    attention_mask = [0] * pad_len + [1] * current_len
                else:
                    attention_mask = [1] * current_len + [0] * pad_len
            else:
                # No padding, all content is valid
                attention_mask = [1] * current_len
                if return_token_mask:
                    token_mask = all_token_masks[idx] if all_token_masks else None
            
            padded_input_ids.append(input_ids)
            attention_masks.append(attention_mask)
            if return_token_mask and token_mask is not None:
                if padded_token_masks is not None:
                    padded_token_masks.append(token_mask)
        
        # Build result
        result = {
            "input_ids": padded_input_ids if is_batch else padded_input_ids[0],
            "attention_mask": attention_masks if is_batch else attention_masks[0]
        }
        
        if return_token_mask and padded_token_masks is not None:
            result["token_mask"] = padded_token_masks if is_batch else padded_token_masks[0]
        
        # Convert to tensor
        if return_tensors == "pt":
            if is_batch:
                result["input_ids"] = torch.tensor(result["input_ids"])
                result["attention_mask"] = torch.tensor(result["attention_mask"])
                if return_token_mask and "token_mask" in result:
                    result["token_mask"] = torch.tensor(result["token_mask"])
            else:
                result["input_ids"] = torch.tensor([result["input_ids"]])
                result["attention_mask"] = torch.tensor([result["attention_mask"]])
                if return_token_mask and "token_mask" in result:
                    result["token_mask"] = torch.tensor([result["token_mask"]])
        
        return result
    
    def batch_encode(
        self,
        texts: List[str],
        **kwargs
    ) -> List[List[int]]:
        """Batch encode (deprecated, use __call__ instead)."""
        return [self.encode(text, **kwargs) for text in texts]
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save base tokenizer
        self.base_tokenizer.save_pretrained(save_directory)
        
        # Save DNA configuration
        dna_config = {
            "k": self.k,
            "dna_start_id": self.dna_start_id,
            "dna_vocab_size": self.dna_vocab_size,
            "dna_special_tokens": self.dna_special_tokens,
            "dna_token_to_id": self.dna_token_to_id,
            "dna_id_to_token": {str(k): v for k, v in self.dna_id_to_token.items()},
            "config": {
                "vocab_size": self.vocab_size,
                "model_max_length": self.model_max_length,
                "padding_side": self.padding_side,
                "truncation_side": self.truncation_side
            }
        }
        
        config_path = os.path.join(save_directory, "dna_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(dna_config, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs):
        """Load tokenizer from saved directory."""
        base_tokenizer = AutoTokenizer.from_pretrained(save_directory, trust_remote_code=True)
        
        # Load DNA configuration
        config_path = os.path.join(save_directory, "dna_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            dna_config = json.load(f)
        
        # Create instance
        instance = cls.__new__(cls)
        
        # Set base tokenizer
        instance.base_tokenizer = base_tokenizer
        
        # Set basic attributes
        instance.bos_token_id = base_tokenizer.bos_token_id
        instance.eos_token_id = base_tokenizer.eos_token_id
        instance.pad_token_id = base_tokenizer.pad_token_id
        instance.unk_token_id = base_tokenizer.unk_token_id
        
        # Set DNA-related attributes
        instance.k = dna_config["k"]
        instance.dna_start_id = dna_config["dna_start_id"]
        instance.dna_vocab_size = dna_config["dna_vocab_size"]
        instance.dna_special_tokens = dna_config["dna_special_tokens"]
        instance.dna_token_to_id = dna_config["dna_token_to_id"]
        instance.dna_id_to_token = {int(k): v for k, v in dna_config["dna_id_to_token"].items()}
        
        # Set DNA special token IDs
        instance.dna_begin_token_id = instance.dna_token_to_id["<dna>"]
        instance.dna_end_token_id = instance.dna_token_to_id["</dna>"]
        instance.oov_token_id = instance.dna_token_to_id["<oov>"]
        
        # Extend vocabulary
        instance._extend_vocab()
        
        # Set other configurations
        config = dna_config.get("config", {})
        instance.model_max_length = config.get("model_max_length", 8192)
        instance.padding_side = config.get("padding_side", "right")
        instance.truncation_side = config.get("truncation_side", "right")
        
        return instance