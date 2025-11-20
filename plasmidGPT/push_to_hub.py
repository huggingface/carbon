from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast
import torch

repo_id = "lingxusb/PlasmidGPT"
upload_id = "PlasmidGPT"
print("Uploading tokenizer...")
tokenizer_file_path = hf_hub_download(
    repo_id=repo_id,
    filename="addgene_trained_dna_tokenizer.json",
    repo_type="model"
)
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.push_to_hub(upload_id)


print("Uploading model...")
pt_file_path = hf_hub_download(
    repo_id=repo_id,
    filename="pretrained_model.pt",
    repo_type="model"
)

model = torch.load(pt_file_path, weights_only=False)
model.push_to_hub(upload_id)