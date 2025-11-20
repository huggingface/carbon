from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceFW/fineweb-edu",
    allow_patterns="sample/10BT/*",
    local_dir="/fsx/leandro/data/fineweb-edu-10bt",
    repo_type="dataset"
)


snapshot_download(
    repo_id="HuggingFaceFW/finepdfs-edu",
    allow_patterns="data/eng_Latn/train/000_0000*",
    local_dir="/fsx/leandro/data/finepdf-edu-en10",
    repo_type="dataset"
)