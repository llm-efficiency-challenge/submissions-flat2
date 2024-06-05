import os
from pathlib import Path
from huggingface_hub import snapshot_download, login
 
login(token=os.environ["HUGGINGFACE_TOKEN"])

# 저장할 경로가 없으면 생성
if not os.path.exists('checkpoint-llama1-65b'):
    os.makedirs('checkpoint-llama1-65b')

repo_id = 'decapoda-research/llama-65b-hf'

download_files = ["tokenizer*", "config.json"]
#download_files.append("*.bin*")

directory = Path("checkpoint-llama1-65b")
snapshot_download(
    repo_id,
    local_dir=directory,
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=download_files,
)