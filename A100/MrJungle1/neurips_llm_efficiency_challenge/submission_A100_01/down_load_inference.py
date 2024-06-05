import os
from huggingface_hub import snapshot_download, login
login(token=os.environ["HUGGINGFACE_TOKEN"])
snapshot_download(repo_id=os.environ["HUGGINGFACE_REPO"], local_dir="/workspace/llama2-exl2-4.0bpw", cache_dir="/workspace/cache")
