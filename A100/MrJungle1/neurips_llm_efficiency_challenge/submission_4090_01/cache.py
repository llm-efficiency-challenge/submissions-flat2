import os
from huggingface_hub import snapshot_download, login
login(token=os.environ["HUGGINGFACE_TOKEN"])
snapshot_download(repo_id=os.environ["HUGGINGFACE_BASE_REPO"], local_dir="/workspace/checkpoints", cache_dir="/workspace/cache")
snapshot_download(repo_id=os.environ["HUGGINGFACE_ADAPTER_REPO"], local_dir="/workspace/adapter", cache_dir="/workspace/cache")
