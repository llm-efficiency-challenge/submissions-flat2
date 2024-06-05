from huggingface_hub import snapshot_download, login
login(token="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")
snapshot_download(repo_id="Qwen/Qwen-14B", local_dir="/workspace/Qwen-14B", cache_dir="/workspace/cache")
