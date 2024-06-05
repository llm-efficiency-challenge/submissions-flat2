from huggingface_hub import snapshot_download, login
login(token="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")
snapshot_download(repo_id="meta-llama/Llama-2-70b-hf", local_dir="/workspace/Llama-2-70b-hf", cache_dir="/workspace/cache", ignore_patterns=["pytorch_model*"], resume_download=True, max_workers=16, etag_timeout=100)
