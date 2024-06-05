import os
import sys
from pathlib import Path
from typing import Optional

import torch
from lightning_utilities.core.imports import RequirementCache

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

_SAFETENSORS_AVAILABLE = RequirementCache("safetensors")

def download_from_hub(
    repo_id: Optional[str] = None, directory: Path = Path("hf_model_test"), access_token: Optional[str] = os.getenv("HUGGINGFACE_TOKEN"), from_safetensors: bool = False
) -> None:
    if repo_id is None:
        from lit_gpt.config import configs

        options = [f"{config['org']}/{config['name']}" for config in configs]
        print("Please specify --repo_id <repo_id>. Available values:")
        print("\n".join(options))
        return

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=access_token,
    )

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)