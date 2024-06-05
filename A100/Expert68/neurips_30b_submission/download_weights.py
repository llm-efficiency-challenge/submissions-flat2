import os
import sys
import argparse
from pathlib import Path
from typing import Optional

wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def download_from_hub(repo_id: Optional[str] = None, access_token: Optional[str] = os.getenv("HF_TOKEN")) -> None:

    from huggingface_hub import snapshot_download

    directory = Path("checkpoints")
    snapshot_download(
        repo_id,
        local_dir=directory,
        local_dir_use_symlinks=False,
        resume_download=True,
        token=access_token,
        allow_patterns="ggml-model-q8_0-parts-*.gguf",
    )

parser = argparse.ArgumentParser()
parser.add_argument("--repo_id", type=str, default='Secbone/llama-33B-instructed',help="repo_id")
args = parser.parse_args()


if __name__ == "__main__":
    download_from_hub(args.repo_id)
