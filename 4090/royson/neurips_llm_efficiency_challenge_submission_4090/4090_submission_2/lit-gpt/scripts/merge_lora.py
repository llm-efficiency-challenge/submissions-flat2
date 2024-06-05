"""This script merges the LoRA weights with the base model"""

import sys
from pathlib import Path
from typing import Optional
import os

import lightning as L
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.lora import GPT, Config, lora_filter, merge_lora_weights
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, lazy_load

# lora_r = 1024
# lora_alpha = lora_r*2
lora_dropout = 0.05
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

def merge_lora(
    lora_path: Path = Path("out/lora/alpaca/lit_model_lora_finetuned.pth"),
    checkpoint_dir: Path = Path("checkpoints/stabilityai/stablelm-base-alpha-3b"),
    out_dir: Path = Path("out/lora/checkpoint"),
    precision: Optional[str] = None,
    lora_r: int = 256,
    all_lora_layers: bool = False,
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT-LoRA model.
    See `finetune/lora.py`.

    Args:
        lora_path: Path to the checkpoint with trained adapter weights, which are the output of
            `finetune/lora.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        out_dir: The path to the merged model that is created by this script.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)
    fabric = L.Fabric(devices=1, precision=precision)

    check_valid_checkpoint_dir(checkpoint_dir)

    os.makedirs(out_dir, exist_ok=True)

    if all_lora_layers:
        config = Config.from_json(
            checkpoint_dir / "lit_config.json",
            r=lora_r,
            alpha=lora_r * 2,
            dropout=lora_dropout,
            to_query=True,
            to_key=True,
            to_value=True,
            to_projection=True,
            to_mlp=True,
            to_head=True,
        )    
    else:
        config = Config.from_json(
            checkpoint_dir / "lit_config.json",
            r=lora_r,
            alpha=lora_r * 2,
            dropout=lora_dropout,
            to_query=lora_query,
            to_key=lora_key,
            to_value=lora_value,
            to_projection=lora_projection,
            to_mlp=lora_mlp,
            to_head=lora_head,
        )

    with fabric.init_module(empty_init=True):
        model = GPT(config)
    checkpoint_path = checkpoint_dir / "lit_model.pth"
    checkpoint = lazy_load(checkpoint_path)
    lora_checkpoint = lazy_load(lora_path)
    checkpoint.update(lora_checkpoint.get("model", lora_checkpoint))
    model.load_state_dict(checkpoint)

    merge_lora_weights(model)

    save_path = out_dir / "lit_model.pth"
    fabric.print(f"Saving weights to {str(save_path)!r}")
    # remove lora parameters and the lora linear substring
    state_dict = {k.replace("linear.", ""): v for k, v in model.state_dict().items() if not lora_filter(k, v)}
    torch.save(state_dict, save_path)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(merge_lora)