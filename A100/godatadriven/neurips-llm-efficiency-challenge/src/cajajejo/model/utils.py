import typing
import logging

try:
    from transformers import AutoTokenizer
except ImportError:
    _has_training_extras = False
else:
    _has_training_extras = True

from cajajejo.utils import requires_extras


logger = logging.getLogger("cajajejo.model.utils")


@requires_extras(_has_training_extras, "training")
def get_tokenizer(model: str, use_fast: bool):
    """Load a tokenizer for a pre-trained HF model"""
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        use_fast=use_fast,
    )
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    logger.debug(
        f"pre-trained model's BOS EOS and PAD token id: {bos}, {eos}, {pad} => It should be 1 2 None"
    )
    if model == "adept/persimmon-8b-base":
        tokenizer.eos_token_id = tokenizer.bos_token_id
        tokenizer.pad_token_id = tokenizer.bos_token_id
    else:
        tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    return tokenizer


# Taken from the lit-gpt repository
def generate_prompt(example: typing.Dict[str, str]) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    if example["input"]:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context "
            "and possibly several examples that illustrate the task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        )
    return (
        "Below is an instruction that describes a task and possibly several examples that illustrate the task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    )
