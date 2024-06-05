import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests
import torch
from torch.utils.data import random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt.tokenizer import Tokenizer


def prepare(
    destination_path: Path = Path("data/oasst1"),
    checkpoint_dir: Path = Path("checkpoints/meta-llama/Llama-2-7b-hf"),
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
    train_data_file_name: str = "oasst1_train.parquet",
    train_data_file_url: str = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/data/train-00000-of-00001-b42a775f407cee45.parquet",
    test_data_file_name: str = "oasst1_test.parquet",
    test_data_file_url: str = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
    ignore_index: int = -1,
    max_seq_length: Optional[int] = None,
) -> None:
    """Prepare the OASST1 dataset for instruction tuning.

    The output is a training and test dataset saved as `train.pt` and `test.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """
    if max_seq_length is None:
        with open(checkpoint_dir / "lit_config.json", "r", encoding="utf-8") as file:
            config = json.load(file)
            max_seq_length = config["block_size"]

    destination_path.mkdir(parents=True, exist_ok=True)

    # Train data prep
    train_data_file_path = destination_path / train_data_file_name
    print("Loading train data file...")
    download_if_missing(train_data_file_path, train_data_file_url)

    print("Reading train Parquet file...")
    table = pq.read_table(train_data_file_path)
    train_data = table.to_pandas()

    print("Filtering only english messages...")
    train_data = train_data[train_data["lang"] == "en"]
    print(f"train_data has {len(train_data):,} samples")

    # Test data prep
    test_data_file_path = destination_path / test_data_file_name
    print("Loading test data file...")
    download_if_missing(test_data_file_path, test_data_file_url)

    print("Reading test Parquet file...")
    table = pq.read_table(test_data_file_path)
    test_data = table.to_pandas()

    print("Filtering only english messages...")
    test_data = test_data[test_data["lang"] == "en"]
    print(f"test_data has {len(test_data):,} samples")

    print("Loading tokenizer...")
    tokenizer = Tokenizer(checkpoint_dir)

    print("Processing train split ...")
    train_set = []
    for row in tqdm(train_data.iterrows(), total=len(train_data)):
        if row[1]["role"] == "assistant":
            sample = prepare_sample(
                example=row[1],
                tokenizer=tokenizer,
                max_length=max_seq_length,
                mask_inputs=mask_inputs,
                ignore_index=ignore_index,
                data=train_data,
            )
            if sample is not None:
                train_set.append(sample)

    torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_set = []
    for row in tqdm(test_data.iterrows(), total=len(test_data)):
        if row[1]["role"] == "assistant":
            sample = prepare_sample(
                example=row[1],
                tokenizer=tokenizer,
                max_length=max_seq_length,
                mask_inputs=mask_inputs,
                ignore_index=ignore_index,
                data=test_data,
            )
            if sample is not None:
                test_set.append(sample)

    torch.save(test_set, destination_path / "test.pt")


def download_if_missing(file_path: Path, file_url: str):
    """Downloads the raw json data file and saves it in the given destination."""
    if file_path.exists() and file_path.stat().st_size > 0:
        return
    with open(file_path, "wb") as f:
        f.write(requests.get(file_url).content)


def prepare_sample(
    example: dict,
    tokenizer: Tokenizer,
    max_length: int,
    mask_inputs: bool,
    ignore_index: int,
    data,
):
    """Processes a single sample.

    Each sample in the dataset consists of:
    - message_id: A unique identifier for each message
    - parent_id: The unique identifier of the parent message this message is replying to
    - user_id: A unique identifier for each user (either assistant or prompter)
    - text: The content of the message
    - role: The role of the user who sent this message (either assistant or prompter)
    - lang: The language of the message

    This function processes this data to produce a prompt text and a label for supervised training.
    The prompt text is formed as a concatenation of all parent messages up to the root.
    The label/target is the same message but with the response attached.

    Finally, both the prompt and the label get tokenized. If desired, all tokens in the label that correspond to
    the original input prompt get masked out (default).
    """

    # Concatenate all parent messages to form the full prompt
    full_prompt = ""
    parent_id = example["parent_id"]
    while parent_id is not None:
        parent_message = find_message_by_id(parent_id, data)
        if parent_message.empty:
            break
        full_prompt = (
            parent_message["text"] + "\n" + full_prompt
        )  # concatenate with a new line
        parent_id = parent_message["parent_id"]
    if parent_message.empty:
        return None
    full_prompt = generate_prompt(full_prompt)
    full_prompt_and_response = full_prompt + "\n" + example["text"]
    encoded_full_prompt = tokenizer.encode(full_prompt, max_length=max_length)
    encoded_full_prompt_and_response = tokenizer.encode(
        full_prompt_and_response, eos=True, max_length=max_length
    )

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[: len(encoded_full_prompt)] = ignore_index

    return {
        **example,
        "input_ids": encoded_full_prompt_and_response,
        "input_ids_no_response": encoded_full_prompt,
        "labels": labels,
    }


def generate_prompt(full_prompt):
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\nYou are an assistant having a conversation, answer any question to the best of your ability.\n\n### Input:\n{full_prompt}\n\n### Response:"
    )


def find_message_by_id(message_id, data):
    """Find a message in the dataset by its ID."""
    filtered_data = data[data["message_id"] == message_id]
    return filtered_data if filtered_data.empty else filtered_data.iloc[0]


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
