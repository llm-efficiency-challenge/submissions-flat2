# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets
import json

import torch
from torch.utils.data import Dataset
from llama_recipes.datasets.utils import Concatenator

MAX_WORDS = 1024


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, parsed_json, tokenizer, partition="train", max_words=30):
        self.ann = parsed_json
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:200]

        self.max_words = max_words
        # tokenizer = Tokenizer(model_path=model_path + "./tokenizer.model")
        self.tokenizer = tokenizer
        # self.tokenizer1 = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }


def get_custom_dataset(dataset_config, tokenizer, split):
    dataset = datasets.load_dataset("takojunior/llama_2_finetune_small", split=split, keep_default_na=False)
    df = dataset.to_pandas()
    df.fillna("", inplace=True)
    json_records = df.to_json(orient='records', indent=2)
    parsed = json.loads(json_records)

    return InstructionDataset(parsed, tokenizer, max_words=MAX_WORDS)

    # def apply_prompt_template(ann):
    #     if ann["input"] == "":
    #         prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
    #     else:
    #         prompt = PROMPT_DICT["prompt_input"].format_map(ann)
        
    #     example = prompt + ann["output"]

    #     return {
    #         "text": example
    #     }

    # dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    # dataset = dataset.map(
    #     lambda sample: tokenizer(sample["text"]),
    #     batched=True,
    #     remove_columns=list(dataset.features),
    # ).map(Concatenator(chunk_size=50), batched=True)

    # return dataset
