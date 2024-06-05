"""
Helpers to support streaming generate output.
Borrowed from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/callbacks.py
"""

import gc
import traceback
from queue import Queue
from threading import Thread
from typing import Dict

import datasets
import torch
import transformers


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True


def get_oasst_dataset(path):
    from datasets import load_dataset
    import pandas as pd
    import os
    from datasets import Dataset


    os.makedirs(path, exist_ok=True)
    ds = load_dataset("OpenAssistant/oasst1")
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()

    df = pd.concat([train, val], axis=0).reset_index(drop=True)

    df_assistant = df[(df.role == "assistant")].copy()
    df_prompter = df[(df.role == "prompter")].copy()
    df_prompter = df_prompter.set_index("message_id")
    df_assistant["output"] = df_assistant["text"].values

    inputs = []
    parent_ids = []
    for _, row in df_assistant.iterrows():
        input = df_prompter.loc[row.parent_id]
        inputs.append(input.text)
        parent_ids.append(input.parent_id)

    df_assistant["instruction"] = inputs
    df_assistant["parent_id"] = parent_ids

    df_assistant = df_assistant[
        ["instruction", "output", "message_id", "parent_id", "lang", "rank"]
    ].rename(columns={"message_id": "id"})

    df_assistant[(df_assistant["rank"] == 0.0) & (df_assistant["lang"] == "en")][
        ["instruction", "output", "id", "parent_id"]
    ].to_parquet(os.path.join(path, "train_full.pq"), index=False)

    df_assistant[df_assistant["lang"] == "en"][
        ["instruction", "output", "id", "parent_id"]
    ].to_parquet(os.path.join(path, "train_full_allrank.pq"), index=False)

    df_assistant[df_assistant["rank"] == 0.0][
        ["instruction", "output", "id", "parent_id"]
    ].to_parquet(os.path.join(path, "train_full_multilang.pq"), index=False)

    df_assistant[["instruction", "output", "id", "parent_id"]].to_parquet(
        os.path.join(path, "train_full_multilang_allrank.pq"), index=False
    )

    data = df_assistant[(df_assistant["rank"] == 0.0) & (df_assistant["lang"] == "en")]

    data = Dataset.from_pandas(data)
    def add_input(example):
        example['input'] = ''
        return example
    data = data.map(add_input, num_proc=96)

    data = data.train_test_split(test_size=0.01)

    return data


def get_dolly_dataset(path):

    from datasets import load_dataset

    data = load_dataset("databricks/databricks-dolly-15k",
                        cache_dir=path)

    data = data.rename_column('context', "input")
    data = data.rename_column('response', "output")


    return data


def get_flan21_dataset(path):

    from datasets import load_dataset

    #data = load_dataset(path=path)

    data =  load_dataset("SirNeural/flan_v2", cache_dir=path)
    data = data.filter(lambda example: example['task'].startswith('cot'))
    data = data.rename_column('inputs', "instruction")
    data = data.rename_column('targets', "output")
    #data = data.remove_columns(['task_name'])

    def add_input(example):
        example['input'] = ''
        return example

    data = data.map(add_input, num_proc=96)

    return data

def mmlu():

    from datasets import load_dataset

    dataset_mmlu  = load_dataset("cais/mmlu", 'all')['auxiliary_train']
    dataset_mmlu = dataset_mmlu.shuffle(42)
    ans = {0: 'A',
           1: 'B',
           2: 'C',
           3: 'D'}

    def create_prompt(example):
        example['instruction'] = f"Question:{example['question']}, Choices:{example['choices']}."
        example['output'] = f"{ans[example['answer']]}"
        example['input'] = ''
        return example

    dataset_mmlu = dataset_mmlu.map(create_prompt, num_proc=96)
    dataset_mmlu = dataset_mmlu.remove_columns(['question', 'choices', 'answer', 'subject'])

    return  dataset_mmlu

def cnn():
    from datasets import load_dataset

    dataset_cnn = load_dataset("cnn_dailymail", '1.0.0')['train']
    dataset_cnn = dataset_cnn.shuffle(42)
    def create_prompt(example):
        example['instruction'] = f"{example['article']}"
        example['output'] = f"{example['highlights']} "
        example['input'] = ''
        return example

    dataset_cnn = dataset_cnn.map(create_prompt, num_proc=96)
    dataset_cnn = dataset_cnn.remove_columns(['id'])

    return  dataset_cnn
#
# def truthqa():
#     #!!!!!!!!!No this dataset I found shouldn't be used since
#     from datasets import load_dataset
#     dataset_truth = load_dataset("json", data_files='templates/finetune_truth.jsonl')['train']
#
#
#     dataset = load_dataset("truthful_qa")['validation']
#
#     def create_prompt(example):
#         example['instruction'] = f"{example['prompt']}"
#         example['output'] = f"{example['completion']} "
#         example['input'] = ''
#         return example
#     dataset_truth = dataset_truth.map(create_prompt, num_proc=96)
#
#     return dataset_truth

def gsm_8k():
    from datasets import load_dataset
    dataset_gsm = load_dataset("gsm8k", 'main')['train']

    def create_prompt(example):
        example['instruction'] = f"{example['question']}"
        example['output'] = f"{example['answer']} "
        example['input'] = ''
        return example

    dataset_gsm = dataset_gsm.map(create_prompt, num_proc=96)

    return dataset_gsm


def get_my_dataset(path):

    data = datasets.concatenate_datasets([gsm_8k()])
    data = data.train_test_split(0.0001)
    return data



def data_selection():
    """
    select data based on mix rate
    :return:
    """

    data = datasets.concatenate_datasets([mmlu(), gsm_8k()])
    data = data.train_test_split(0.0001)

    return data

