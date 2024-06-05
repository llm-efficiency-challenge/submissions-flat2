#!/usr/bin/env python3
#coding: utf8
#author: Andy Wong
#usage:
# conda activate neurips2023-train_QLoRA
# python bedrock/framework/test2/llama2_inference.py

from absl import app
from absl import flags
import torch
import os
import glob
import json
import textwrap
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    # AutoModelForCausalLM,
    # AutoTokenizer,
    BitsAndBytesConfig,
    # HfArgumentParser,
    # TrainingArguments,
    # pipeline,
    # logging,
)
from transformers.models.llama import convert_llama_weights_to_hf
from bedrock.framework.test2.config import llama2_dir, special_tokens_dict, DEVICE_MAP
from bedrock.framework.test2.system_check import check_bf16_support

# NOTE:
# IMPORTANT:
# after installing transformer, need to modify a line of code, as seen on https://github.com/huggingface/transformers/pull/26102/files
# \\wsl.localhost/Ubuntu/home/ndeewong/miniconda3/envs/neurips2023-train_QLoRA/lib/python3.9/site-packages/transformers/modeling_utils.py

# =================================
# Convert llama weights to hf model
# =================================

# References
# 1. [hf transformers code] (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py)
# 2. [hf transformers doc] (https://huggingface.co/docs/transformers/main/model_doc/llama2)
# 3. [Colab turtorial] (https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=OJXpOgBFuSrc)

# Based on references, there are two solutions to missing padding token
# a) add_special_tokens({"pad_token":"<pad>"})
# b) set tokenizer.pad_token = tokenizer.eos_token
# [TODO] To investigate: hf transformers doc metioned that during model initialization, 
# we should set self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx)
# or self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=tokenizer.pad_token_id)
# However, setting this might cause pre-training parameter to disapear. Not sure how to implement this.

# Here are more resources for looking into the padding implementation.
# For now, I will simply follow the colab tutorial.
# 1. https://github.com/huggingface/transformers/issues/22312
# 2. https://github.com/huggingface/transformers/pull/25088
# 3. https://github.com/centerforaisafety/tdc2023-starter-kit/issues/8
# 4. https://github.com/huggingface/transformers/pull/26102/files

# # ---------------------------------------------------------------------------------------------------------------------------------

model_size_dict = {
    "7b":"7B",
    "7b-chat":"7Bf",
    "13b":"13B",
    "13b-chat":"13Bf",
    "70b":"70B",
    "70b-chat":"70Bf",
}

quantization_modes = ["4bit", "8bit"]

def convert_llama_to_hf(
    model_home_dir=llama2_dir,
    selected_model="7b",
    safe_serialization=True,
):
    spm_path = os.path.join(model_home_dir, "tokenizer.model")
    pth_path = os.path.join(model_home_dir, f"llama-2-{selected_model}")
    output_dir = os.path.join(model_home_dir, f"hf-llama-2-{selected_model}")
    convert_llama_weights_to_hf.write_model(
        model_path=output_dir,
        input_base_path=pth_path,
        model_size=model_size_dict[selected_model],
        safe_serialization=safe_serialization,
        tokenizer_path=spm_path,
    )


def path_finder(selected_model="7b"):
    hf_model_path = os.path.join(llama2_dir, f"hf-llama-2-{selected_model}")
    print("-"*20)
    print('hf_model path:\n',hf_model_path)
    print("-"*20)
    return(hf_model_path)


def load_hf_llama_tokenizer(tokenizer_path):
    # Alternative to transformers.AutoTokenizer.from_pretrained
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.add_special_tokens(special_tokens_dict) # Option s.
    tokenizer.pad_token = tokenizer.eos_token # Option b.
    tokenizer.padding_side = "right" #? left or right? some user used left TODO.
    print("-"*20)
    print("tokenizer loaded.")
    print("model provided tokenizer.eos_token:\n", tokenizer.eos_token)
    print("we defined tokenizer.pad_token as:\n", tokenizer.pad_token)
    print("-"*20)
    return tokenizer


def load_bnb_config(mode="4bit"):
    bnb_config = None
    if mode=="4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4", # "fp4"
            bnb_4bit_compute_dtype=torch.bfloat16, # torch.float16,
            bnb_4bit_use_double_quant=False,
        )
    elif mode=="8bit":
        raise NotImplementedError(f"Quantize mode ({mode}) yet to be tested.")
    return bnb_config


def load_hf_llama_model(
        model_path,
        tokenizer,
        bnb_config=None,
        torch_dtype=torch.bfloat16, # torch.float16
        device_map=DEVICE_MAP, # 'auto',

    ): 
    # alternative to transformers.AutoModelForCausalLM.from_pretrained
    if bnb_config is not None:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=8,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1 # NOTE: >1 == experimental feature on tensor parallelism rank.
    return model


def inference(
    prompt = 'Q: Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse\nA:',
    tokenizer = None,
    model = None,
    max_new_tokens=128,

):
    print("-"*20, "inference testing", "-"*20, "\n")
    print("Prompt:\n", prompt)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE_MAP[""])
    print("Prompt -> input_ids:\n", input_ids)
    generation_output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
    )
    response = tokenizer.decode(generation_output[0])
    print("Response:\n", response)
    print("\n","-"*50)


SELECTED_MODEL = flags.DEFINE_enum(
    'selected_model',
    default="7b",
    enum_values=list(model_size_dict.keys()),
    help=
    (f"Which llama2 to choose from {model_size_dict.keys()}?"),
)

QUANTIZED_MODE = flags.DEFINE_enum(
    'quantized_mode',
    default="4bit",
    enum_values=quantization_modes,
    help=
    (f"Which mode to choose from {quantization_modes}?"),
)

def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    check_bf16_support()
    hf_model_path = path_finder(selected_model=SELECTED_MODEL.value)
    if len(glob.glob(os.path.join(hf_model_path, "model-*.safetensors")))<1:
        print("-"*20)
        print(f"Missing hf version of llama-2-{SELECTED_MODEL.value}. Execute model conversion")
        convert_llama_to_hf(
            model_home_dir=os.path.dirname(hf_model_path),
            selected_model=SELECTED_MODEL.value,
            safe_serialization=True,
        )
        print("-"*20)

    if len(glob.glob(os.path.join(hf_model_path, "model-*.safetensors")))<1:
        raise ValueError(f"hf version of llama-2-{SELECTED_MODEL.value} is still missing. Please toubleshoot.")
    
    tokenizer = load_hf_llama_tokenizer(hf_model_path)
    bnb_config = load_bnb_config(mode=QUANTIZED_MODE.value)
    model = load_hf_llama_model(
        hf_model_path,
        tokenizer,
        bnb_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE_MAP,
    )
    print(f"load model (quantization mode: {QUANTIZED_MODE.value}) sucess!")

    inference(
        prompt = 'Q: Create a detailed description for the following product: Corelogic Smooth Mouse, belonging to category: Optical Mouse\nA:',
        tokenizer = tokenizer,
        model = model,
        max_new_tokens=128,
    )

    # TODO: figure out batch processing.


if __name__ == '__main__':
  app.run(main)