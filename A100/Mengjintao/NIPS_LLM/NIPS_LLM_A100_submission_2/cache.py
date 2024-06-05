import uvicorn
import argparse
from fastapi import FastAPI

import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig

torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

app = FastAPI()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
total_time = 0
decode_time = 0
print (type(total_time))

QA_hashmap = {} 


def main(args) -> None:
#    model_id = args.checkpoint_dir
    model_id = "meta-llama/Llama-2-70b-hf"
    
    print("base model load...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
#        rope_scaling={"type": "dynamic", "factor": 2},
        load_in_4bit=True,
#       device_map="auto",
#        bnb_4bit_quant_type="nf4",
#        bnb_4bit_use_double_quant=True,
#        bnb_4bit_compute_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    print (model.get_memory_footprint())
    print (model.device)

    LLAMA2_CONTEXT_LENGTH = 2548

