from fastapi import FastAPI

import logging

import sys
import time
from pathlib import Path
import json

from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
import torch

torch.set_float32_matmul_precision("high")

# Toy submission imports
from helper import response_generation
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

# load model
base_model = "meta-llama/Llama-2-7b-hf"
lora_weights = "thanhdath/llama2-self-instruct"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    ),
    torch_dtype=torch.float32,
    device_map="auto"
)
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float32
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        L.seed_everything(input_data.seed)

    t0 = time.perf_counter()
    tokens, logprobs, top_logprobs = response_generation(
        model,
        tokenizer,
        input_data.prompt,
        max_new_tokens=input_data.max_new_tokens,
        temperature=input_data.temperature,
        top_k=input_data.top_k
    )
    
    t = time.perf_counter() - t0

    output = tokenizer.decode(tokens)

    generated_tokens = []
    for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.decode([idx])
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprobs_sum = sum(logprobs)
    # Process the input data here
    return ProcessResponse(
        text=output, tokens=generated_tokens, logprob=logprobs_sum, request_time=t
    )


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    tokens = tokenizer.encode(input_data.text)
    t = time.perf_counter() - t0
    return TokenizeResponse(tokens=tokens, request_time=t)
