from fastapi import FastAPI

import logging
import os
import time

import torch
from huggingface_hub import login
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig

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
# Configure the logging module
logging.basicConfig(level=logging.INFO)

login(token=os.environ["HUGGINGFACE_TOKEN"])

#model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-13b-hf',return_dict=True,device_map="auto")
#model.eval()
#tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-13b-hf')

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf')

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-70b-hf',
#   rope_scaling={"type": "dynamic", "factor": 2},
    load_in_4bit=True,
#       device_map="auto",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()
