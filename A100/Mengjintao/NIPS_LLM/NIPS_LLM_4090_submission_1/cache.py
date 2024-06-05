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

#model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-70b-hf',return_dict=True,device_map="auto")
#model.eval()
#tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf')

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B", device_map="auto", trust_remote_code=True).eval()
