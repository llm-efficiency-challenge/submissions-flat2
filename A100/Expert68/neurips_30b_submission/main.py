from fastapi import FastAPI, Depends

import logging
import os
import time
from threading import Lock
from llama_cpp import Llama
# import torch
# from huggingface_hub import login
# from transformers import LlamaTokenizer, LlamaForCausalLM
# from llama_recipes.inference.model_utils import load_peft_model

# torch.set_float32_matmul_precision("high")

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

# login(token=os.environ["HUGGINGFACE_TOKEN"])


LLAMA2_CONTEXT_LENGTH = 2048

llm = Llama(
    model_path="./llama-model.gguf",
    logits_all = True,
    n_ctx = LLAMA2_CONTEXT_LENGTH,
    n_gpu_layers = -1,
    n_threads = 10,
)


llama_outer_lock = Lock()
llama_inner_lock = Lock()


def get_llama():
    # NOTE: This double lock allows the currently streaming llama model to
    # check if any other requests are pending in the same thread and cancel
    # the stream if so.
    llama_outer_lock.acquire()
    release_outer_lock = True
    try:
        llama_inner_lock.acquire()
        try:
            llama_outer_lock.release()
            release_outer_lock = False
            yield llm
        finally:
            llama_inner_lock.release()
    finally:
        if release_outer_lock:
            llama_outer_lock.release()


@app.post("/process")
async def process_request(
    input_data: ProcessRequest,
    llama: Llama = Depends(get_llama),
) -> ProcessResponse:
    
    t0 = time.perf_counter()

    output = llama.create_completion(
        prompt = input_data.prompt,
        max_tokens = input_data.max_new_tokens,
        temperature = input_data.temperature,
        logprobs = llama.n_ctx(),
        echo = input_data.echo_prompt,
        top_k = input_data.top_k,
    )
    
    t = time.perf_counter() - t0

    choice = output['choices'][0]
    text = choice['text']
    
    tokens = []
    logprob_sum = 0
    for token, lp, tlp in zip(choice['logprobs']['tokens'], choice['logprobs']['token_logprobs'], choice['logprobs']['top_logprobs']):
        tokens.append(
            Token(text=token, logprob=lp, top_logprob=tlp)
        )
        logprob_sum += lp
    
    return ProcessResponse(
        text=text, tokens=tokens, logprob=logprob_sum, request_time=t
    )



@app.post("/tokenize")
async def tokenize(
    input_data: TokenizeRequest,
    llama: Llama = Depends(get_llama),
) -> TokenizeResponse:

    t0 = time.perf_counter()
    tokens = llama.tokenize(
        input_data.text.encode('utf8')
    )
    t = time.perf_counter() - t0
    return TokenizeResponse(tokens=tokens, request_time=t)
