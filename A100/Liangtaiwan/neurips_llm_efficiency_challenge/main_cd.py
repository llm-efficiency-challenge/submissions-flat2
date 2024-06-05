from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lightning as L
import torch

torch.set_float32_matmul_precision("high")

from lit_gpt import GPT, Tokenizer, Config
from lit_gpt.utils import lazy_load, gptq_quantization, load_checkpoint

# Toy submission imports
from helper import contrastive_generate
from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

quantize = "bnb.nf4-dq"  # 4-bit NormalFloat with Double-Quantization (see QLoRA paper)
checkpoint_dir = Path("checkpoints/meta-llama/Llama-2-13b-hf")
st_checkpoint_dir = Path("checkpoints/princeton-nlp/Sheared-LLaMA-1.3B")
precision = "bf16-true"  # weights and data in bfloat16 precision

fabric = L.Fabric(devices=1, accelerator="cuda", precision=precision)

with open(checkpoint_dir / "lit_config.json") as fp:
    config = Config(**json.load(fp))

with open(st_checkpoint_dir / "lit_config.json") as fp:
    st_config = Config(**json.load(fp))

checkpoint_path = checkpoint_dir / "lit_model.pth"
st_checkpoint_path = st_checkpoint_dir / "lit_model.pth"
logger.info(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}")
with fabric.init_module(empty_init=True), gptq_quantization(quantize == "gptq.int4"):
    model = GPT(config)
    st_model = GPT(st_config)

# with lazy_load(checkpoint_path) as checkpoint:
#     model.load_state_dict(checkpoint, strict=quantize is None)

load_checkpoint(fabric, model, checkpoint_path)
load_checkpoint(fabric, st_model, st_checkpoint_path)

model.eval()
model = fabric.setup(model)
st_model.eval()
st_model = fabric.setup(st_model)

tokenizer = Tokenizer(checkpoint_dir)


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        L.seed_everything(input_data.seed)
    logger.info("Using device: {}".format(fabric.device))
    # input_data.prompt = generate_prompt(input_data.prompt)
    # input_data.prompt += "\n"
    encoded = tokenizer.encode(
        input_data.prompt, bos=True, eos=False, device=fabric.device
    )
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens

    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        model.set_kv_cache(batch_size=1)
        # set the max_seq_length to limit the memory usage to what we need
        st_model.max_seq_length = max_returned_tokens
        # enable the kv cache
        st_model.set_kv_cache(batch_size=1)


    t0 = time.perf_counter()
    # print(input_data)
    tokens, logprobs, top_logprobs = contrastive_generate(
        model,
        st_model,
        encoded,
        max_returned_tokens,
        temperature=input_data.temperature,
        st_temperature=0.5,
        top_k=input_data.top_k,
    )

    t = time.perf_counter() - t0

    if input_data.echo_prompt is False:
        output = tokenizer.decode(tokens[prompt_length:])
    else:
        output = tokenizer.decode(tokens)
    tokens_generated = tokens.size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    for t, lp, tlp in zip(tokens, logprobs, top_logprobs):
        idx, val = tlp
        tok_str = tokenizer.processor.decode([idx])
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
    logger.info("Using device: {}".format(fabric.device))
    t0 = time.perf_counter()
    encoded = tokenizer.encode(
        input_data.text, bos=True, eos=False, device=fabric.device
    )
    t = time.perf_counter() - t0
    tokens = encoded.tolist()
    return TokenizeResponse(tokens=tokens, request_time=t)


@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    logger.info("Using device: {}".format(fabric.device))
    t0 = time.perf_counter()
    # decoded = tokenizer.decode(torch.Tensor(input_data.tokens))
    decoded = tokenizer.processor.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)


def generate_prompt(text: str) -> str:
    """Generates a standardized message to prompt the model with an instruction, optional input and a
    'response' field."""

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{text}\n\n### Response:"
    )

