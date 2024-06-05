from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse,
)
from fastapi import FastAPI

import logging
import os
import time

import torch
from huggingface_hub import login
from peft import PeftModel

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer


# torch.set_float32_matmul_precision("high")


app = FastAPI()

logger = logging.getLogger(__name__)

# Configure the logging module
logging.basicConfig(level=logging.INFO)

login(token=os.environ["HUGGINGFACE_TOKEN"])


base_model = os.environ["BASE_MODEL"]
lora_weights = os.environ["LORA_WEIGHTS"]
cache_dir = os.environ["BASE_MODEL_CACHE"]


print(f'load model from {base_model}')
print(f'load lora from {lora_weights}')

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    use_flash_attention_2=False,
    trust_remote_code=True,
    device_map='auto')

model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)

if 'Qwen' in base_model:
    tokenizer = AutoTokenizer.from_pretrained(lora_weights,
                                              pad_token='<|extra_0|>',
                                              trust_remote_code=True)
    # # qwen tokenizer hack
    tokenizer.eos_token_id = tokenizer.eod_id
    LLAMA2_CONTEXT_LENGTH = 2048
else:

    tokenizer = AutoTokenizer.from_pretrained(lora_weights,
                                              trust_remote_code=True)
    LLAMA2_CONTEXT_LENGTH = 4096

model.eval()



@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    encoded = tokenizer(input_data.prompt,
                        truncation=True,
                        max_length=LLAMA2_CONTEXT_LENGTH,
                        padding=False,
                        return_tensors="pt")

    prompt_length = encoded["input_ids"][0].size(0)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
        max_returned_tokens,
        LLAMA2_CONTEXT_LENGTH,
    )

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )

    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(
            outputs.sequences[0][prompt_length:], skip_special_tokens=True
        )
    else:
        output = tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True)

    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(
        f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []

    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(
        log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(
        log_probs, 2, top_indices[:, :, None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(
        gen_sequences.tolist()[0],
        gen_logprobs.tolist()[0],
        zip(top_indices, top_logprobs),
    ):
        idx, val = tlp
        tok_str = tokenizer.decode(idx)
        token_tlp = {tok_str: val}
        generated_tokens.append(
            Token(text=tokenizer.decode(t), logprob=lp, top_logprob=token_tlp)
        )
    logprob_sum = gen_logprobs.sum().item()

    return ProcessResponse(
        text=output,
        tokens=generated_tokens,
        logprob=logprob_sum,
        request_time=t)


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    t0 = time.perf_counter()
    encoded = tokenizer(input_data.text)
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)


@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    t0 = time.perf_counter()
    decoded = tokenizer.decode(torch.Tensor(input_data.tokens).to(torch.int64))
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)
