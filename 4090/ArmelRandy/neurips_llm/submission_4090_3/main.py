from fastapi import FastAPI
import logging
import time

# import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse
)

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)
# model_name = "mistralai/Mistral-7B-v0.1"
model_name = "ArmelRandy/Qwen-14b-mostfouren-45"
tokenizer_name = None
if "mistral" in model_name:
    tokenizer_name = "mistralai/Mistral-7B-v0.1"
elif "llama-2" in model_name:
    tokenizer_name = "NousResearch/Llama-2-13b-hf"
elif "llama" in model_name:
    tokenizer_name = "huggyllama/llama-7b"
elif "Qwen" in model_name:
    tokenizer_name = "Qwen/Qwen-14B"
else:
    tokenizer_name = model_name

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
).eval()

LLAMA2_CONTEXT_LENGTH = 2048
app = FastAPI()


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)

    encoded = tokenizer(input_data.prompt, return_tensors="pt")

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
            eos_token_id=tokenizer.eos_token_id,
        )

    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(
            outputs.sequences[0][prompt_length:], skip_special_tokens=True
        )
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print(output)
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []

    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1] :]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:, :, None]).squeeze(-1)
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
        text=output, tokens=generated_tokens, logprob=logprob_sum, request_time=t
    )


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
    decoded = tokenizer.decode(torch.LongTensor(input_data.tokens))
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)