from fastapi import FastAPI

import logging
import os
import time

import torch
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_recipes.inference.model_utils import load_peft_model

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
logging.basicConfig(level=logging.INFO, filename="./llama_7b_ft.log")

login(token=os.environ["HUGGINGFACE_TOKEN"])

LLAMA2_CONTEXT_LENGTH = 4096

model_name = 'meta-llama/Llama-2-7b-hf'
#fine_tuned_model_path = "./trained_models/Finetuned_Llama2-7B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

#  Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    return_dict=True,
    torch_dtype=torch.bfloat16,
    )

# Load fine-tuned model
model = load_peft_model(model, os.environ["HUGGINGFACE_REPO"])
#model = PeftModel.from_pretrained(
#    model=model, 
#    model_id=fine_tuned_model_path, 
#    device_map="auto"
#)

model.eval()

import re
def preprocess(texts):
    texts = [texts]
    pattern_remained_chars = r'[^A-Za-z0-9.,!?:;\'"\-\[\]{}()_$%&+\s]'
    pattern_symbol_seq     = r"([,!?:;'\-\[\]{}()_])\1{2,}"
    texts = [re.sub('<.*?>', ' ', t) for t in texts]               # Delete html tags
    texts = [re.sub('http\S+', ' ', t) for t in texts]             # Delete links
    texts = [re.sub(pattern_remained_chars, '', t) for t in texts] # Leave only alphanumeric characters, common symbols, any space character
    texts = [re.sub(pattern_symbol_seq, r"\1", t) for t in texts]  # Replace three or more repetitions of the same symbol with a single symbol
    texts = [re.sub('\s+', ' ', t) for t in texts]                 # Replace a sequence of whitespace with a single whitespace character
    texts = [t.strip() for t in texts]                     # strip only
    
    return texts[0]

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    
    # preprocess
    input_data.prompt = preprocess(input_data.prompt)
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
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,

            max_new_tokens=100,
            temperature=0.1,
            top_p=0.1,
            top_k=10,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    generated_tokens = []
    
    log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

    gen_sequences = outputs.sequences[:, encoded["input_ids"].shape[-1]:]
    gen_logprobs = torch.gather(log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    top_indices = torch.argmax(log_probs, dim=-1)
    top_logprobs = torch.gather(log_probs, 2, top_indices[:,:,None]).squeeze(-1)
    top_indices = top_indices.tolist()[0]
    top_logprobs = top_logprobs.tolist()[0]

    for t, lp, tlp in zip(gen_sequences.tolist()[0], gen_logprobs.tolist()[0], zip(top_indices, top_logprobs)):
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
    encoded = tokenizer(
        input_data.text
    )
    t = time.perf_counter() - t0
    tokens = encoded["input_ids"]
    return TokenizeResponse(tokens=tokens, request_time=t)