from fastapi import FastAPI, HTTPException

import logging
import os
import time

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
from llama_recipes.inference.model_utils import load_peft_model
from pathlib import Path

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

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-hf')
tokenizer.truncation_side="left"

directory = Path("/workspace/checkpoint-llama2-70b")

model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-70b-hf',
    #'decapoda-research/llama-65b-hf',
    #'mistralai/Mistral-7B-v0.1',
    return_dict=True,
    quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ), 
    use_flash_attention_2=True,
    torch_dtype=torch.float16,
    force_download=True,
    resume_download=False,
    )
      
model = load_peft_model(model, os.environ["HUGGINGFACE_REPO"])

model.eval()
LLAMA2_CONTEXT_LENGTH = 4096


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
        
    generate_token_num = 1
    encoded = tokenizer(input_data.prompt, return_tensors="pt") # original
    
    # cutting
    if input_data.max_new_tokens == 1: #  BBQ, MMLU, TruthfulQA => 무조건 1개
        # generate token 설정
        generate_token_num = input_data.max_new_tokens
        # encoding engineering
        encoded = tokenizer(input_data.prompt, max_length=1408, return_tensors="pt")
        cut_encoded_str = tokenizer.decode(encoded['input_ids'][0])
        # 구분자로 찾기
        result_string = cut_encoded_str
        # BBQ -> "Passage:"와 "Question:" 둘다 소유
        if "Passage:" in cut_encoded_str and "Question:" in cut_encoded_str:
            start_index = cut_encoded_str.find("Passage:")
            if start_index != -1:
                result_string = "The following are multiple choice questions (with answers).\n \n" + cut_encoded_str[start_index:]
        # MMLU & Truthful -> onlt "Question:"
        elif "Passage:" not in cut_encoded_str and "Question:" in cut_encoded_str:
            start_index = cut_encoded_str.find("Question:")
            if start_index != -1:
                result_string = "The following are multiple choice questions (with answers).\n \n" + cut_encoded_str[start_index:]
        print(result_string)
        encoded = tokenizer(result_string, max_length=1408, return_tensors="pt")

    elif input_data.max_new_tokens > 1 and input_data.max_new_tokens <= 16 :
        # generate token 설정
        generate_token_num = input_data.max_new_tokens
        # encoding engineering
        encoded = tokenizer(input_data.prompt, max_length=841, return_tensors="pt")
    elif input_data.max_new_tokens > 16 and input_data.max_new_tokens <= 32 :
        encoded = tokenizer(input_data.prompt, max_length=822, return_tensors="pt")
        # generate token 설정
        generate_token_num = input_data.max_new_tokens
        # encoding engineering
    elif input_data.max_new_tokens > 32 and input_data.max_new_tokens <= 64 :
        encoded = tokenizer(input_data.prompt, max_length=822, return_tensors="pt")
        # generate token 설정
        generate_token_num = input_data.max_new_tokens
        # encoding engineering
    elif input_data.max_new_tokens > 64 and input_data.max_new_tokens < 128 :
        # generate token 설정
        generate_token_num = input_data.max_new_tokens
        # encoding engineering
        encoded = tokenizer(input_data.prompt, max_length=768, return_tensors="pt")
    elif input_data.max_new_tokens >= 128: # output token_num is 128 -> gsm and cnn/dm
        # generate token 설정
        generate_token_num = 128
        # encoding engineering
        encoded = tokenizer(input_data.prompt, max_length=768, return_tensors="pt")
        cut_encoded_str = tokenizer.decode(encoded['input_ids'][0])
        # 구분자로 찾기
        result_string = cut_encoded_str
        # CNN/DM -> "###"
        if "###" in cut_encoded_str:
            start_index = cut_encoded_str.find("###")
            if start_index != -1:
                result_string = cut_encoded_str[start_index:]
        # GSM -> "Q:"
        if "Q:":
            start_index = cut_encoded_str.find("Q:")
            if start_index != -1:
                result_string = cut_encoded_str[start_index:]
        encoded = tokenizer(result_string, max_length=768, return_tensors="pt")

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
            max_new_tokens=generate_token_num,
            do_sample=True,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
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
