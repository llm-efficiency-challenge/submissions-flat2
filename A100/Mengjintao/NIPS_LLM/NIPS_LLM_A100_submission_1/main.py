import uvicorn
import argparse
from fastapi import FastAPI

import logging
import os
import time

import torch
from transformers import GPTQConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
total_time = 0 


def main(args) -> None:
    
    model_path = args.checkpoint_dir
    adapter_path = args.lora_path
    quantize = args.quantize
    model_path = "/data_test/LLM/huggingface/checkpoints/Qwen-14B"
    quantize = '16'

    print("base model load...")
    if quantize == '16':
        model = AutoModelForCausalLM.from_pretrained(model_path,return_dict=True,device_map="auto",low_cpu_mem_usage=True,torch_dtype=torch.float16,trust_remote_code=True)
    else :
        if quantize == '4':
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif quantize == '8':
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantize == 'gptq':
            quantization_config = GPTQConfig(bits=4, disable_exllama=False, use_cuda_fp16=True)
        else:
            raise ValueError(f'quantize just in [4,8,16,gptq]')
        model = AutoModelForCausalLM.from_pretrained(model_path,return_dict=True,quantization_config=quantization_config,device_map="auto",low_cpu_mem_usage=True,torch_dtype=torch.float16,trust_remote_code=True)


    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)

    LLAMA2_CONTEXT_LENGTH = 4096


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
                do_sample=False,
                temperature=input_data.temperature,
                top_k=input_data.top_k,
                return_dict_in_generate=True,
                output_scores=True,
            )
        

        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
        output = output.split('$', 1)[0]
        output = output.split("\n\n")[0]

        t = time.perf_counter() - t0
        tokens_generated = outputs.sequences[0].size(0) - prompt_length
        logger.info(f"Input token length: {prompt_length, }, max_returned_tokens: {max_returned_tokens,}, tokens_generated: {tokens_generated,}")
        logger.info(
            f"Time for inference: {t:.02f} sec,  {tokens_generated / t:.02f} tokens/sec"
        )

        global total_time
        total_time = total_time + t
        logger.info(f"Time for inference: {t:.02f} sec total, Accumated time: {total_time:.02f} sec")

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

