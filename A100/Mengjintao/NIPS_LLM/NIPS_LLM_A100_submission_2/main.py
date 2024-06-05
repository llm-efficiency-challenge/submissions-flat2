import uvicorn
import argparse
from fastapi import FastAPI

import logging
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig

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
logging.basicConfig(level=logging.INFO)
total_time = 0
decode_time = 0
print (type(total_time))

QA_hashmap = {} 


def main(args) -> None:
#    model_id = args.checkpoint_dir
    model_id = "meta-llama/Llama-2-70b-hf"
    
    print("base model load...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
#        rope_scaling={"type": "dynamic", "factor": 2},
        load_in_4bit=True,
#       device_map="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.eval()

    print (model.get_memory_footprint())
    print (model.device)

    LLAMA2_CONTEXT_LENGTH = 2548

    @app.post("/process")
    async def process_request(input_data: ProcessRequest) -> ProcessResponse:
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
        
        t0 = time.perf_counter()

        inputs = tokenizer(input_data.prompt, return_tensors="pt")
        prompt_length = inputs["input_ids"][0].size(0)
        max_returned_tokens = prompt_length + input_data.max_new_tokens

        assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
            max_returned_tokens,
            LLAMA2_CONTEXT_LENGTH,
        )

        inputs = inputs.to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=False,
#           do_sample=False,
            temperature=input_data.temperature,
#            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        t1 = time.perf_counter() - t0

        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
        
        t = time.perf_counter() - t0

        output = output.split('$', 1)[0]
        output = output.split("\n\n")[0]
        tokens_generated = outputs.sequences[0].size(0) - prompt_length
        logger.info(f"Input token length: {prompt_length, }, max_returned_tokens: {max_returned_tokens,}, tokens_generated: {tokens_generated,}")
        logger.info(
            f"Time for inference: {t:.02f} sec, for decode: {t1:.02f} sec, {tokens_generated / t:.02f} tokens/sec"
        )

        global total_time
        global decode_time
        total_time = total_time + t
        logger.info(f"Time for inference: {t:.02f} sec total, Accumated time: {total_time:.02f} sec")

        logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
        generated_tokens = []
        
        log_probs = torch.log(torch.stack(outputs.scores, dim=1).softmax(-1))

        gen_sequences = outputs.sequences[:, inputs["input_ids"].shape[-1]:]
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

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description="FastAPI Application")
#    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
#    parser.add_argument("--port", default=8080, type=int, help="端口号")
#    parser.add_argument("--checkpoint_dir", default="/data_test/LLM/huggingface/checkpoints/Llama-2/meta-llama/Llama-2-7b-hf/", help="模型检查点目录")
#    args = parser.parse_args()
#    main(args)
#    uvicorn.run(app, host=args.host, port=args.port, worker=2)
