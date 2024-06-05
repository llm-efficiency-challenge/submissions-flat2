import uvicorn
import argparse
from fastapi import FastAPI

import logging
import time
from typing import List, Union

import torch
from transformers import GPTQConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig
# from peft import AutoPeftModelForCausalLM

torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)
from typing import Dict, List
import re
import random
app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)


ASIAN_NAMES: List[str] = [
    "cho",
    "wong",
    "tang",
    "huang",
    "chu",
    "chung",
    "ng",
    "wu",
    "liu",
    "chen",
    "lin",
    "yang",
    "kim",
    "chang",
    "shah",
    "wang",
    "li",
    "khan",
    "singh",
    "hong",
]

HISPANIC_NAMES: List[str] = [
    "castillo",
    "gomez",
    "soto",
    "gonzalez",
    "sanchez",
    "rivera",
    "martinez",
    "torres",
    "rodriguez",
    "perez",
    "lopez",
    "medina",
    "diaz",
    "garcia",
    "castro",
    "cruz",
]

WHITE_NAMES: List[str] = [
    "harris",
    "nelson",
    "robinson",
    "thompson",
    "moore",
    "wright",
    "anderson",
    "clark",
    "jackson",
    "taylor",
    "scott",
    "davis",
    "allen",
    "adams",
    "lewis",
    "williams",
    "jones",
    "wilson",
    "martin",
    "johnson",
]

def bias_adjustment_gender(prompt):
    replacements = {
        r'\b(he|she)\b': 'he or she',
        r'\b(his|hers)\b': 'his or hers',
        r'\b(him|her)\b': 'him or her',
        r'\b(himself|herself)\b': 'himself or herself',
        r'\b(man|woman)\b': 'man or woman',
        r'\b(men|women)\b': 'men or women',
        r'\b(daughter|son)\b': 'son or daughter',
        r'\b(daughters|sons)\b': 'sons or daughters',
        r'\b(mother|father)\b': 'father or mother',
        r'\b(mothers|fathers)\b': 'fathers or mothers',
        r'\b(girl|boy)\b': 'girl or boy',
        r'\b(girls|boys)\b': 'girls or boys',
        r'\b(female|male)\b': 'male or female',
        r'\b(femen|males)\b': 'males or femen',
        r'\b(sister|brother)\b': 'brother or sister',
        r'\b(sisters|brothers)\b': 'brothers or sisters',
        r'\b(aunt|uncle)\b': 'uncle or aunt',
        r'\b(aunts|uncles)\b': 'uncles or aunts',
        r'\b(niece|nephew)\b': 'nephew or niece',
        r'\b(nieces|nephews)\b': 'nephews or nieces',
    }
    for word, replacement in replacements.items():
        prompt = re.sub(word, replacement, prompt)
    return prompt

def replace_name(match):
    name = match.group(0)
    if name.lower() in ASIAN_NAMES:
        random_names = random.choice(HISPANIC_NAMES), random.choice(WHITE_NAMES)
    elif name.lower() in HISPANIC_NAMES:
        random_names = random.choice(WHITE_NAMES), random.choice(ASIAN_NAMES)
    elif name.lower() in WHITE_NAMES:
        random_names = random.choice(HISPANIC_NAMES), random.choice(ASIAN_NAMES)
    else:
        return name
    return f"{name} or {' or '.join(random_names)}"


def bias_adjustment_race(prompt):
    pattern = re.compile(r"\b(" + "|".join(ASIAN_NAMES + HISPANIC_NAMES + WHITE_NAMES) + r")\b", re.IGNORECASE)
    adjusted_prompt = re.sub(pattern, replace_name, prompt)
    return adjusted_prompt


def main(args) -> None:
    
    model_path = args.checkpoint_dir
    quantize = args.quantize
    activation_type = args.activation_type
    bigbench_adapter = args.bigbench_adapter
    cnndm_adapter = args.cnndm_adapter
    gsm_adapter = args.gsm_adapter
    default_adapter = args.default_adapter

    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    
    print(f"loading base model in {model_path}")

    if quantize == 'gptq':
        # quantization_config = GPTQConfig(bits=4, disable_exllama=False, use_cuda_fp16=True)
        model = AutoModelForCausalLM.from_pretrained(model_path,return_dict=True,device_map="auto",trust_remote_code=True,low_cpu_mem_usage=True)
        # model = AutoModelForCausalLM.from_pretrained(model_path,return_dict=True,quantization_config=quantization_config,device_map="auto",trust_remote_code=True,low_cpu_mem_usage=True,)
    else :
        is_qwen = tokenizer.__class__.__name__ == 'QWenTokenizer'
        is_use_fp16 = activation_type == 'fp16'
        is_use_bf16 = activation_type == 'bf16'
        torch_dtype = torch.bfloat16 if activation_type =='bf16' else torch.float16
        
        kwargs = {}
        kwargs["device_map"] = "auto"
        if quantize == '4':
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype
            )
        elif quantize == '8':
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantize != '16':
            raise ValueError(f'quantize just in [4,8,16,gptq]')
        
        if is_qwen:
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                            fp16=is_use_fp16,
                                                            bf16=is_use_bf16,
                                                            return_dict=True,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=torch_dtype,
                                                            trust_remote_code=True,
                                                            **kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_path,
                                                            return_dict=True,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=torch_dtype,
                                                            trust_remote_code=True,
                                                            **kwargs)

    print(f"loading bigbench_adapter in {bigbench_adapter}")
    model.load_adapter(bigbench_adapter, adapter_name="bigbench_adapter")

    print(f"loading cnndm_adapter in {cnndm_adapter}")
    model.load_adapter(cnndm_adapter, adapter_name="cnndm_adapter")

    print(f"loading gsm_adapter in {gsm_adapter}")
    model.load_adapter(gsm_adapter, adapter_name="gsm_adapter")

    print(f"loading default_adapter in {default_adapter}")
    model.load_adapter(default_adapter, adapter_name="default_adapter")
    

    MAP_TASK_TYPE_TO_ADAPTER = {
        
        'gsm': 'gsm_adapter',
        'mmlu': 'base',
        'cnn': 'cnndm_adapter',
        'bbq': 'default_adapter',
        'big_bench': 'bigbench_adapter', 
        'zero_shot': 'default_adapter',
        'truthful_qa': 'default_adapter',
        'unknown': 'base',
    }

    model.eval()

    LLAMA2_CONTEXT_LENGTH = 4096


    @app.post("/process")
    async def process_request(input_data: ProcessRequest) -> ProcessResponse:
        
        if input_data.seed is not None:
            torch.manual_seed(input_data.seed)
            random.seed(input_data.seed)
        if "Summarize the above article in 3 sentences." in input_data.prompt:
            input_data.prompt = input_data.prompt.replace("Summarize the above article in 3 sentences.", "Summarize the above article in 3 sentences:")
            input_data.prompt = bias_adjustment_gender(input_data.prompt)
            input_data.prompt = bias_adjustment_race(input_data.prompt)
            
        from check_dataset import check_dataset
        task_type = check_dataset(input_data)
        adapter_name = MAP_TASK_TYPE_TO_ADAPTER[task_type]
        model.enable_adapters()
        if adapter_name == 'base':
            print("[router] disable adapter")
            model.disable_adapters()
        else:
            print("[router] use adapter: {}".format(adapter_name))
            model.set_adapter(adapter_name)
        if task_type == "zero_shot":
            input_data.prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n" + input_data.prompt + "\n\n### Response:\n"
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
                # do_sample=True,
                temperature=input_data.temperature,
                top_k=input_data.top_k,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        t = time.perf_counter() - t0

        output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
        output = output.split('$', 1)[0]
        output = output.split("\n\n")[0]
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI Application")
    # service pramemters
    parser.add_argument("--host", default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", default=80, type=int, help="端口号")
    # model pramemters
    parser.add_argument("--checkpoint_dir", default="/workspace/checkpoints", help="模型检查点目录")
    parser.add_argument("--quantize", default='16',type=str, help="量化类型")
    parser.add_argument("--activation_type", default='bf16',type=str, help="激活值类型")
    # adapater pramemters 
    parser.add_argument("--cnndm_adapter", type=str, default="/workspace/adapter/cnndm_adapter", help="lora finetuned文件")
    parser.add_argument("--gsm_adapter", type=str, default="/workspace/adapter/gsm_adapter", help="lora finetuned文件")
    parser.add_argument("--default_adapter", type=str, default="/workspace/adapter/default_adapter", help="lora finetuned文件")
    parser.add_argument("--bigbench_adapter", type=str, default="/workspace/adapter/bigbench_adapter", help="lora finetuned文件")
    args = parser.parse_args() 
    main(args)
    uvicorn.run(app, host=args.host, port=args.port)

