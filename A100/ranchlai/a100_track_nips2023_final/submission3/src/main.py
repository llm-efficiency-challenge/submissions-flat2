from fastapi import FastAPI
import logging
import time
from peft import PeftModel
from auto_gptq import exllama_set_max_input_length
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import GPTQConfig
from prompting import Prompt

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
    DecodeRequest,
    DecodeResponse,
    ChatRequest,
    ChatResponse,
)

app = FastAPI()
logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)
# no need to login
# login(token=os.environ["HUGGINGFACE_TOKEN"])
torch.set_float32_matmul_precision("high")

base_model_for_peft = AutoModelForCausalLM.from_pretrained(
    "/workspace/llama2-70b-gptq-4bit",
    use_safetensors=False,
    trust_remote_code=True,
    device_map={"": 0},
    torch_dtype=torch.float16,
    quantization_config=GPTQConfig(disable_exllama=False, use_cuda_fp16=True, bits=4),
)
model = PeftModel.from_pretrained(
    base_model_for_peft, "/workspace/llama2-lora", device_map={"": 0}
)
model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(
    "/workspace/llama2-70b-gptq-4bit",
    add_special_tokens=True,
    trust_remote_code=True,
    use_fast=True,
)

# model = exllama_set_max_input_length(model, 4096)
model.eval()
prompt = Prompt(specialization="auto", safty_level=0, format="simple")
LLAMA2_CONTEXT_LENGTH = 2048


def extract_answer(prompt: str, output: str):
    logger.info("========> extract answer from output: =>" + output)
    if output.endswith("."):
        output = output[:-1]
    # if A. B. C. in prompt, its a multiple choice question
    if "A. " in prompt and "B. " in prompt:
        if len(output) > 0 and output.split()[-1] in "ABCD":
            return output.split()[-1]
        # if any of A, B, C, D in output, return it
        if " A" in output:
            return "A"
        if " B" in output:
            return "B"
        if " C" in output:
            return "C"
        if " D" in output:
            return "D"
        if " E" in output and "E. " in prompt:
            return "E"
        if " F" in output and "F. " in prompt:
            return "F"
        if " G" in output and "G. " in prompt:
            return "G"

    if " answer is " in output:
        answer = output.split(" answer is ")[-1]
        return answer.strip()

    logger.info("========> this is output: \n" + output)
    return output


@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    text = input_data.prompt
    text = text.strip()

    # # If it is CNN, we don't need few shot. Will do it only in zero shot to save time.
    # if "Article:" in text and "Summarize the above" in text:
    #     text_ = "Article:" + text.split("Article:")[-1]
    #     text = text_
    # elif "Q: " in text and "A: " in text and "The answer is " in text:
    #     text_ = "Q: " + text.split("Q: ")[-1]
    #     text = text_

    text, _, _ = prompt.format_prompt(instruction=text, output="")

    encoded = tokenizer(text, return_tensors="pt")
    prompt_length = encoded["input_ids"][0].size(0)

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    if input_data.temperature < 1e-5:
        do_sample = False
    else:
        do_sample = True
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=do_sample,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    t = time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(
            outputs.sequences[0][prompt_length:], skip_special_tokens=True
        )
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    output = extract_answer(text, output)

    # This is to add a period at the end of the answer, for HELM GSM8K evaluation.
    # If no period, HELM will return 0 score (but the answer is correct)
    if "Q: " in text and "A:" in text and (not output.endswith(".")):
        output = "The answer is " + output + "."

    # ?
    if "Article:" in text and "Summarize the above" in text:
        if not output.endswith(" ."):
            output += " ."

    tokens_generated = outputs.sequences[0].size(0) - prompt_length
    logger.info(
        f"Time for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec"  # noqa
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
    decoded = tokenizer.decode(torch.tensor(input_data.tokens, dtype=torch.long))
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)


# additional test interface
@app.post("/chat")
async def chat_request(input_data: ChatRequest) -> ChatResponse:
    if input_data.seed is not None:
        torch.manual_seed(input_data.seed)
    text = input_data.prompt
    text = text.strip()

    text, _, _ = prompt.format_prompt(instruction=text, output="")

    encoded = tokenizer(text, return_tensors="pt")
    prompt_length = encoded["input_ids"][0].size(0)

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    if input_data.temperature < 1e-5:
        do_sample = False
    else:
        do_sample = True
    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=input_data.max_new_tokens,
            do_sample=do_sample,
            temperature=input_data.temperature,
            top_k=input_data.top_k,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    time.perf_counter() - t0
    if not input_data.echo_prompt:
        output = tokenizer.decode(
            outputs.sequences[0][prompt_length:], skip_special_tokens=True
        )
    else:
        output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    return ChatResponse(text=output)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
