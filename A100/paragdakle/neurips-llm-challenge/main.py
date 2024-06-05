from fastapi import FastAPI

import logging

# Lit-GPT imports
import sys
from pathlib import Path
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

from handler import ModelHandler
from typing import List, Dict

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

handler = ModelHandler(model_location="mistralai/Mistral-7B-v0.1",
                       bbq_adapter_path="paragdakle/mistral-7b-bbq-lora",
                       cnn_adapter_path="paragdakle/mistral-7b-cnn-daily-lora-rs-v2",
                       math_adapter_path="paragdakle/mistral-stem-lw-lora",
                       routing_model_path="routing")

def truncate_sequence(sequence, request):
    """
    Certain providers have bugs where they aren't respecting max_tokens,
    stop_sequences and the end of text token, so as a hack, we have to manually
    truncate the suffix of `sequence` and `tokens` as a post-hoc process.
    """
    # TODO: if echo_prompt, then we should only ignore the prompt, but we don't
    # know how many tokens the prompt takes up.
    # In the benchmark, usually echo_prompt is only used for language modeling,
    # where max_tokens = 0, so there's nothing to truncate.
    if request['echo_prompt']:
        if request['max_new_tokens'] != 0:
            print("WARNING: don't know how to handle echo_prompt and max_tokens > 0, not truncating")
        return sequence

    for stop in request['stop_sequences']:
        # Find `stop` in the text
        try:
            new_text = sequence['text'][: sequence['text'].index(stop)]
        except ValueError:
            # The stop sequence doesn't exist, but it might exist in the list of tokens.
            new_text = sequence['text']

        # Strip `stop` off the tokens
        new_tokens = []
        # Need to start
        for token in sequence['tokens']:
            # Note: we can only strip at token boundaries
            if token['text'].startswith(stop):
                break
            new_tokens.append(token)

        if len(new_text) < len(sequence['text']) and len(new_tokens) == len(sequence['tokens']):
            print(
                f"WARNING: Stripped characters from text ({len(sequence['text'])} -> {len(new_text)}), "
                f"but wasn't able to strip the tokens"
            )

        # Recompute log probability
        new_logprob = sum(token['logprob'] for token in new_tokens)

        print(f"WARNING: truncate_sequence needs to strip {json.dumps(stop)}")

        sequence = {'text': new_text, 'logprob': new_logprob, 'tokens': new_tokens}

    # Truncate based on the max number of tokens.
    if len(sequence['tokens']) > request['max_new_tokens']:
        print(f"WARNING: truncate_sequence needs to truncate {len(sequence['tokens'])} down to {request['max_new_tokens']}")
        new_tokens = sequence['tokens'][: request['max_new_tokens']]

        # This is imperfect stitching together of tokens, so just to make sure this is okay
        # TODO: should use the proper detokenizer since T5-style models.
        # Usually, in our benchmark, max_tokens is active when it's 1, so hopefully this isn't an issue.
        new_text = "".join(token['text'] for token in new_tokens)
        if not sequence['text'].startswith(new_text):
            print(f"WARNING: {json.dumps(sequence.text)} does not start with truncated text {json.dumps(new_text)}")

        new_logprob = sum(token['logprob'] for token in new_tokens)

        sequence = {'text': new_text, 'logprob': new_logprob, 'tokens': new_tokens}

    return sequence

@app.post("/process")
async def process_request(input_data: ProcessRequest) -> Dict:
    # print(input_data)
    echo_prompt = False
    request = {
        "prompt": input_data.prompt,
        "temperature": input_data.temperature,
        "num_return_sequences": input_data.num_samples,
        "max_new_tokens": input_data.max_new_tokens,
        "top_k_per_token": input_data.top_k,
        "seed": input_data.seed,
        'echo_prompt': echo_prompt,
        'stop_sequences': [],
    }
    try:
        response = handler.pipeline_generate(request)
        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens = []

            if echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append({'text': token_text, 'logprob': 0.0, 'top_logprobs':{}})
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                top_logprobs_dict = {k: v for k, v in top_logprobs_dict.items() if v != float("-Inf")}
                tokens.append(Token(text=token_text, logprob=logprob, top_logprob=top_logprobs_dict))
                sequence_logprob += logprob

            completion = {
                'text': raw_completion["text"],
                'logprob': sequence_logprob,
                'tokens': tokens,
                'request_time': response['request_time']
            }
            completion = truncate_sequence(completion, request)
            completions.append(ProcessResponse(
                text=completion['text'], tokens=completion['tokens'], logprob=completion['logprob'], request_time=response['request_time']
            ))
        return completions[0]
    except Exception as e:
        print("Got an exception ", e)
    return {}


@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> Dict:
    request = {
        "prompt": input_data.text,
        "truncation": input_data.truncation,
        "max_length": input_data.max_length,
    }
    try:
        response = handler.tokenize(request)
        return TokenizeResponse(tokens=response['tokens'], request_time=response['request_time'])
    except Exception as e:
        print("Got an exception ", e)
    return {}
