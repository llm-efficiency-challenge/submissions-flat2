from fastapi import FastAPI
from contextlib import asynccontextmanager

import logging

# Lit-GPT imports
import sys
import time
from pathlib import Path
from transformers import (BitsAndBytesConfig, AutoModelForCausalLM, 
                          AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, 
                          set_seed)
import torch


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


# get API schema and helper functions
from helper import get_model_and_tokenizer
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

llm = {} 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: change these to environment/config variables
    my_token = "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" # Andrey's token
    tuned_repo = "gromovand/L70_lc20_r8_lr1e-4_lastlayer_79"
    layer_dropping=True
    layers_to_drop = "20,59,79" 
    layer_norm=True

    tuned_model, tokenizer = get_model_and_tokenizer(
        tuned_repo,
        layer_dropping=layer_dropping,
        layers_to_drop=layers_to_drop,
        layer_norm=layer_norm,
        token=my_token
    )

    llm['tokenizer'] = tokenizer
    llm['tuned_model'] = tuned_model
    # llm['tuned_model'].eval() # the generate function is eval by default

    yield
    
app = FastAPI(lifespan=lifespan)

@app.get('/')
def hello_world():
    return {'hello':'world'}
 
 
@app.post("/process")
async def process_request(input_data: ProcessRequest) -> ProcessResponse:
    logger.info("%r" % llm)
    tokenizer = llm['tokenizer']
    model = llm['tuned_model']
    if input_data.seed is not None:
        set_seed(input_data.seed)
    # logger.info("Using device: {}".format(fabric.device))
    
    encoded = tokenizer(input_data.prompt,
                                 return_tensors = 'pt').to("cuda")
    prompt_length = encoded['input_ids'].size(1)
    max_returned_tokens = prompt_length + input_data.max_new_tokens
    assert max_returned_tokens <= model.config.max_position_embeddings, (
         max_returned_tokens,
         model.config.max_positino_embeddings,
     )  # maximum rope cache length

    t0 = time.perf_counter()
    # generate tokens
    tokens = model.generate(**encoded,
                            max_new_tokens = input_data.max_new_tokens,
                            num_beams=1,
                            temperature = input_data.temperature,
                            top_k = input_data.top_k,
                            renormalize_logits=True, 
                            return_dict_in_generate=True,
                            output_scores=True)
    t = time.perf_counter() - t0
    # get the transition scores
    logprobs = model.compute_transition_scores(
        tokens.sequences, tokens.scores,
    )[0]

    # extract the tokens with highest logprop at each gen as per API spec
    generated_tokens = []
    num_generated = len(tokens.scores)
    # truncate to tokens generated
    generated = tokens.sequences[0][-num_generated:]
    for token_idx, scores, logprob in zip(generated, tokens.scores, logprobs):
        token_str = tokenizer.decode([token_idx])
        top_idx = torch.argmax(scores).item()
        top_logprob = scores[0][top_idx]
        top_str = tokenizer.decode([top_idx])
        top_dict = {top_str: top_logprob.item()}
        generated_tokens.append(
            Token(text=token_str, logprob=logprob.item(), top_logprob=top_dict)
        )   
    # logprob of generated sequence
    generated_logprob = torch.sum(logprobs).item() 

    # decode generate text, possibly including the prompt
    if input_data.echo_prompt is False:
        output_text = tokenizer.decode(generated)
    else:
        output_text = tokenizer.decode(tokens.sequences[0])
    

    logger.info(
        f"Time for inference: {t:.02f} sec total, {len(generated) / t:.02f} tokens/sec"
    )

    logger.info(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    # return everything
    return ProcessResponse(
        text=output_text, 
        tokens=generated_tokens, 
        logprob=generated_logprob, 
        request_time=t
    )

@app.post("/tokenize")
async def tokenize(input_data: TokenizeRequest) -> TokenizeResponse:
    tokenizer = llm['tokenizer']
    #logger.info("Using device: {}".format(fabric.device))
    t0 = time.perf_counter()
    encoded= tokenizer(input_data.text, return_tensors = 'pt') #.to("cuda")
    # encoded = tokenizer.encode(
    #     input_data.text, bos=True, eos=False, device=fabric.device
    # )
    t = time.perf_counter() - t0
    
    # TODO fix this -- this is from Aaron, not sure why it's wrong?
    tokens = [int(i) for i in encoded.input_ids[0]]
    
    return TokenizeResponse(tokens=tokens, request_time=t)

@app.post("/decode")
async def decode(input_data: DecodeRequest) -> DecodeResponse:
    # logger.info("Using device: {}".format(fabric.device))
    tokenizer = llm['tokenizer']
    t0 = time.perf_counter()
    decoded = tokenizer.decode(torch.Tensor(input_data.tokens))
    # decoded = tokenizer.processor.decode(input_data.tokens)
    t = time.perf_counter() - t0
    return DecodeResponse(text=decoded, request_time=t)