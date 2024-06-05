import uvicorn

import torch
import argparse
from fastapi import FastAPI
from huggingface_hub import login

import logging


torch.set_float32_matmul_precision("high")

from api import (
    ProcessRequest,
    ProcessResponse,
    TokenizeRequest,
    TokenizeResponse,
    Token,
)

import os
import time
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Optional, Union

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config
from torch.nn import CrossEntropyLoss
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Dict, List
import re
import random
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


class Exllamav2HF(PreTrainedModel):
    def __init__(self, config: ExLlamaV2Config):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlamaV2(config)
        split = None
        self.ex_model.load(split)

        self.generation_config = GenerationConfig()

        self.ex_cache = ExLlamaV2Cache(self.ex_model)
        self.past_seq = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache", True)
        labels = kwargs.get("labels", None)
        past_key_values = kwargs.get("past_key_values", None)

        if len(args) > 0:
            input_ids = args[0]
            is_negative = True
            past_seq = self.past_seq_negative
            ex_cache = self.ex_cache_negative
        else:
            input_ids = kwargs["input_ids"]
            is_negative = False
            past_seq = self.past_seq
            ex_cache = self.ex_cache

        seq = input_ids[0].tolist()
        if is_negative and past_key_values is not None:
            seq = past_key_values + seq

        seq_tensor = torch.tensor(seq)
        reset = True

        # Make the forward call
        if labels is None:
            if past_seq is not None:
                min_length = min(past_seq.shape[0], seq_tensor.shape[0])
                indices = torch.nonzero(
                    ~torch.eq(
                        past_seq[:min_length], seq_tensor[:min_length]
                    )
                )
                if len(indices) > 0:
                    longest_prefix = indices[0].item()
                else:
                    longest_prefix = min_length

                if longest_prefix > 0:
                    reset = False
                    ex_cache.current_seq_len = longest_prefix
                    if len(seq_tensor) - longest_prefix > 1:
                        self.ex_model.forward(
                            seq_tensor[longest_prefix:-1].view(1, -1),
                            ex_cache,
                            preprocess_only=True,
                        )

            if reset:
                ex_cache.current_seq_len = 0
                if len(seq_tensor) > 1:
                    self.ex_model.forward(
                        seq_tensor[:-1].view(1, -1),
                        ex_cache,
                        preprocess_only=True,
                    )

            logits = self.ex_model.forward(
                seq_tensor[-1:].view(1, -1), ex_cache
            ).to(input_ids.device)
        else:
            ex_cache.current_seq_len = 0
            logits = self.ex_model.forward(
                seq_tensor.view(1, -1), ex_cache, last_id_only=False
            )

        if is_negative:
            self.past_seq_negative = seq_tensor
        else:
            self.past_seq = seq_tensor

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=seq if use_cache else None,
            loss=loss,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[
            Union[str, os.PathLike]
        ],
        *model_args,
        **kwargs,
    ):
        assert (
            len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(
                pretrained_model_name_or_path
            )

        config = ExLlamaV2Config()
        config.model_dir = str(pretrained_model_name_or_path)
        config.prepare()

        # config.max_seq_len = shared.args.max_seq_len
        # config.scale_pos_emb = shared.args.compress_pos_emb
        # config.scale_alpha_value = shared.args.alpha_value

        return Exllamav2HF(config)


def get_model(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = Exllamav2HF.from_pretrained(model)
    model.eval()
    return model

app = FastAPI()

logger = logging.getLogger(__name__)
# Configure the logging module
logging.basicConfig(level=logging.INFO)

login(token=os.environ["HUGGINGFACE_TOKEN"])


    # Initialize model and cache

model_path =  "/workspace/llama2-exl2-4.0bpw"
print("Loading model: " + model_path)

model = get_model(model_path)
tokenizer = LlamaTokenizer.from_pretrained(model_path)
generation_config = GenerationConfig.from_pretrained(model_path)
generation_kwargs = generation_config.to_dict()


# cache = ExLlamaV2Cache(model)
# generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
# generator.warmup()

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

    encoded = tokenizer(input_data.prompt, return_tensors="pt")
    prompt_length = encoded["input_ids"][0].size(0)
    # max_returned_tokens = prompt_length + input_data.max_new_tokens
    # assert max_returned_tokens <= LLAMA2_CONTEXT_LENGTH, (
    #     max_returned_tokens,
    #     LLAMA2_CONTEXT_LENGTH,
    # )

    t0 = time.perf_counter()
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    generation_kwargs["max_new_tokens"]=input_data.max_new_tokens
    generation_kwargs["temperature"]=input_data.temperature
    generation_kwargs["top_k"]=input_data.top_k
    generation_kwargs["return_dict_in_generate"]=True
    generation_kwargs["output_scores"]=True
    generation_kwargs["do_sample"]=False


    with torch.no_grad():
        outputs = model.generate(
            **encoded, **generation_kwargs,
        )

    t = time.perf_counter() - t0

    output = tokenizer.decode(outputs.sequences[0][prompt_length:], skip_special_tokens=True)
    output = output.split('$', 1)[0]
    output = output.split("\n\n")[0]

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

