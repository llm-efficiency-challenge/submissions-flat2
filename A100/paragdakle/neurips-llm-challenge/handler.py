from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import os
import logging
import math
import torch
from copy import deepcopy
import json
import time
import random
import numpy as np
from typing import Dict
from transformers import BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from dynamic_adapter import load_models, get_model

class ModelHandler:

    def __init__(self, model_location, bbq_adapter_path, cnn_adapter_path, math_adapter_path, routing_model_path) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_location)
        self.models = load_models(model_location,
                                  bbq_adapter_path,
                                  cnn_adapter_path,
                                  math_adapter_path,
                                  routing_model_path)
        self.model = None
        torch.set_float32_matmul_precision("high")

    def set_seed(self, seed) -> None:
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def pipeline_generate(self, raw_request: Dict):
        if raw_request['seed'] is not None:
            self.set_seed(raw_request['seed'])
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
            self.model.device
        )
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(
                raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
            )
            assert len(stop_sequence_ids.input_ids) == 1, "Total number of stop words should be 1."
            assert len(stop_sequence_ids.input_ids[0]) == 1, "Total number of tokens in each stop word should be 1."
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }
        t0 = time.perf_counter()
        self.model = get_model(self.models, self.tokenizer, raw_request["prompt"], None)
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)
        t = time.perf_counter() - t0
        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0]), 'request_time': t}

    def tokenize(self, inputs):
        t0 = time.perf_counter()
        if inputs['truncation']:
            tokens = self.tokenizer.encode(
                inputs['prompt'],
                truncation=inputs['truncation'],
                max_length=inputs['max_length'],
                add_special_tokens=False,
            )
        else:
            tokens = self.tokenizer.encode(inputs['prompt'], add_special_tokens=False)
        t = time.perf_counter() - t0
        return {'tokens':tokens, 'request_time': t}
