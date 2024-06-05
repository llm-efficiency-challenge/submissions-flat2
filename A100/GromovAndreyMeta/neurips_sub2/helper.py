from typing import Optional, Tuple, List
from transformers import (BitsAndBytesConfig, AutoModelForCausalLM, 
                          AutoTokenizer, LlamaTokenizer, LlamaForCausalLM)
import torch
import torch.nn as nn
import json
import logging
import os
from peft import PeftModel
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

def get_model_and_tokenizer(tuned_repo, layer_dropping=False,
        layers_to_drop=None, layer_norm=False, token=None):

    # Set global variables based on `push_model_hub_config.json`
    device_map="cuda:0"

    MODEL_PATH = "tuned-model-snapshot"

    # Find the config file for the model
    config_path = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        print(f"Config path {config_path} does not exist... exiting...")

    config = None
    with open(config_path) as f:
        config = json.load(f)

    if 'quantization_config' in config:
        q_config = config["quantization_config"]
        # Read the bits and bytes config from the `config.json` to make sure we use the same one
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=q_config["load_in_8bit"],
            load_in_4bit=q_config["load_in_4bit"],
            llm_int8_threshold=q_config["llm_int8_threshold"],
            llm_int8_skip_modules=q_config["llm_int8_skip_modules"],
            llm_int8_enable_fp32_cpu_offload=q_config["llm_int8_enable_fp32_cpu_offload"],
            llm_int8_has_fp16_weight=q_config["llm_int8_has_fp16_weight"],
            bnb_4bit_compute_dtype=q_config["bnb_4bit_compute_dtype"],
            bnb_4bit_quant_type=q_config["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=q_config["bnb_4bit_use_double_quant"],
        )

        # Set path to the base model with which we will merge our trained LoRA weights. Can be local.
        base_model_path = config["_name_or_path"]

        # Load base model
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            quantization_config=bnb_config,
            device_map=device_map,
            token=token
        )

    # Load the tokenizer from the directory
        tokenizer = LlamaTokenizer.from_pretrained(base_model_path, token=token)

        # If we decide to drop layers, do it before loading LORA adapters
        if layer_dropping:
            # Cutting a few layers out.
            def get_layers_to_drop(ix, start, end):
                    start = start
                    end = end
                    step = (end - start) // (ix)
                    layer_ixs = torch.arange(start, end, step)
                    return layer_ixs

            ix,start_idx,end_idx = layers_to_drop.strip().split(",")
            ix,start_idx,end_idx = int(ix),int(start_idx),int(end_idx)
            blocks_to_remove = get_layers_to_drop(ix, start_idx, end_idx)
            print(f"We are removing: {blocks_to_remove} layers")

            model_base.model.layers = nn.ModuleList([block for idx, block in enumerate(model_base.model.layers) if idx not in blocks_to_remove])

                    
        # Load & merge the adapters
        tuned_model = PeftModel.from_pretrained(
            model_base,
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
        )

        if layer_norm:
            # Function to merge layer norm weights
            def merge_LN(model, path_to_params):
                for p in model.parameters():
                    p.requires_grad=False 

                ln_params = {}
                with open(path_to_params, 'r') as file:
                    ln_params = json.load(file)

                for (n, p) in model.named_parameters():
                    for key in ln_params.keys():
                        new_key = key.replace('_orig_mod.', '', 1) # strip the '_orig_mod.'
                        if new_key == n:
                            p.copy_(torch.tensor(ln_params[key]).to("cuda")) #copy the tensor
                return model

            # Load the layer norm weights we used for this model
            path_to_layer_norm_weights = os.path.join(MODEL_PATH, "LN_weights.json")
            tuned_model = merge_LN(tuned_model, path_to_layer_norm_weights)
    else:
        print("Error loading model, could not find the quantization config")

    del model_base
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    return tuned_model, tokenizer


# def get_base_model(base_model_path, my_token):
    
#     # Set device
#     device_map = "auto"

#     # Set bnb parameters. We are doing 4bit + computeation in bfloat.
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
    

#     logger.info("pulling base model from hugging face")

#     # Load base model
#     model_base = LlamaForCausalLM.from_pretrained(
#         base_model_path, 
#         quantization_config=bnb_config,
#         device_map=device_map,
#         token = my_token
#     )
    
#     return model_base

# def make_tokenizer(base_model_path, my_token):
    
#     if os.path.exists(f'./tokenizers/{base_model_path}'):
#         logger.info("loading from saved tokenizer")
#         tokenizer = LlamaTokenizer.from_pretrained(f"./tokenizers/{base_model_path}")
#     else:
#         logger.info("downloading tokenizer from hugging face")
#         # Get tokenizer from the same place.
#         tokenizer = LlamaTokenizer.from_pretrained(base_model_path, token=my_token)
#         tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
#         tokenizer.padding_side = "right"
#         tokenizer.save_pretrained(f'tokenizers/{base_model_path}')
#     return tokenizer

# def get_tuned_model(base_model,tuned_model_path, torch_dtype):
#     """
#     Merges tuned model weights with base model using PeftModel

#     Args:
#         base_model (_type_): _description_
#         tuned_model_path (_type_): _description_
#         torch_dtype (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     tuned_model = PeftModel.from_pretrained(
#         base_model,
#         tuned_model_path,
#         torch_dtype=torch.bfloat16,
#     )
#     return tuned_model