import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import (
    get_peft_model,
    prepare_model_for_int8_training,
    LoraConfig,
    TaskType,
    PeftModel
)

import os
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_TOKEN"])

class ModelManager():
    def __init__(self, model_name_or_path, architecture, resume_path):
        self.model_name_or_path = model_name_or_path
        self.architecture = architecture
        self.resume_path = resume_path
    
    def load_tokenizer(self, use_fast_tokenizer=True, add_pad_token=False):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=use_fast_tokenizer
        )
        if add_pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_config(self, output_hidden_states=False, output_attentions=False, return_dict=False):
        config = AutoConfig.from_pretrained(
            self.model_name_or_path,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict,
            predict_with_generate=True,
            #trust_remote_code=True,     # Whether or not to allow for custom models defined on the Hub in their own modeling files
            use_cache=False,
        )
        return config
    
    def load_model(self, config, dtype=torch.bfloat16, load_in_8bit=False, device_map="auto", add_pad_token=False):
        if not self.resume_path:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=config,
                torch_dtype=dtype,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                from_tf=bool(".ckpt" in self.model_name_or_path),
            )
        else:
            raise Exception("Resume from checkpoint is not implemented!")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=config,
                torch_dtype=dtype,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            )
            # Load fine-tuned model
            model = PeftModel.from_pretrained(
                model, 
                self.resume_path, 
                device_map="auto"
            )
        if load_in_8bit:
            model = prepare_model_for_int8_training(model)
        if add_pad_token:
            model.config.pad_token_id = model.config.eos_token_id
        
        return model
        
    def get_lora_config(self, r, lora_alpha, target_modules, lora_dropout, bias, task_type):
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type
        )
        return lora_config
        
    def get_lora_model(self, model, lora_config):
        return get_peft_model(model, lora_config)
