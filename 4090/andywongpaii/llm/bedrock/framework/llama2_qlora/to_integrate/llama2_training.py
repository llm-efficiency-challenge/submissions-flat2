#!/usr/bin/env python3
#coding: utf8
#author: Andy Wong
#usage:
# conda activate neurips2023-train_QLoRA
# python bedrock/framework/test2/llama2_training.py

# check https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=OJXpOgBFuSrc
# peft_config ..ect.

# TODO: 
# 1. What is this?
# The embed_tokens layer of the model is initialized with 
# self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.config.padding_idx), 
# self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=tokenizer.pad_token_id)
# which makes sure that encoding the padding token will output zeros, so passing it when initializing is recommended.
# ---------------------------------------------------------------------------------------------------------------------------------

from bedrock.framework.test2.qlora_finetune_config import qlora_finetune_config
print(qlora_finetune_config)

# TODO: follow this instruction (https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=ib_We3NLtj2E)
# TODO: then pack workflow into this (https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/pull/21/files) which is based on lit-gpt
# MUST check: https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py