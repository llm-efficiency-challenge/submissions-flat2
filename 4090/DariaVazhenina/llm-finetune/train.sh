#!/bin/bash

# Directory Settings
save_dir=$1       # output dir to place trained models

# Model name 
#model_name='meta-llama/Llama-2-7b-hf'
model_name='Llama-2-7b-hf'

# Parameters related to dataset
dataset_name="csv"     # Choice from imdb/cnndm/dolly
sort_strategy="random" # Choice from random/perplexity
csv_path_train="./data/combined_dataset.csv"

# Create a separate dir under save_dir to save the model for this training
training_name="Finetuned_Llama2-7B"

# Hyper Parameters
epochs=1
learning_rate=2e-4
train_batch_size=4 
eval_batch_size=1
max_seq_length=1024
eval_accumulation_steps=1

eval_steps=1000
save_steps=1000
logging_steps=20

output_dir="${save_dir%/}/${training_name}"

`python run.py  \
  --log_level=info\
  --do_train=True \
  --do_eval=False \
  --csv_path_train=$csv_path_train \
  --csv_path_val=$csv_path_val \
  --model_name_or_path=$model_name \
  --dataset_name=$dataset_name \
  --sort_strategy=$sort_strategy \
  --per_device_train_batch_size=$train_batch_size \
  --max_seq_length=$max_seq_length \
  --learning_rate=$learning_rate \
  --num_train_epochs=$epochs \
  --output_dir=$output_dir \
  --overwrite_output_dir \
  --save_strategy=steps \
  --save_steps=$save_steps \
  --logging_strategy=steps \
  --logging_steps=$logging_steps \
`
