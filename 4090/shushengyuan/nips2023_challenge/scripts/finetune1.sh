#!/bin/bash
conda init bash &&
source ~/.bashrc &&
conda activate nips2023 &&
dataset=dolly-orca
export HUGGINGFACE_TOKEN="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"
python qlora.py \
    --use_ema True\
    --bf16 \
    --bits 4 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --learning_rate 0.0002 \
    --source_max_len 16 \
    --target_max_len 1536\
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --output_dir output \
    --do_train True\
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 40 \
    --do_eval False\
    --evaluation_strategy epoch \
    --per_device_eval_batch_size 1 \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_modules all \
    --do_predict False\
    --do_mmlu_eval False\
    --dataset $dataset \
    --max_new_tokens 1024 \
    --dataloader_num_workers 0 \
    --group_by_length \
    --gradient_checkpointing \
    --remove_unused_columns False \
    --warmup_ratio 0.03 \
    --lr_scheduler_type constant \
    --data_seed 42 \
    --use_auth \
    --gradient_accumulation_steps 1 \
    --adam_beta2 0.999 \
    --max_grad_norm 0.3 \
    --seed 0 \
    --weight_decay 0.0 \
    --max_training_time 86400
