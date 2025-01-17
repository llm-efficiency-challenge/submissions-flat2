WANDB_DISABLED=True deepspeed --include localhost:0 --master_port 9090 ./LLaMA-Factory/src/train_bash.py \
    --deepspeed ./configs/ds_config_zero2.json \
    --stage sft \
    --model_name_or_path Qwen/Qwen-14B \
    --use_fast_tokenizer True \
    --do_train \
    --dataset_dir /home/nips/Nips_challenge/data/dataset \
    --dataset merge_data \
    --template default \
    --cutoff_len 2048 \
    --neft_alpha 5 \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_target "c_attn","c_proj" \
    --output_dir ./output/qwen_merge \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 20000 \
    --warmup_ratio 0.01 \
    --learning_rate 3e-4 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16 \
    --overwrite_output_dir
