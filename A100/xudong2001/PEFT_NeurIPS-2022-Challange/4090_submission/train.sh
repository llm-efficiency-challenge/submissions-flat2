cd FastChat
pip3 install --upgrade pip
pip install --no-cache-dir --upgrade -r requirements.txt
pip install flash-attn --nobuild-ioslation
pip3 install -e ".[model_worker,webui]"

python FastChat/fastchat/train/train_lora_llama2.py \
  --model_name_or_path Nousresearch/Llama-2-70b-hf \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --q_lora True \
  --data_path ../data/alpaca-gpt4/alpaca-gpt4-sharegpt.json \
  --bf16 True \
  --output_dir ../output/fastchat/llama-2-70b-alpaca-gpt4-sharegpt \
  --num_train_epochs 3 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 100 \
  --learning_rate 1e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 512 \
  --flash_attn True \
  --gradient_checkpointing True \
  --lazy_preprocess True