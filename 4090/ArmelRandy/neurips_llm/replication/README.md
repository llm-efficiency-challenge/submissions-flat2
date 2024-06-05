# How to replicate this work?

It is fairly easy to replicate the results that we obtained in this challenge. We used 3 datasets, each of them was a combination of 3 subsets :
- [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) : We sample 5000 instances from the version 1.0.0 and we clean it.
- [TruthfulQA (helm version)](https://huggingface.co/datasets/lighteval/truthfulqa_helm) : We consider all the examples of the training set. 
- [BBQ (helm version)](https://huggingface.co/datasets/lighteval/bbq_helm) : For each bias, we randomly select 8% of the training set.
- [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1) : We followed the preprocessing of guanaco to build a dataset of about 9K instances.

We thus built :
- [most_precious](https://huggingface.co/datasets/ArmelRandy/most_precious) : 5000 CNN/DM + TruthfulQA + BBQ + Oasst_en
- [most_precious_4](https://huggingface.co/datasets/ArmelRandy/most_precious_4) : 1.2K CNN/DM + TruthfulQA + BBQ + Oasst_en
- [most_precious_5](https://huggingface.co/datasets/ArmelRandy/most_precious_5) : TruthfulQA + BBQ + Oasst

The command is :

```
python train.py \
  --model_path "Qwen/Qwen-14B"\
  --dataset_name "ArmelRandy/most_precious_5"\
  --seq_length 2048\
  --input_column_name "prompt"\
  --max_steps 1000\
  --batch_size 1\
  --gradient_accumulation_steps 16 \
  --lora_r 16 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --no_gradient_checkpointing \
  --target_modules "c_attn c_proj w1 w2"\
  --learning_rate 1e-4\
  --lr_scheduler_type "cosine"\
  --num_warmup_steps 10\
  --weight_decay 0.05\
  --log_freq 5\
  --eval_freq 5\
  --save_freq 5\
  --output_dir "./checkpoints-qwen-14b"\
  --use_flash_attn \
  --neftune_noise_alpha 5
```

We also used a slightly modified version which includes the following lines :

```python
tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True, trust_remote_code=True, pad_token="<|endoftext|>")
tokenizer.eos_token = tokenizer.pad_token
tokenizer.eos_token_id = tokenizer.pad_token_id
```
