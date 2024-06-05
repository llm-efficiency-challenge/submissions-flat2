python merge_lora.py \
  --checkpoint_dir "$HOME/work/checkpoints/mistralai/Mistral-7B-v0.1" \
  --lora_path "$HOME/work/output/mistral_no_instruct_qlora_lit/iter-351999-ckpt.pth" \
  --out_dir "$HOME/work/output/lora_merged/Mistral-7B-v0.1-qlora"
# python merge_lora.py \
#   --checkpoint_dir "$HOME/work/LLama-2-13b-hf" \
#   --lora_path "$HOME/work/output/llama2_13b_qlora_lit/iter-191999-ckpt.pth" \
#   --out_dir "$HOME/work/output/lora_merged/llama-2-13b-qlora-instruct"
