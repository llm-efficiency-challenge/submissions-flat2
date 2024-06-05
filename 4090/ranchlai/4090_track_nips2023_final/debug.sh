
# this is for local debugging, without donloading the whole model
docker run -it -p 8081:80 --gpus "device=0" \
-v /mnt/data3/kl/Qwen-14B-gptq-4bits:/workspace/qwen-gptq \
 qwen_recipes_inference:latest /bin/bash 

# -v /root/code/nips2023_final/eval/output-4bit-bf16-v13a-selected-merged:/workspace/qwen_lora \
