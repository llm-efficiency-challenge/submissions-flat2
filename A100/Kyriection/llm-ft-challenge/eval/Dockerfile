# FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN pip install flash-attn --no-build-isolation 

RUN pip install transformers_stream_generator

RUN pip install tiktoken


RUN pip install ipdb #debug only delete afterwards

ENV HUGGINGFACE_TOKEN="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"
#ENV BASE_MODEL="mistralai/Mistral-7B-v0.1"
#ENV LORA_WEIGHTS="/data1/lxh/llm_ft_exp/mis_7B/mis_7B_flan2021_submix_lora_16_16"
ENV BASE_MODEL='Qwen/Qwen-14B'
ENV LORA_WEIGHTS="/data1/lxh/llm_ft_exp/Qwen-14B_ablate/Qwen-14B_my_data_mmlu_gsm_lora_8_32_0.3e/"
#ENV LORA_WEIGHTS="xxyyy123/Qwen14B-1018_dolly8k_cnn4kD_bbq12k_mmlu29k-2e5_r2"
ENV BASE_MODEL_CACHE='/data1/lxh/llm_data/Qwen-14B'
#ENV BASE_MODEL_CACHE="/data1/lxh/llm_data/mis_7B_weights"
#ENV CUDA_VISIBLE_DEVICES=0,1,2,3
#ENV BASE_MODEL='meta-llama/Llama-2-13b-hf'
#ENV LORA_WEIGHTS="/data1/lxh/llm_ft_exp/Llama-2-13b/Llama-2-13b_my_data_mmlu_gsm_lora_8_16_1e/"
##ENV LORA_WEIGHTS="xxyyy123/Qwen14B-1018_dolly8k_cnn4kD_bbq12k_mmlu29k-2e5_r2"
#ENV BASE_MODEL_CACHE='/data1/lxh/llama_weights'
# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
