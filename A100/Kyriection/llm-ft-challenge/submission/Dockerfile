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




ENV HUGGINGFACE_TOKEN="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"
ENV CUDA_VISIBLE_DEVICES=0
ENV BASE_MODEL='meta-llama/Llama-2-13b-hf'
ENV LORA_WEIGHTS="Xianhang/llama_mmlu_03"

#ENV BASE_MODEL_CACHE='/data1/lxh/llama_weights'

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
