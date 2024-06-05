FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt
RUN pip install huggingface_hub
RUN pip install --upgrade git+https://github.com/huggingface/transformers
RUN pip install accelerate tiktoken einops scipy peft deepspeed
#RUN pip install optimum auto-gptq

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="onuralp/Llama-2-13b-QLoRA-02-merged"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
