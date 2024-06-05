FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Use latest official release with CUDA support https://hub.docker.com/r/pytorch/pytorch/tags
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container to /submission
# WORKDIR /submission

# Copy the specific file into the container at /submission
# COPY /lit-gpt/ /submission/

# Setup server requriements
# COPY ./fast_api_requirements.txt fast_api_requirements.txt
# RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN apt-get update && apt-get install -y git
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install huggingface_hub sentencepiece tokenizers bitsandbytes scipy

# some huggingface_hub versions require that the target dir exists
# RUN mkdir -p checkpoints/openlm-research/open_llama_3b
# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
# RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
# RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b

# Copy over single file server
# COPY ./main.py /submission/main.py
# COPY ./helper.py /submission/helper.py
# COPY ./api.py /submission/api.py
# Run the server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


RUN export PIP_DEFAULT_TIMEOUT=10000
RUN apt-get update  && apt-get install -y git python3-virtualenv wget
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793
WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt
RUN pip install optimum
RUN pip install auto-gptq


RUN chmod 1777 /tmp

# WORKDIR /root
# RUN chmod 1777 /tmp
# RUN chmod 1777 /tmp


RUN apt-get update && apt-get install -y git-lfs
# RUN git clone https://huggingface.co/kl1996/qwen-gptq
# # RUN git clone https://huggingface.co/TheBloke/CodeLlama-34B-GPTQ

# # WORKDIR /workspace/CodeLlama-34B-GPTQ/
# # RUN ls -la
# # RUN git lfs pull
# # WORKDIR /workspace
# # RUN ls -la

# RUN git clone https://huggingface.co/kl1996/qwen_v8_lora
# WORKDIR /workspace/qwen_v8_lora/
# RUN ls -la
# RUN git lfs pull
# WORKDIR /workspace
# RUN ls -la
ENV HUGGINGFACE_TOKEN="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"
# ENV HUGGINGFACE_REPO="kl1996/test_1"
COPY ./main.py main.py
COPY ./api.py api.py
COPY ./prompting.py prompting.py
RUN pip install transformers_stream_generator einops tiktoken loguru
# pip install tiktoken`

RUN ls -la
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]