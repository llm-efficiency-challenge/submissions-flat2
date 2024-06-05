# prebuild Pytorch docker with torch 2.0, CUDA 11.8, python 3.8
# [Nvidia docker image Release 22.12 ]
# (https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-12.html#rel-22-12)
FROM nvcr.io/nvidia/pytorch:22.12-py3

# Set the working directory in the container to /submission
# WORKDIR /submission
WORKDIR /workspace

# Copy the specific file/folder(s) into the container at /submission
# COPY /DeepSpeedExamples /workspace/DeepSpeedExamples
# COPY /MOSS-RLHF /workspace/MOSS-RLHF
# COPY /utils /workspace/utils

# Setup server requriements (upgrade to torch==2.0.1)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY ./dependencies.txt dependencies.txt
RUN pip3 install --no-cache-dir --upgrade -r dependencies.txt
RUN pip3 uninstall -y apex

# Update linux apt-get and install required packages
# RUN apt-get update
# RUN apt-get install -y libaio-dev

# Install deepspeed
# RUN python -m pip install --upgrade pip
# # RUN DS_BUILD_OPS=1 pip install deepspeed
# RUN DS_BUILD_SPARSE_ATTN=0 pip install deepspeed
# RUN ds_report

# Install any needed packages specified in requirements.txt that come from lit-gpt
# RUN apt-get update && apt-get install -y git
# RUN pip install -r requirements.txt huggingface_hub sentencepiece

# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
# RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
# RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b


####################################################################
# Training procedure
####################################################################



####################################################################
# Inference server
####################################################################

# Copy over single file server
# COPY ./main.py /submission/main.py
# COPY ./helper.py /submission/helper.py
# COPY ./api.py /submission/api.py

# Run the server
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]