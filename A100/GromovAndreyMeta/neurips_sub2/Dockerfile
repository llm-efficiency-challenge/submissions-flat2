# Use an official Python runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# due to dependency hell lets install everything without requirements.txt

# Setup server requriements
RUN pip install uvicorn fastapi
RUN pip install tokenizers sentencepiece protobuf
RUN pip install transformers
RUN pip install peft

# compile bitsandbytes, install git and make first
RUN apt-get update && apt-get install -y git make
RUN git clone https://github.com/timdettmers/bitsandbytes.git
WORKDIR /workspace/bitsandbytes
RUN CUDA_VERSION=118 make cuda11x
RUN python setup.py install
# install bitsandbytes requirement
RUN pip install scipy

# # Set the working directory in the container to /submission
WORKDIR /submission


# ## get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md

# download
# ## RUN python scripts/download.py --repo_id openlm-research/open_llama_3b
# ## RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/openlm-research/open_llama_3b
COPY ./download_model.py /submission/download_model.py
RUN python download_model.py


# Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
