# FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0
FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04
#FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

WORKDIR /workspace
ENV CUDA_HOME /usr/local/cuda
ENV PATH $PATH:/usr/local/cuda/bin

RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

# RUN pip install packaging
RUN git clone https://github.com/Dao-AILab/flash-attention
RUN cd flash-attention && pip install .
# CMD ["python3", "-c", "print('Flash Attention installed successfully!')"]

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

ENV HUGGINGFACE_TOKEN="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"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py

COPY ./cache.py cache.py
RUN python3 cache.py
RUN pip install -U --no-cache-dir accelerate
RUN pip install -U --no-cache-dir -i https://test.pypi.org/simple/ bitsandbytes

# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
