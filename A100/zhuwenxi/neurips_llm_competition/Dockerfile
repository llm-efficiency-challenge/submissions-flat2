FROM nvcr.io/nvidia/pytorch:22.12-py3
RUN apt-get update  && apt-get install -y git python3-virtualenv wget
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install git-lfs \
    && git lfs install \
    && pip install huggingface_hub

WORKDIR /workspace
COPY download_qwen.py .
RUN python download_qwen.py
