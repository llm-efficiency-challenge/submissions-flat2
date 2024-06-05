FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

RUN pip3 install "fschat[model_worker,webui]"

WORKDIR /workspace
# Setup server requriements
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install flash-attn --nobuild-ioslation
RUN pip install "fschat[model_worker,webui]"

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="xudong2023ucas/llama-2-70b-lora"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
