FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@acce2d8770455711ae9f2ca61c01b15bc316cdbbls
RUN pip install torch==2.0.0
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="CJ-gyuwonpark/ch-70b-v3"

COPY ./download.py download.py
#RUN python download.py

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
