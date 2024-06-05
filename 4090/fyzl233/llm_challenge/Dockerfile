FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793
COPY ./requirements.txt requirements.txt
RUN pip install -U -r requirements.txt

WORKDIR /workspace
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

ENV HUGGINGFACE_TOKEN="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"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
