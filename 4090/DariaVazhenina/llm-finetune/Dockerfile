FROM ghcr.io/pytorch/pytorch-nightly:b3874ab-cu11.8.0

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace

RUN wget https://gist.githubusercontent.com/mreso/ec65015cbfbd395f0c2adc17147adf1f/raw/41070f1058820b9e89bae885968cc666a7d6aa59/custom_dataset.py

COPY ./make_dataset.py make_dataset.py
COPY ./README.md README.md
COPY ./requirements.txt requirements.txt
COPY ./run.py run.py
COPY ./install.sh install.sh
COPY ./train.py train.py
COPY ./train.sh train.sh
COPY ./utils utils
COPY ./dataset_classes dataset_classes

ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="nicolasbo/bai_team"

COPY ./install.sh install.sh
RUN bash install.sh

# train
# CMD [ "python", "make_dataset.py"]
# RUN bash train.sh trained_models

# inference
# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
