FROM --platform=linux/amd64 python:3.9-slim-bullseye

WORKDIR /app

RUN pip3 install --upgrade pip \
    && pip3 --no-cache-dir install crfm-helm==0.2.4 mlflow \
    && python3 -c "import nltk; nltk.download('punkt')"

RUN apt-get update && apt-get install -y wget

COPY /.requirements_cache/requirements_helm.txt /app/requirements.txt

RUN pip3 install -r \
    requirements.txt \
    --no-cache-dir

COPY dist /app/dist

RUN pip3 install --no-deps \
    "/app/dist/$(ls /app/dist | grep .whl)" \
    google-cloud-logging==3.6.0

ENTRYPOINT [ "cajajejo" ]
