FROM --platform=linux/amd64 nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV GIT_PYTHON_REFRESH=quiet
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get remove -y python3.10 && \
    apt-get install -y python3.9 \
        python3.9-venv \
        python3-pip \
        python3.9-distutils \
        python3-apt \
        git && \
    rm -rf /var/lib/apt/lists/*

RUN unlink /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python3 && \
    ln -s /usr/bin/python3.9 /usr/bin/python

WORKDIR /app

RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy separately so docker does not re-run installation
#  since the whl file changes more often than the requirements.txt
COPY /.requirements_cache/requirements_evaluation.txt /app/requirements.txt

RUN pip3 install -r \
    requirements.txt \
    --no-cache-dir

# Need to install this manually because of a bug in transformers
RUN pip3 install "tokenizers==0.14.0"

COPY dist /app/dist

# Need to install tokenizers manually because of a bug in transformers
RUN pip3 install --no-deps \
    "/app/dist/$(ls /app/dist | grep .whl)[training,submission]" \
    google-cloud-logging==3.6.0

ENTRYPOINT [ "cajajejo" ]
