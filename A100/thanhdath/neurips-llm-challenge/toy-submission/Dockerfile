FROM continuumio/miniconda3:latest AS compile-image
# Specify py version
ENV PYTHON_VERSION=3.8
# Install apt libs - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile


# ENV http_proxy=http://10.30.153.169:3128
# ENV https_proxy=http://10.30.153.169:3128
# ENV no_proxy=localhost,127.0.0.1,0.0.0.0,localaddress,.localdomain.com
# RUN echo "Acquire::http::Proxy \"http://10.30.153.169:3128\";" >> /etc/apt/apt.conf.d/proxy.conf
# RUN echo "Acquire::https::Proxy \"http://10.30.153.169:3128\";" >> /etc/apt/apt.conf.d/proxy.conf
RUN apt-get update && \
    apt-get install -y curl git wget software-properties-common git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists*

# Install audio-related libraries 
RUN apt-get update && \
    apt install -y ffmpeg

RUN apt install -y libsndfile1-dev
RUN git lfs install

# Create our conda env - copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# RUN conda create --name peft python=${PYTHON_VERSION} ipython jupyter pip
RUN conda install python=${PYTHON_VERSION} ipython jupyter pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Below is copied from https://github.com/huggingface/accelerate/blob/main/docker/accelerate-gpu/Dockerfile
# We don't install pytorch here yet since CUDA isn't available
# instead we use the direct torch wheel
# ENV PATH /opt/conda/envs/peft/bin:$PATH
# Activate our bash shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]
# Activate the conda env and install transformers + accelerate from source
RUN python3 -m pip install --no-cache-dir \
    librosa \
    "soundfile>=0.12.1" \
    scipy \
    transformers \
    accelerate \
    peft

RUN python3 -m pip install --no-cache-dir bitsandbytes

# Stage 2
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04 AS build-image
COPY --from=compile-image /opt/conda /opt/conda
ENV PATH /opt/conda/bin:$PATH
ENV HUGGINGFACE_TOKEN="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"

# # Install apt libs
# RUN apt-get update && \
#     apt-get install -y curl git wget && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists*

# RUN echo "source activate peft" >> ~/.profile

# Activate the virtualenv
# CMD ["/bin/bash"]

WORKDIR /submission
# ENV http_proxy=http://10.30.153.169:3128
# ENV https_proxy=http://10.30.153.169:3128
# ENV no_proxy=localhost,127.0.0.1,0.0.0.0,localaddress,.localdomain.com
# RUN echo "Acquire::http::Proxy \"http://10.30.153.169:3128\";" >> /etc/apt/apt.conf.d/proxy.conf
# RUN echo "Acquire::https::Proxy \"http://10.30.153.169:3128\";" >> /etc/apt/apt.conf.d/proxy.conf
# # Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt
RUN pip install lightning==2.0.6 scipy

# COPY ./requirements.txt  requirements.txt
# RUN pip install -r requirements.txt
# # Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# # Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
