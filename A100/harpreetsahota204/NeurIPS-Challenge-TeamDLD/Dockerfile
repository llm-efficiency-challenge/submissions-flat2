FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

RUN apt update && apt install -y git

# Removed the creation and switching to a specific user to avoid user-specific details
# Removed setting the HOME environment variable

WORKDIR /workspace

# Removed the --chown flag to avoid user-specific details
COPY . /workspace

COPY ./deci-lm/requirements.txt ./
# COPY ./deci-lm/train.py ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Setup server requirements
COPY ./fast_api_requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN mkdir -p /workspace/models && \
    chmod 777 /workspace/models

# For API server
COPY ./main.py ./
COPY ./deci-lm/download_model.py ./
RUN python download_model.py

# CMD [ "python", "train.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
