FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the lit-gpt files into the container at /submission
COPY /lit-gpt/ /submission/

# Remove unneeded libs
RUN pip uninstall -y torchaudio torchvision torchdata torchtext

# Setup server requriements
COPY ./fast_api_requirements.txt .
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN apt-get update && apt-get install -y git
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install -r requirements.txt huggingface_hub sentencepiece tokenizers bitsandbytes scipy

# some huggingface_hub versions require that the target dir exists
RUN mkdir -p checkpoints/meta-llama/Llama-2-7b-hf
# get llama weights
RUN python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
# TODO: download finetuned lora?

# Copy over single file server
COPY ./main.py .
COPY ./helper.py .
COPY ./api.py .
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]