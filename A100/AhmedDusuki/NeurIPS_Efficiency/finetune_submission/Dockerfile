FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the lit-gpt files into the container at /submission
COPY /lit-gpt/ /submission/

RUN apt-get update && apt-get install -y git
# Remove unneeded libs
RUN pip uninstall -y torchaudio torchvision torchdata torchtext
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install -r requirements.txt huggingface_hub sentencepiece tokenizers bitsandbytes scipy

# some huggingface_hub versions require that the target dir exists
RUN mkdir -p checkpoints/meta-llama/Llama-2-7b-hf
# get llama weights
RUN python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf

# Copy over single file server
COPY ./lora.py /finetune/lora.py
COPY ./prepare_oasst1.py /scripts/prepare_oasst1.py
# prepare oasst1 data
RUN python scripts/prepare_oasst1.py

RUN python finetune/lora.py --quantize "bnb.nf4-dq" --precision "bf16-true" --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf" --data_dir "data/oasst1" --out_dir "out/lora/oasst1"