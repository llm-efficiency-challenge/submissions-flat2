# Use an official Python runtime as a parent image
# other options in https://github.com/orgs/pytorch/packages/container/pytorch-nightly/versions?filters%5Bversion_type%5D=tagged
# Lit-GPT requires current nightly (future 2.1) for the latest attention changes
FROM ghcr.io/pytorch/pytorch-nightly:c69b6e5-cu11.8.0

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /NeurIPS_LLM_Kyutech-Jakee/ /submission/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN apt-get update && apt-get install -y git
RUN pip install -r requirements.txt huggingface_hub sentencepiece tokenizers bitsandbytes scipy

ENV HUGGINGFACE_TOKEN="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"
# get llama2 weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_llama_2.md
RUN python scripts/download.py --repo_id meta-llama/Llama-2-13b-hf --access_token ${HUGGINGFACE_TOKEN}
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf

# dir: model saved
ARG dir="eval_model"
ARG repo_id="miz22/Llama-2-13b-hf_limaoasst_lora_swa_r8a16_AdamW_bnb.nf4-dq_10000_128_1_0.0008_0.01_2023-10-25"

# RUN mkdir -p ${dir}
COPY ./download_ftmodel.py /submission/download_ftmodel.py
# download finetuned model from your repository in HaggingFace
RUN python download_ftmodel.py --repo_id ${repo_id} --directory ${dir} --access_token ${HUGGINGFACE_TOKEN}

# Convert lit-gpt model format
ENV CUDA_VISIBLE_DEVICES=0
RUN python scripts/merge_r8a16.py \
--checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf \
--lora_path ${dir}/lit_model_ave_Fix_AdamW_50000_128_1_0.0003_0.001.pth \
--out_dir ${dir}

# Copy *.json toknizer.model
RUN cp checkpoints/meta-llama/Llama-2-13b-hf/*.json \
${dir}
RUN cp checkpoints/meta-llama/Llama-2-13b-hf/tokenizer.model \
${dir}

# Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]