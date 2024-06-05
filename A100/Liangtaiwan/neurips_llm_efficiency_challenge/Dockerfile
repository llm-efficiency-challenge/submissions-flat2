# Use latest official release with CUDA support https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /lit-gpt/ /submission/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN apt-get update && apt-get install -y git
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install -r requirements.txt huggingface_hub sentencepiece tokenizers bitsandbytes scipy

# some huggingface_hub versions require that the target dir exists
# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
# some huggingface_hub versions require that the target dir exists
RUN mkdir -p checkpoints/open/open_llama_3b
RUN python scripts/download.py --repo_id meta-llama/Llama-2-13b-hf
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-13b-hf

RUN mkdir -p checkpoints/princeton-nlp/Sheared-LLaMA-1.3B
RUN python scripts/download.py --repo_id princeton-nlp/Sheared-LLaMA-1.3B
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/princeton-nlp/Sheared-LLaMA-1.3B --model_name Llama-2-1.3b-hf



# Copy over single file server
COPY ./main_cd.py /submission/main_cd.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main_cd:app", "--host", "0.0.0.0", "--port", "80"]
