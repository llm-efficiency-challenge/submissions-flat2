# Use latest official release with CUDA support https://hub.docker.com/r/pytorch/pytorch/tags
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
# FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Set the working directory in the container to /submission
WORKDIR /submission

# Copy the specific file into the container at /submission
COPY /lit-gpt/ /submission/

# Setup server requriements
COPY ./fast_api_requirements.txt fast_api_requirements.txt
RUN pip install --no-cache-dir --upgrade -r fast_api_requirements.txt

RUN apt-get update && apt-get install -y git
# Install any needed packages specified in requirements.txt that come from lit-gpt plus some optionals
RUN pip install -r requirements.txt huggingface_hub sentencepiece bitsandbytes tokenizers scipy

# ARG all_lora_layers=False

# Model to use and ft weights
### no quant eval
# ARG llm_model_group=meta-llama
# ARG llm_model=Llama-2-13b-chat-hf
# ARG ft_weights=llama_htv3r1024icl5_s1000
# ARG ft_weights=llama_htv4r1024icl5_s500
# ARG ft_weights=llama_htv5r256icl5_s500
# ARG lora_r=256
# ARG all_lora_layers=True
# ARG lora_r=1024


### quant 4 bit eval
# ARG llm_model_group=internlm
# ARG llm_model=internlm-20b
# ARG ft_weights=intern_htv3r256icl5_s1000
# ARG lora_r=256

### quant 4 bit eval
ARG llm_model_group=meta-llama
ARG llm_model=Llama-2-13b-chat-hf
# # ARG ft_weights=llama_htv3r128icl5_s1000
ARG ft_weights=iter-010399-ckpt.pth
# ARG lora_r=128

# some huggingface_hub versions require that the target dir exists
RUN mkdir -p checkpoints/$llm_model_group/$llm_model
# get open-llama weights: https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/download_openllama.md
RUN python scripts/download.py --repo_id $llm_model_group/$llm_model --access_token=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
RUN python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/$llm_model_group/$llm_model

# download lora weights
#RUN huggingface-cli login --token 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
RUN huggingface-cli download mikim21/lora_bt $ft_weights --repo-type=model --local-dir . --local-dir-use-symlinks False

# merge lora ft with llm_model. If you changed the LoRA hyperparameters (lora_r, lora_key, etc.) in the finetune/lora.py script, it is important to update the hyperparameter configuration in the scripts/merge_lora.py script accordingly. Otherwise, you will encounter size mismatch errors upon merging.
RUN python scripts/new_merge_lora.py --checkpoint_dir checkpoints/$llm_model_group/$llm_model --lora_path $ft_weights --out_dir out/lora_merged/final_model/ 
RUN cp -r checkpoints/$llm_model_group/$llm_model/*.json out/lora_merged/final_model/
RUN cp checkpoints/$llm_model_group/$llm_model/tokenizer.model out/lora_merged/final_model/


### Download entire weight
# RUN mkdir -p out/lora_merged/final_model/
# RUN huggingface-cli login --token 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
# RUN huggingface-cli download royson/htv2 tokenizer.model --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False
# RUN huggingface-cli download royson/htv2 generation_config.json --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False
# RUN huggingface-cli download royson/htv2 lit_config.json --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False
# RUN huggingface-cli download royson/htv2 pytorch_model.bin.index.json --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False
# RUN huggingface-cli download royson/htv2 tokenizer.json --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False
# RUN huggingface-cli download royson/htv2 tokenizer_config.json --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False
# RUN huggingface-cli download royson/htv2 lit_model.pth --subfolder bayestune_st2 --repo-type=model --local-dir out/lora_merged/final_model/ --local-dir-use-symlinks False

# Copy over single file server
COPY ./main.py /submission/main.py
COPY ./helper.py /submission/helper.py
COPY ./api.py /submission/api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
