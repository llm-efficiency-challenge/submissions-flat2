from huggingface_hub import snapshot_download
import os
import json
from transformers import AutoModelForCausalLM, LlamaTokenizer

if __name__ == "__main__":

    my_token = "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" # Andrey's token
    tuned_repo = "gromovand/LLama2-70B-layer-cut-35-rank-128"
    MODEL_PATH = "tuned-model-snapshot"

    # --------------------- Download our model ---------------------
    snapshot_download(
        repo_id=tuned_repo, 
        allow_patterns="*", 
        local_dir=MODEL_PATH, 
        token=my_token
    )

    # --------------------- Download everything needed to load our model ---------------------
    device_map = "auto"
    config_path = os.path.join(MODEL_PATH, "config.json")
    if not os.path.exists(config_path):
        print(f"Config path {config_path} does not exist... exiting...")

    config = None
    with open(config_path) as f:
        config = json.load(f)

    if '_name_or_path' in config:
        base_model_path = config["_name_or_path"]

        # Load base model
        model_base = AutoModelForCausalLM.from_pretrained(
            base_model_path, 
            device_map=device_map,
            token=my_token
        )

        # Load the tokenizer from the directory
        tokenizer = LlamaTokenizer.from_pretrained(base_model_path, token=my_token)