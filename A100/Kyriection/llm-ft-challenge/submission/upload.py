
from huggingface_hub import HfApi
from huggingface_hub import login
login(token='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')
api = HfApi()


api.upload_folder(
    folder_path="/data1/lxh/llm_ft_exp/Llama-2-13b/Llama-2-13b_my_data_mmlu_gsm_lora_8_16_0.3e",
    repo_id="Xianhang/llama_mmlu_03",
    path_in_repo="",
    allow_patterns="*",
    delete_patterns="*.txt", # Delete all remote text files before
)