import os
import sys

from huggingface_hub import login, HfApi 

def main():
    login(token="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")
    api = HfApi() 
    # api.create_repo(f"jiangchensiat/{sys.argv[2]}")
    api.upload_folder( 
        # folder_path=sys.argv[1], 
        folder_path='/workspace/Llama-2-70b-hf-4.0bpw',
        repo_id=f"jiangchensiat/llama2-exl2-4.0bpw", 
        repo_type='model', 
    )

if __name__ == "__main__":
    main()
