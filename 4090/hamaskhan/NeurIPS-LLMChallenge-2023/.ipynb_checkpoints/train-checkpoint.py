import os

from huggingface_hub import login, HfApi 
from llama_recipes.finetuning import main as finetuning

def main():
    login(token=os.environ["HUGGINGFACE_TOKEN"])
    
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=256'
    #os.environ['RANK'] = '0'
    #os.environ['LOCAL_RANK'] = '0'
    #os.environ['WORLD_SIZE'] = '1'
    #os.environ['MASTER_ADDR'] = 'localhost'
    #os.environ['MASTER_PORT'] = '12355'

    kwargs = {
        "model_name": "meta-llama/Llama-2-7b-hf",
        "use_peft": True,
        "peft_method": "lora",
        "quantization": True,
        "batch_size_training": 2,
        "dataset": "custom_dataset",
        "custom_dataset.file": "./custom_dataset.py",
        "output_dir": "./output_dir",
        "enable_fsdp": False,
        "low_cpu_fsdp": False,
        "run_validation": True,
        "gradient_accumulation_steps": 1,
        "num_epochs": 2,
        "num_workers_dataloader": 1,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "gamma": 0.85,
        "seed": 42,
        "use_fp16": False,
        "mixed_precision": True,
        "val_batch_size": 1,
        "freeze_layers": False,
        "num_freeze_layers": 1,
        "one_gpu": False,
        "save_model": True,
        "dist_checkpoint_root_folder": "PATH/to/save/FSDP/model", # will be used if using FSDP
        "dist_checkpoint_folder": "fine-tuned", # will be used if using FSDP
        "save_optimizer": True, # will be used if using FSDP
        "use_fast_kernels": True, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        
    }
    
    finetuning(**kwargs)

    api = HfApi() 

    api.upload_folder( 
        folder_path='./output_dir/', 
        repo_id=os.environ["HUGGINGFACE_REPO"], 
        repo_type='model', 
    )

if __name__ == "__main__":
    print('\nStarting Training!!')
    main()