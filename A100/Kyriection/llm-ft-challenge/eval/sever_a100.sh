
MODEL_PATH=$HOME

export HUGGINGFACE_TOKEN="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"
export BASE_MODEL="mistralai/Mistral-7B-v0.1"
export LORA_WEIGHTS="$HOME/data_efficient_llm/mis_7B/mis_7B_dolly_lora_16_16/"
export BASE_MODEL_CACHE="$HOME/llm_data/mis_7B_weights"
#export CUDA_VISIBLE_DEVICES=2,3

uvicorn main:app --host="0.0.0.0"  --port 80