#!/bin/bash

#$ -l rt_AG.small=1
#$ -N llama13b_lit
#$ -j y
#$ -o /scratch/acf15860xt/logs/
#$ -cwd


export MODULEPATH=$MODULEPATH:$HOME/.modulefiles
source /etc/profile.d/modules.sh
module load miniconda3 cudnn cuda
conda activate llmec2

export CUDA_VISIBLE_DEVICES=0

# MODEL_SIZE=7B
# NUM_GPUS=1
# BATCH_SIZE_PER_GPU=1
# TOTAL_BATCH_SIZE=16
# GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
# MODEL_NAME="mistralai/Mistral-7B-v0.1"
# 
python lora.py \
  --quantize "bnb.nf4" \
  --data_dir data/tulu_llama2 \
  --model_name Llama-2-13b-hf \
  --checkpoint_dir ~/work/LLama-2-13b-hf \
  --out_dir ~/work/output/llama2_13b_qlora_lit_no_instruct \
  --precision "bf16-true"
