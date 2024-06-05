#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
# conda activate neurips2023-train
# bash bedrock/framework/test/training/step1_supervised_finetuning/training_scripts/opt/single_gpu/run_125m_test.sh
# if functional, on a separate terminal window,  checkout tensorboard: 
# conda activate neurips2023-train
# tensorboard --logdir=bedrock/framework/test/training/step1_supervised_finetuning/test_run_output/ds_tensorboard_logs/step1_model_tensorboard

HOMEDIR="bedrock/framework/test/training/step1_supervised_finetuning"
OUTPUT="./$HOMEDIR/test_run_output"
ZERO_STAGE=0
# if [ "$OUTPUT" == "" ]; then
#     OUTPUT=./output
# fi
# if [ "$ZERO_STAGE" == "" ]; then
#     ZERO_STAGE=0
# fi
mkdir -p $OUTPUT

deepspeed --num_gpus 1 $HOMEDIR/main.py --model_name_or_path facebook/opt-125m \
   --gradient_accumulation_steps 1 \
   --zero_stage $ZERO_STAGE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --lora_dim 32 \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --deepspeed --output_dir $OUTPUT &> $OUTPUT/training.log