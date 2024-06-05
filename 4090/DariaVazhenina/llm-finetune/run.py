import logging
import os
import random
import sys

import datasets
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    Trainer, Seq2SeqTrainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from peft import TaskType

import re
import torch
import torch.nn

from dataset_classes.dataset_loader import load_dataset
from utils.data_arguments import DataTrainingArguments
from utils.model_arguments import ModelArguments
from utils.model_manager import ModelManager
from utils.data_collator import InstructCollator
from utils.trainer_no_shuffle import NoShuffleTrainer
from huggingface_hub import login

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"

login(token=os.environ["HUGGINGFACE_TOKEN"])


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    # Setup logging ---------------------------------------------------
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Training/evaluation parameters {training_args}")
    # -------------------------------------------------------------------

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load tokenizer/Load pretrained model/Apply LoRA ---------------------------------------
    add_pad_token = True
    model_manager = ModelManager(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        architecture="GPTNeoX",
        resume_path = model_args.resume_path
    )
    tokenizer = model_manager.load_tokenizer(use_fast_tokenizer=True, add_pad_token=add_pad_token)
    config = model_manager.load_config()
    model = model_manager.load_model(
        config,
        dtype=torch.bfloat16,
        load_in_8bit=True, # Load quantized model
        device_map="auto",
        add_pad_token=add_pad_token
    )
    lora_config = model_manager.get_lora_config(
        r=8,
        lora_alpha=32,
        target_modules = ["q_proj", "v_proj"],  #Llama2
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = model_manager.get_lora_model(model, lora_config)
    # ----------------------------------------------------------------------------------------
    
    # Get max_seq_length
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Load dataset
    train_dataset = load_dataset(
        name=data_args.dataset_name,
        split="train",
        is_train_or_eval=True,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        sort=data_args.sort_strategy,
        csv_path=data_args.csv_path_train
    )
    if training_args.do_eval:
        eval_dataset = load_dataset(
            name=data_args.dataset_name,
            split="val",
            is_train_or_eval=True,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            sort=data_args.sort_strategy,
            csv_path=data_args.csv_path_val
        )
    else:
        eval_dataset = None
    
    data_collator = InstructCollator(tokenizer)

    if data_args.sort_strategy == "random":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    elif data_args.sort_strategy == "perplexity":
        trainer = NoShuffleTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
    else:
        raise ValueError("Invalid sort strategy")
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()
