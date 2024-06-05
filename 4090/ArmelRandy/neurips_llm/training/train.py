import argparse
import os

import torch
import random
import warnings
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, IA3Config, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)
from transformers import TrainingArguments

"""
Fine-Tune a model on an instruction dataset
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="bigcode/starcoderbase-1b")
    parser.add_argument(
        "--dataset_name", type=str, default="HuggingFaceH4/CodeAlpaca_20K"
    )
    parser.add_argument("--subset", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--size_valid_set", type=int, default=1000)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--num_of_sequences", type=int, default=1000)

    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str)
    parser.add_argument("--targets_only", action="store_true")  # default value of False

    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--eos_token_id", type=int, default=49152)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="c_proj c_attn q_attn")

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--no_fp16", action="store_false") #
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_gradient_checkpointing", action="store_false") # default = True
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=10, type=int)
    parser.add_argument("--eval_freq", default=10, type=int)
    parser.add_argument("--save_freq", default=50, type=int)
    parser.add_argument("--deepspeed", type=str)
    parser.add_argument("--use_flash_attn", action="store_true")
    return parser.parse_args()


def chars_token_ratio(
    dataset, tokenizer, input_column_name, output_column_name, nb_examples=400
):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example, input_column_name, output_column_name):
    """Prepare the text from a sample of the dataset."""
    if output_column_name:
        text = f"Question: {example[input_column_name]}\n\nAnswer: {example[output_column_name]}"
    else:
        text = example[input_column_name]
    return text

class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        input_column_name,
        output_column_name,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        shuffle=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else args.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.shuffle = shuffle

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(
                        prepare_sample_text(
                            next(iterator),
                            self.input_column_name,
                            self.output_column_name,
                        )
                    )
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


class TLConstantLengthDataset(ConstantLengthDataset):
    """
    Target Loss ConstantLengthDataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        left = "Question: "
        middle = "\n\n" + "Answer: "
        while more_examples:
            buffer_list, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    example = next(iterator)
                    q_str = example[self.input_column_name]
                    a_str = example[self.output_column_name]
                    buffer_list.append((q_str, a_str))
                    buffer_len += len(left + q_str + middle + a_str)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            all_token_ids = []
            all_label_ids = []
            for q_str, a_str in buffer_list:
                question_token_ids = self.tokenizer(left + q_str + middle)["input_ids"]
                answer_token_ids = self.tokenizer(a_str)["input_ids"]
                all_token_ids.extend(
                    question_token_ids + answer_token_ids + [self.concat_token_id]
                )
                all_label_ids.extend(
                    [-100] * len(question_token_ids)
                    + answer_token_ids
                    + [self.concat_token_id]
                )

            # sanity check
            assert len(all_token_ids) == len(all_label_ids)

            input_examples = []
            output_examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                label_ids = all_label_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    input_examples.append(input_ids)
                    output_examples.append(label_ids)

            if self.shuffle:
                examples = list(zip(input_examples, output_examples))
                random.shuffle(examples)
                input_examples, output_examples = zip(*examples)
                input_examples, output_examples = list(input_examples), list(
                    output_examples
                )

            for input_ids, label_ids in zip(input_examples, output_examples):
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(label_ids),
                }


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        args.dataset_name,
        data_dir=args.subset,
        split=args.split,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
    else:
        try : 
          train_data = dataset["train"]
          valid_data = dataset["test"]
        except :
          dataset = dataset.train_test_split(test_size=args.size_valid_set)
          train_data = dataset["train"]
          valid_data = dataset["test"]
        print(
            f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}"
        )

    if not args.output_column_name:
        warnings.warn(
            "You did not provide a output column name. If you're not going to work on 2 columns, ignore this warning."
        )

    chars_per_token = chars_token_ratio(
        train_data, tokenizer, args.input_column_name, args.output_column_name
    )
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    if args.targets_only:
        train_dataset = TLConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
        valid_dataset = TLConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
    else:
        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
        valid_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=args.seq_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    print("Loading the model")
    # disable caching mechanism when using gradient checkpointing
    """
    from transformers import BitsAndBytesConfig
    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False
    # More stuff
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    """
    #"""
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        use_auth_token=True,
        use_cache=not args.no_gradient_checkpointing,
        load_in_4bit=True,
        #load_in_8bit=True,
        device_map={"": Accelerator().process_index},
        #use_flash_attention_2=args.use_flash_attn,
        use_flash_attn=False,
        trust_remote_code=True,
    )
    #"""

    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=not args.no_gradient_checkpointing)
    
    #"""
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(" "),
    )
    model = get_peft_model(model, lora_config)
    #"""
    """
    ia3_config = IA3Config(
        **{
            "task_type": "CAUSAL_LM",
            "inference_mode": False,
            "target_modules": ["c_attn", "c_proj"],
            "feedforward_modules": ["w1", "w2"]
        },
    )
    model = get_peft_model(model, ia3_config)
    """
    

    print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_freq,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=f"{args.model_path.split('/')[-1]}-{args.dataset_name.split('/')[-1]}",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data
    )

    print("Training...")
    trainer.train()


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_auth_token=True, trust_remote_code=True)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)