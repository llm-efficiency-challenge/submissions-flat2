# fine-tune [llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) models 
# on the [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.
# leverage the PEFT library from Hugging Face, as well as QLoRA for more memory efficient finetuning.

# we will need `accelerate`, `peft`, `transformers`, `datasets` and TRL to leverage the recent [`SFTTrainer`](https://huggingface.co/docs/trl/main/en/sft_trainer).
# We will use `bitsandbytes` to [quantize the base model into 4bit](https://huggingface.co/blog/4bit-transformers-bitsandbytes). 
# [?] We will also install `einops` as it is a requirement to load Falcon models. einops==0.6.1

import os
import datasets
import torch
from peft import LoraConfig, PeftModel, PeftConfig
from transformers import (
  # AutoModelForCausalLM,
  # AutoTokenizer,
  LlamaTokenizer,
  LlamaForCausalLM,
  BitsAndBytesConfig,
  TrainingArguments,
)
from trl import SFTTrainer
# TODO: replace with llama model and tokenizer

def load_dataset(
  dataset_name = "databricks/databricks-dolly-15k",
  split="train",
  via="datasets",
):
  if via=="datasets":
    dataset = datasets.load_dataset(dataset_name, split=split)
  else:
    raise NotImplementedError(f"{via} is not yet supported.")
  return(dataset)

dataset = load_dataset()

def generate_prompt_template(
  intro_blurb = "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
  instruction_key = "### Instruction:",
  input_key = "Input:",
  response_key = "### Response:",
  end_key = "### End",
):
  prompt_no_input_format = """{intro}

  {instruction_key}
  {instruction}

  {response_key}
  {response}

  {end_key}""".format(
    intro=intro_blurb,
    instruction_key=instruction_key,
    instruction="{instruction}",
    response_key=response_key,
    response="{response}",
    end_key=end_key,
  )

  prompt_with_input_format = """{intro}

  {instruction_key}
  {instruction}

  {input_key}
  {input}

  {response_key}
  {response}

  {end_key}""".format(
    intro=intro_blurb,
    instruction_key=instruction_key,
    instruction="{instruction}",
    input_key=input_key,
    input="{input}",
    response_key=response_key,
    response="{response}",
    end_key=end_key,
  )

  return(prompt_no_input_format, prompt_with_input_format)

PROMPT_NO_INPUT_FORMAT, PROMPT_WITH_INPUT_FORMAT = generate_prompt_template()

def apply_prompt_template(examples):
  instruction = examples["instruction"]
  response = examples["response"]
  context = examples.get("context")

  if context:
    full_prompt = PROMPT_WITH_INPUT_FORMAT.format(instruction=instruction, response=response, input=context)
  else:
    full_prompt = PROMPT_NO_INPUT_FORMAT.format(instruction=instruction, response=response)
  return { "text": full_prompt }

dataset = dataset.map(apply_prompt_template)

print("\n\nsample 0 of dataset:\n", dataset["text"][0])

# NOTE:load the [llama2 7b hf](), quantize it in 4bit and attach LoRA adapters on it.

# model = "meta-llama/Llama-2-7b-hf"
model = "decapoda-research/llama-7b-hf"
# revision = "351b2c357c69b4779bde72c0e7f7da639443d904"

# tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer = LlamaTokenizer.from_pretrained(model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("-"*20)
print("tokenizer loaded.")
print("model provided tokenizer.eos_token:\n", tokenizer.eos_token)
print("we defined tokenizer.pad_token as:\n", tokenizer.pad_token)
print("-"*20)

bnb_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.float16, #TODO: try bfloat16?
)

# model = AutoModelForCausalLM.from_pretrained(
#   model,
#   quantization_config=bnb_config,
#   revision=revision,
#   trust_remote_code=True,
# )
model = LlamaForCausalLM.from_pretrained(
  model,
  quantization_config=bnb_config,
  # device_map=device_map,
  # revision=revision,
  trust_remote_code=True,
)
model.resize_token_embeddings(
    len(tokenizer),
    # pad_to_multiple_of=8,
)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.config.pretraining_tp = 1 # NOTE: >1 == experimental feature on tensor parallelism rank.

# Load the configuration file in order to create the LoRA model. 
# NOTE: According to QLoRA paper, it is important to consider all linear layers in the transformer block for maximum performance. 
# Therefore we will add `dense`, `dense_h_to_4_h` and `dense_4h_to_h` layers in the target modules in addition to the mixed query key value layer.

lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'k_proj', 'v_proj'] # Choose all linear layers from the model
)

# Here we will use the [`SFTTrainer` from TRL library](https://huggingface.co/docs/trl/main/en/sft_trainer) that gives a wrapper around transformers `Trainer` 
# to easily fine-tune models on instruction based datasets using PEFT adapters. Let's first load the training arguments below.

output_dir = "/workspace/PPO_learning/NDEE/sft/databricks/results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 500
logging_steps = 100
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 1000
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
  output_dir=output_dir,
  per_device_train_batch_size=per_device_train_batch_size,
  gradient_accumulation_steps=gradient_accumulation_steps,
  optim=optim,
  save_steps=save_steps,
  logging_steps=logging_steps,
  learning_rate=learning_rate,
  fp16=True,
  max_grad_norm=max_grad_norm,
  max_steps=max_steps,
  warmup_ratio=warmup_ratio,
  group_by_length=True,
  lr_scheduler_type=lr_scheduler_type,
  ddp_find_unused_parameters=False,
)

max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

# NOTE: We will also pre-process the model by upcasting the layer norms in float 32 for more stable training
for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

# Train the model
trainer.train()

# Save the LORA model
trainer.save_model(os.path.join(output_dir, "llama2-7b-lora-fine-tune"))

# TODO: later use the model in MLFlow

# Log the fine tuned model to MLFlow
# peft_model_id = "/local_disk0/llamav2-7b-lora-fine-tune"
# config = PeftConfig.from_pretrained(peft_model_id)

# from huggingface_hub import snapshot_download
# # Download the Llama-2-7b-hf model snapshot from huggingface
# snapshot_location = snapshot_download(repo_id=config.base_model_name_or_path)


# # COMMAND ----------

# import mlflow
# class LLAMAQLORA(mlflow.pyfunc.PythonModel):
#   def load_context(self, context):
#     self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts['repository'])
#     self.tokenizer.pad_token = tokenizer.eos_token
#     config = PeftConfig.from_pretrained(context.artifacts['lora'])
#     base_model = AutoModelForCausalLM.from_pretrained(
#       context.artifacts['repository'], 
#       return_dict=True, 
#       load_in_4bit=True, 
#       device_map={"":0},
#       trust_remote_code=True,
#     )
#     self.model = PeftModel.from_pretrained(base_model, context.artifacts['lora'])
  
#   def predict(self, context, model_input):
#     prompt = model_input["prompt"][0]
#     temperature = model_input.get("temperature", [1.0])[0]
#     max_tokens = model_input.get("max_tokens", [100])[0]
#     batch = self.tokenizer(prompt, padding=True, truncation=True,return_tensors='pt').to('cuda')
#     with torch.cuda.amp.autocast():
#       output_tokens = self.model.generate(
#           input_ids = batch.input_ids, 
#           max_new_tokens=max_tokens,
#           temperature=temperature,
#           top_p=0.7,
#           num_return_sequences=1,
#           do_sample=True,
#           pad_token_id=tokenizer.eos_token_id,
#           eos_token_id=tokenizer.eos_token_id,
#       )
#     generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

#     return generated_text

# # COMMAND ----------

# from mlflow.models.signature import ModelSignature
# from mlflow.types import DataType, Schema, ColSpec
# import pandas as pd
# import mlflow

# # Define input and output schema
# input_schema = Schema([
#     ColSpec(DataType.string, "prompt"), 
#     ColSpec(DataType.double, "temperature"), 
#     ColSpec(DataType.long, "max_tokens")])
# output_schema = Schema([ColSpec(DataType.string)])
# signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# # Define input example
# input_example=pd.DataFrame({
#             "prompt":["what is ML?"], 
#             "temperature": [0.5],
#             "max_tokens": [100]})

# with mlflow.start_run() as run:  
#     mlflow.pyfunc.log_model(
#         "model",
#         python_model=LLAMAQLORA(),
#         artifacts={'repository' : snapshot_location, "lora": peft_model_id},
#         pip_requirements=["torch", "transformers", "accelerate", "einops", "loralib", "bitsandbytes", "peft"],
#         input_example=pd.DataFrame({"prompt":["what is ML?"], "temperature": [0.5],"max_tokens": [100]}),
#         signature=signature
#     )

# # COMMAND ----------

# # MAGIC %md
# # MAGIC Run model inference with the model logged in MLFlow.

# # COMMAND ----------

# import mlflow
# import pandas as pd


# prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
# ### Instruction:
# if one get corona and you are self isolating and it is not severe, is there any meds that one can take?

# ### Response: """
# # Load model as a PyFuncModel.
# run_id = run.info.run_id
# logged_model = f"runs:/{run_id}/model"

# loaded_model = mlflow.pyfunc.load_model(logged_model)

# text_example=pd.DataFrame({
#             "prompt":[prompt], 
#             "temperature": [0.5],
#             "max_tokens": [100]})

# # Predict on a Pandas DataFrame.
# loaded_model.predict(text_example)