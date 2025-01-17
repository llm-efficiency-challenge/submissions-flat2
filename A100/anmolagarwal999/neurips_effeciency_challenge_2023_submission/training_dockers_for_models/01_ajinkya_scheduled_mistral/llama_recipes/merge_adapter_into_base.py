# code taken from here: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the adapter name"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_name is not None, "please provide the name of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
if peft_config.task_type == "SEQ_CLS":
    # The sequence classification task is used for the reward model in PPO
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.base_model_name, num_labels=1, torch_dtype=torch.bfloat16
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
    )

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_name)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")

# pushing the model to HF HUB
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)


# command to merge: python trl/examples/research_projects/stack_llama/scripts/merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="dpo/final_checkpoint/" --output_name="stack-llama-2"

# command to merge:
# python3 merge_adapter_into_base.py --base_model_name="meta-llama/Llama-2-7b-hf" \
#     --adapter_model_name="/home/anmol/nips_challenge/efficiency_challenge_repo/code/00_starter_repo/neurips_llm_efficiency_challenge/sample-submissions/llama_recipes/models_saved/128_128_70d8ee38-5410-4304-bd17-dca01d2146f7/best_model_yet_epoch_4" \
#         --output_name="/home/anmol/nips_challenge/efficiency_challenge_repo/code/00_starter_repo/neurips_llm_efficiency_challenge/sample-submissions/llama_recipes/models_saved/128_128_70d8ee38-5410-4304-bd17-dca01d2146f7/WHOLE_best_model_yet_epoch_4"

# python3 merge_adapter_into_base.py --base_model_name="meta-llama/Llama-2-7b-hf" \
#     --adapter_model_name="/home/anmol/nips_challenge/efficiency_challenge_repo/code/00_starter_repo/neurips_llm_efficiency_challenge/sample-submissions/llama_recipes/models_saved/32_32_8fa6081f-d6df-4d0b-95d1-f209962c35b1/best_model_yet_epoch_8" \
#         --output_name="/home/anmol/nips_challenge/efficiency_challenge_repo/code/00_starter_repo/neurips_llm_efficiency_challenge/sample-submissions/llama_recipes/models_saved/32_32_8fa6081f-d6df-4d0b-95d1-f209962c35b1/WHOLE_best_model_yet_epoch_8"
