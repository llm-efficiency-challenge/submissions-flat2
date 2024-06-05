import os
from os.path import join
import torch
import transformers

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from copy import deepcopy
from utils.util import (
    is_ipex_available,
    find_all_linear_names,
    smart_tokenizer_and_embedding_resize,
)
import evaluate
from tqdm import tqdm

DEFAULT_PAD_TOKEN = "[PAD]"
IGNORE_INDEX = -100


def save_model(args, state, model):
    print('Saving PEFT checkpoint...')
    if state.best_model_checkpoint is not None:
        checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
    else:
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}-ema")

    peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
    model.save_pretrained(peft_model_path)

    pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
    if os.path.exists(pytorch_model_path):
        os.remove(pytorch_model_path)


class EpochwiseExponentialMovingAveragingCallback(transformers.TrainerCallback):
    def __init__(self, decay, model):
        self.decay = decay
        self.average_model = {}
        self.model_weights = {}

    def on_init_end(self, args: transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):

        for name, param in model.named_parameters():
           if param.requires_grad:
               self.average_model[name] = param.data.clone()
               self.model_weights[name] = param.data.clone()
    
    def on_epoch_end(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        print("update ema model...")
        self.update_parameters(model)
        if control.should_evaluate:
            print("need to evaluate, chaging model to ema model")
            self.transfer_weights(model, self.model_weights)
            self.transfer_weights(self.average_model, model)

    def on_evaluate(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        print("start evaluating, change model to original.")
        self.transfer_weights(self.model_weights, model)

    def on_save(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        print("save original model")
        print("changing to ema")
        self.transfer_weights(self.average_model, model)
        
        save_model(args, state, model)

        print("changing to original")
        self.transfer_weights(self.model_weights, model)

    @staticmethod
    def transfer_weights(src_model, dst_model):
        if isinstance(src_model, dict):
            for name, param in dst_model.named_parameters():
                if name in src_model.keys():
                    param.data = src_model[name]
        else:
             for name, param in src_model.named_parameters():
                if name in dst_model.keys():
                    dst_model[name] = param.data.clone()

    def update_parameters(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.average_model
                new_average =  (1.0 - self.decay) * param.data + self.decay * self.average_model[name]
                self.average_model[name] = new_average.clone()

class ExponentialMovingAveragingCallback(transformers.TrainerCallback):
    def __init__(self, decay, model):
        self.decay = decay
        self.average_model = {}
        self.model_weights = {}

    def on_init_end(self, args: transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):

        for name, param in model.named_parameters():
           if param.requires_grad:
               self.average_model[name] = param.data.clone().to("cpu")
               self.model_weights[name] = param.data.clone().to("cpu")

    def on_step_end(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        print("update ema model...")
        self.update_parameters(model)

    def on_epoch_end(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        if control.should_evaluate:
            print("need to evaluate, changing model to ema model")
            self.transfer_weights(model, self.model_weights)
            self.transfer_weights(self.average_model, model)

    def on_evaluate(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        print("strat evaluation, change model to original.")
        self.transfer_weights(self.model_weights, model)

    def on_save(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        print("save original model")
        print("changing to ema")
        self.transfer_weights(self.average_model, model)
        
        save_model(args, state, model)

        print("changing to original")
        self.transfer_weights(self.model_weights, model)

    @staticmethod
    def transfer_weights(src_model, dst_model):
        if isinstance(src_model, dict):
            for name, param in dst_model.named_parameters():
                if name in src_model.keys():
                    param.data = src_model[name].to("cuda")
        else:
             for name, param in src_model.named_parameters():
                if name in dst_model.keys():
                    dst_model[name] = param.data.clone().to("cpu")

    def update_parameters(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.average_model
                new_average =  (1.0 - self.decay) * param.data + self.decay * self.average_model[name].to("cuda")
                self.average_model[name] = new_average.clone().to("cpu")

class SwitchExponentialMovingAveragingCallback(transformers.TrainerCallback):
    def __init__(self, decay, model):
        self.decay = decay
        # 保存影子权重（当前step的每一层的滑动平均权重）
        self.average_model = {}
        # 在进行evaluate的时候，保存原始的模型权重，当执行完evaluate后，从影子权重恢复到原始权重
        self.model_weights = {}

    def on_init_end(self, args: transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        # self.average_model = deepcopy(model)
        # self.model_weights = deepcopy(model)
        for name, param in model.named_parameters():
           if param.requires_grad:
               self.average_model[name] = param.data.clone()
               self.model_weights[name] = param.data.clone()

    def on_epoch_end(self, args:  transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):
        # 更新ema参数
        self.update_parameters(model)
        print("update ema to model")
        # self.transfer_weights(self.average_model, model)


    def update_parameters(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.average_model
                new_average =  (1.0 - self.decay) * param.data + self.decay * self.average_model[name]
                self.average_model[name] = new_average.clone()
                param.data = new_average.clone()


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

def get_accelerate_model(args, checkpoint_dir):

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
        
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}


    if args.full_finetune: assert args.bits in [16, 32]

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if args.type == "lora":
        model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_8bit=args.bits == 8,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=args.trust_remote_code,
        use_flash_attention_2=True,
        # use_auth_token=args.use_auth_token
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            device_map=device_map,
            max_memory=max_memory,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=args.bits == 4,
                load_in_8bit=args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=args.double_quant,
                bnb_4bit_quant_type=args.quant_type,
            ),
            use_flash_attention_2=args.use_flash_attention_2,
            torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
            trust_remote_code=args.trust_remote_code,
            # use_auth_token=args.use_auth_token,
        )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)
            
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
   
    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            cache_dir=args.cache_dir,
            padding_side="left",
            use_fast=False, # Fast tokenizer giving issues.
            tokenizer_type='llama' if 'llama' in args.model_name_or_path else None, # Needed for HF name change
            trust_remote_code=args.trust_remote_code,
            use_auth_token=args.use_auth_token,
        )

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        pad_id =  model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
        print(pad_id)
        if pad_id == None:
            tokenizer.add_special_tokens({
                "pad_token": DEFAULT_PAD_TOKEN,
        })
        tokenizer.add_special_tokens({
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(pad_id) if pad_id else DEFAULT_PAD_TOKEN,
        })
        # if 'mistral' in args.model_name_or_path:
        #     model.resize_token_embeddings(len(tokenizer))
        # print(tokenizer)
    
    if not args.full_finetune:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True)
        else:
            print(f'adding LoRA modules...')
            if args.lora_modules != ['all']:
                modules = args.lora_modules
            else:
                modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    if args.neftune:
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                dims = torch.tensor(module.weight.shape[0] * module.weight.shape[1])
                mag_norm = args.noise_alpha / torch.sqrt(dims)
                scaled_noise = torch.zeros_like(module.weight).uniform_(-mag_norm, mag_norm)
                module.weight += scaled_noise
                break
    return model, tokenizer


