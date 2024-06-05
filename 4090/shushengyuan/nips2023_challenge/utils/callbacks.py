import os
import time

import torch
import transformers
import torch.cuda.nvtx as nvtx
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

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

class TimeLimitCallback(transformers.TrainerCallback):
    def __init__(self, max_training_time,logger):
        self.max_training_time = max_training_time
        self.start_time = time.time()
        self.logger = logger
                        
    def on_step_end(self, args, state, control, model, **kwargs):
        elapsed_time = time.time() - self.start_time
        self.logger.info(f"elapsed_time: {elapsed_time}")
        if elapsed_time > self.max_training_time:
            control.should_training_stop = True
            print(f"Training time limit ({self.max_training_time} seconds) reached.")
                
class ProfileCallback(transformers.TrainerCallback):
        "A callback that prints a message at the beginning of training"
        
        def __init__(self, prof):
            self.prof = prof
            self.count = 0
            self.eval_count = 0

        def on_init_end(self, args, state, control, **kwargs):
            print("After init, allocated GPU mem: {torch.cuda.memory_allocated()} GB")
        
        def on_step_begin(self, args, state, control, **kwargs):
            nvtx.range_push("Batch" + str(self.count+1))
            self.count += 1
        
        def on_step_end(self, args, state, control, model, **kwargs):
            self.prof.step()
            nvtx.range_pop();

        def on_prediction_step(self, args, state, control, model, **kwargs):
            nvtx.range_pop();
            nvtx.range_push("Eval" + str(self.eval_count+1))
            self.eval_count += 1

class ExponentialMovingAveragingCallback(transformers.TrainerCallback):
    def __init__(self, decay, model):
        self.decay = decay
        self.average_model = {}
        self.model_weights = {}

    def on_init_end(self, args: transformers.TrainingArguments, state:  transformers.TrainerState, control:  transformers.TrainerControl, model, **kwargs):

        for name, param in model.named_parameters():
           if param.requires_grad:
               self.average_model[name] = param.data.clone()
               self.model_weights[name] = param.data.clone()

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
