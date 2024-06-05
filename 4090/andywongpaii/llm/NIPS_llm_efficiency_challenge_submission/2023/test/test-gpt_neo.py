from transformers import pipeline
import transformers
import deepspeed
import torch
import os

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

pipe = pipeline(
    'text-generation',
    # model='EleutherAI/gpt-neo-125m',
    model='EleutherAI/gpt-neo-2.7B',
    device=local_rank
    )

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    replace_with_kernel_inject=True,
)
# NOTE: On 4090, when model='EleutherAI/gpt-neo-125m', encountered issue while building torch_extensions.
# ValueError: Unfeasible length constraints: the minimum length (50) is larger than the maximum length (20)
# NOTE:encountered warning:
# /home/ndeewong/miniconda3/envs/neurips2023-train/lib/python3.9/site-packages/transformers/generation/utils.py:1369:
# UserWarning: Using `max_length`'s default (50) to control the generation length. 
# This behaviour is deprecated and will be removed from the config in v5 of Transformers -- 
# we recommend using `max_new_tokens` to control the maximum length of the generation. 
output = pipe(
    "DeepSpeed is", 
    do_sample=True, 
    min_length=50, # mute on 4090 gpt-neo-125m.
)
if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
