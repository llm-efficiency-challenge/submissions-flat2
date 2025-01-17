from transformers import pipeline
import transformers
import deepspeed
import torch
import os
from transformers.models.t5.modeling_t5 import T5Block

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

pipe = pipeline(
    "text2text-generation",
    model="google/t5-v1_1-small",
    device=local_rank
    )

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.float,
    injection_policy={T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}
)

pipe.device = torch.device(f'cuda:{local_rank}')
output = pipe("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy")

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    print(output)
