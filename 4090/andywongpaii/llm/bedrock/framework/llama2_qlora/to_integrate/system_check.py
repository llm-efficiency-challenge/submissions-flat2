import torch

def check_bf16_support():
    # Reference:
    # https://colab.research.google.com/drive/1PEQyJO1-f6j0S_XJ8DV50NkpzasXkrzd?usp=sharing#scrollTo=ib_We3NLtj2E

    support = None

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)
            support = True
        else:
            support = False
            raise ValueError("Your GPU DO NOT supports bfloat16. Please check system setup.")
        
    return support