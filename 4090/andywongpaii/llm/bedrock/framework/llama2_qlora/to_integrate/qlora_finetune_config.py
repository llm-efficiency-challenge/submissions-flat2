import yaml
qlora_finetune_config = yaml.safe_load(
    """
    model_type: llm
    base_model: NIPS_llm_efficiency_challenge_submission/2023/Llama-2/hf-llama-2-7b
    finetuned_model: NIPS_llm_efficiency_challenge_submission/2023/Llama-2/hf-llama-2-7b-v001
    
    datasets:
      - name: mlabonne/guanaco-llama2-1k
        purpose: instuction

    input_features:
      - name: instruction
        type: text
    
    output_features:
      - name: output
        type: text
    
    prompt:
      template: >-
        Below is an instruction that describes a task, paired with an input
        that provides further context. Write a response that appropriately
        completes the request.
    
        ### Instruction: {instruction}
    
        ### Input: {input}
    
        ### Response:
    
    generation:
      temperature: 0.1
      max_new_tokens: 512
    
    adapter:
      type: lora
    
    quantization:
      bits: 4
    
    preprocessing:
      global_max_sequence_length: 512
    
    trainer:
      type: finetune
      basic:
        epochs: 5
        max_steps: -1 # Number of training steps (overrides num_train_epochs)
        batch_size: 4
        eval_batch_size: 4
        group_by_length: True # Group sequences into batches with same length; Saves memory and speeds up training considerably
        gradient_accumulation_steps: 1
        gradient_checkpointing: True
        max_grad_norm: 0.3 # (gradient clipping)
        learning_rate: 0.0002 # Initial learning rate (AdamW optimizer)
        save_steps: 0 # Save checkpoint every X updates steps
        logging_steps: 25 # Log every X updates steps
      optimizer:
        type: paged_adamw_32bit # adam
        params:
          eps: 1.e-8
          betas:
            - 0.9
            - 0.999
          weight_decay: 0.001 # Weight decay to apply to all layers except bias/LayerNorm weights
      learning_rate_scheduler:
        type: cosine
        warmup_ratio: 0.03 # Ratio of steps for a linear warmup (from 0 to learning rate)
        reduce_on_plateau: 0

    lora:
      attention_dimension: 64
      alpha: 16
      dropout: 0.1

    sft:
      max_seq_length: None # Maximum sequence length to use
      packing: False # Pack multiple short examples in the same input sequence to increase efficiency
    """
)
