## IMPORTANT: This module is adapted from [Deepspeed-Chat](https://arxiv.org/abs/2308.01320).
We chose to start our work with deepspeed-chat because it offers:
1. **end-to-end three-stage OpenAI InstructGPT training strategy with Reinforcement Learning Human Feedback (RLHF).** Note: RLHF has limited use case due to the competition's rule and evaluation scenarios;
2. **DeepSpeed Hybrid Engine**: Fast, affordable and scalable RLHF training, built upon your DeepSpeed's system capability such as ZeRO technologies and DeepSpeed-Inference;

```
@article{yao2023dschat,
  title={{DeepSpeed-Chat: Easy, Fast and Affordable RLHF Training of ChatGPT-like Models at All Scales}},
  author={Zhewei Yao and Reza Yazdani Aminabadi and Olatunji Ruwase and Samyam Rajbhandari and Xiaoxia Wu and Ammar Ahmad Awan and Jeff Rasley and Minjia Zhang and Conglong Li and Connor Holmes and Zhongzhu Zhou and Michael Wyatt and Molly Smith and Lev Kurilenko and Heyang Qin and Masahiro Tanaka and Shuai Che and Shuaiwen Leon Song and Yuxiong He},
  journal={arXiv preprint arXiv:2308.01320},
  year={2023}
}
```

## Quick start (delete later)

### 2 hours example
Toy-sample (~2 hours): a **1.3B** model with a single dataset to test deepspeed-chat.<br>
? Where is the toy-sample checkpoint?

  ```bash
  python train.py --actor-model facebook/opt-1.3b --reward-model facebook/opt-350m --deployment-type single_gpu
  ```

  See the following table for the E2E time breakdown for training a 1.3 billion parameter ChatGPT model via DeepSpeed-Chat on a single commodity NVIDIA A6000 GPU with 48GB memory.

  | Model Size (A6000-48G)            | Step 1  | Step 2  | Step 3 | Total  |
  | --------------------------------- | ------- | ------- | ------ | ------ |
  | Actor: OPT-1.3B  Reward: OPT-350M | 2900 Sec | 670 Sec | 1.2hr | 2.2hr |

 </p></details>


#### 🕐 Step 1 - [Supervised Fine-Tuning](./training/step1_supervised_finetuning)

```bash
# Move into the first step of the pipeline
cd training/step1_supervised_finetuning/

# Run the training script
bash training_scripts/opt/single_gpu/run_1.3b.sh

# Evaluate the model
bash evaluation_scripts/run_prompt.sh
```

#### 🕑 Step 2 - [Reward Model](./training/step2_reward_model_finetuning)


```bash
# Move into the second step of the pipeline
cd training/step2_reward_model_finetuning

# Run the training script
bash training_scripts/opt/single_gpu/run_350m.sh

# Evaluate the model
bash evaluation_scripts/run_eval.sh
```

#### 🕒 Step 3 - [Reinforcement Learning with Human Feedback](./training/step3_rlhf_finetuning)

```bash
# Move into the final step of the pipeline
cd training/step3_rlhf_finetuning/

# Run the training script
bash training_scripts/opt/single_gpu/run_1.3b.sh
```

## DeepSpeed Chat Resources:
1. [Release Blog](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-chat) 
2. [Training Performance Evaluation](#-training-performance-evaluation-).  
3. [Documentation and Tutorial](#-documentation-and-tutorial-).



## Adding datasets
1. add a new Class in[`training/utils/data/raw_datasets.py` to define the format when using your data. You need to make sure to follow the APIs and format defined in the PromptRawDataset class to ensure a consistent data format that DeepSpeed-Chat relies on. 
2. add an if condition in function get_raw_dataset in `training/utils/data/data_utils.py` corresponding to your new dataset. The dataset_name string in the if condition should be the dataset name you will provide as a arg for the training scripts. Last, you need to add your new dataset's dataset_name into your "--data_path" arg in your training scripts. NOTE: you should not make `data/` in your local path, it may cause an exception to `load_dataset`.

[TODO: better understand this part.]
One thing to note that some datasets may only have one response instead of two responses. For those datasets, you can only use them in step 1. And in such case, you should add the dataset_name as part of the "--sft_only_data_path" arg instead of the "--data_path" arg. One thing to note is that: If you plan to only do step 1 SFT, adding more single-response datasets is definitely beneficial. However, if you do plan to do steps 2 and 3, then adding too many single-response datasets during SFT could backfire: these data could be different from the data used for steps 2/3, generating different distributions which could cause training instability/worse model quality during step 2/3. That is part of the reason why we focused on trying the datasets with two responses and the preference, and always split a dataset into all 3 steps.

If you have your own dataset in local files, you can also use it by following these rules:
* Pass "local/jsonfile" as the dataset name to the "--data_path" argument.
* Put your train data and evaluation data in applications/DeepSpeed-Chat/data/ with name train.json and eval.json.
* The json data in file should be a single list with each item like ***{"prompt": "Human: I have a question. Assistant:", "chosen": "Good answer.", "rejected": "Bad answer."}***.

What is more, when you use your own dataset files and modified some data in them, pay attention to the parameter "reload" of ***create_prompt_dataset*** function. You should pass a True value to it or the cache files will not refresh.

## Customizing RLHF training pipeline using DeepSpeed-Chat’s RLHF APIs

DeepSpeed-Chat allows users to build their very own RLHF training pipeline using our flexible APIs shown below, which users can use to reconstruct their own RLHF training strategy. This enables a general interface and backend for creating a wide range of RLHF algorithms for research exploration.

```python
engine = DeepSpeedRLHFEngine(
  actor_model_name_or_path=args.actor_model_name_or_path,
  critic_model_name_or_path=args.critic_model_name_or_path,
  tokenizer=tokenizer,
  num_total_iters=num_total_iters,
  args=args)

trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
  out = trainer.generate_experience(prompt_batch)
  actor_loss, critic_loss = trainer.train_rlhf(out)

```

## Serving: Plug-in your final model trained by DeepSpeed-Chat and test it out!
For quickly testing your final models trained by DeepSpeed-Chat, we provide a simple script below.

```bash
# serve the final model
python inference.py --path  ${PATH-to-your-actor-model}
```

## Training Cost

A comprehensive view of the scale and end-to-end training times enabled by DeepSpeed-RLHF system are presented in Table 1. It also demonstrates the most cost-effective way to train models in Azure Cloud along with the associated cost.


| GPU SKUs      | Llama2-7B      | Llama2-13B      | Llama2-70B      | 
|---------------|---------------|----------------|-----------------|
| 1x RTX4090 24G | ? hours      |                |                 |
<p align="center">
Table 1. End-to-end RLHF training (Step 3) for different actor model sizes and a fixed ?B critical model running on RTX4090 with ?M tokens.
</p>

NOTE: 6 open-sourced datasets with 40% used for RLHF training stage, i.e., Dahoas/rm-static, Dahoas/full-hh-rlhf, Dahoas/synthetic-instruct-gptj-pairwise, yitingxie/rlhf-reward-datasets, openai/webgpt_comparisons, and stanfordnlp/SHP from Huggingface Datasets. More specifically, we have in total 67.5M query tokens (131.9k queries with sequence length 256) and 67.5M generated tokens (131.9k answers with sequence length 256), and a maximum global batch size per step of 0.5M tokens (1024 query-answer pairs).
[llama2](https://huggingface.co/models?sort=trending&search=meta-llama%2FLlama-2) | 7B, 13B  | We provide full system support and scripts to try 7B and 13B models.*
[llama2-70b](https://huggingface.co/models?sort=trending&search=meta-llama%2FLlama-2-70b) | 70B  | Llama-2-70B is supported through MixZ++, ZeRO-Offload but not Hybrid Engine.

## Yet to figure out
1. flash attention
2. sophia
3. quantized training/ QLORA/ pruned model training.
