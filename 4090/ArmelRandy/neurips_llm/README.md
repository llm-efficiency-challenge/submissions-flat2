# Neurips 1 LLM 1 GPU Challenge
- [Presentation of the challenge](https://llm-efficiency-challenge.github.io)
- [Starter kit](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge)

# Submissions

This repository contains 3 submissions for the 4090 track. Each submission is a folder named submission_{track}_{index} where $index$ is the index of the submission (going from 1 to 3) and $track$ is either A100 or 4090. Each folder has 4 files :
- Dockerfile
- main.py
- requirements.txt
- api.py

# Team description

We are a team of 3 students.

# Overview
This challenge aims to fine-tune a open LLM (chosen in [this list](https://llm-efficiency-challenge.github.io/challenge)) on an open dataset (constructed with the help of [these datasets](https://llm-efficiency-challenge.github.io/challenge)). The model should perform well when evaluated on a given set of benchmarks with help. These are :
- [BigBench](https://github.com/google/BIG-bench) $\rightarrow$ [helm](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/big_bench_scenario.py)
- [MMLU](https://github.com/hendrycks/test) $\rightarrow$ [helm](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/mmlu_scenario.py)
- [TruthfulQA (Multiple Choice Single value)](https://github.com/sylinrl/TruthfulQA) $\rightarrow$ [helm](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/truthful_qa_scenario.py)
- [CNN/DailyMail](https://github.com/deepmind/rc-data) $\rightarrow$ [helm](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/summarization_scenario.py)
- [GSM8k](https://github.com/openai/grade-school-math) $\rightarrow$ [helm](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/gsm_scenario.py)
- [BBQ](https://github.com/nyu-mll/BBQ) $\rightarrow$ [helm](https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/bbq_scenario.py)

# Models
We are allowed to use a broad set of open source LLM. However, our recent experiments have shown that only a few of them are relevant :
- [Qwen](https://huggingface.co/Qwen/Qwen-14B)
- [Mistral](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [Llama 2](https://huggingface.co/NousResearch/Llama-2-13b-hf)

Here are some benchmarked models

|Benchmarks                              | Llama-2-7b| Llama-2-13b | Qwen-14b | Mistral-7b|
|----------------------------------------|-----------|-------------|----------|-----------|
|MMLU - EM                               |0.47       |0.54         |0.70      |0.64       |
|CNN/DailyMail - ROUGE-2                 |0.16       |0.14         |0.04      |0.10       |
|TruthfulQA - EM                         |0.25       |0.36         |0.50      |0.60       |
|BBQ - EM                                |0.46       |0.62         |0.80      |0.82       |
|GSM8K - EM                              |0.00       |0.00         |0.00      |0.00       |
|MMLU - EM (Robustness)                  |0.40       |0.51         |0.66      |0.62       |
|TruthfulQA - EM (Robustness)            |0.25       |0.36         |0.50      |0.60       |
|MMLU - EM (Fairness)                    |0.40       |0.48         |0.68      |0.60       |
|TruthfulQA - EM (Fairness)              |0.25       |0.36         |0.50      |0.60       |
|CNN/DailyMail - Stereotypes (race)      |0.67       |0.67         |0.00      |0.00       |
|CNN/DailyMail - Stereotypes (gender)    |0.38       |0.27         |0.50      |0.35       |
|CNN/DailyMail - Representation (race)   |0.52       |0.51         |0.42      |0.40       |
|CNN/DailyMail - Representation (gender) |0.15       |0.14         |0.50      |0.16       |

# Datasets
We want to combine several datasets in order to build a single high quality one that will help improve the model on every single benchmark.

- [Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [Open Assistant](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [LIMA](https://huggingface.co/datasets/GAIR/lima)
- [Flan Collection](https://github.com/google-research/FLAN/tree/main/flan/v2)
  - [Flan2021](https://huggingface.co/datasets/conceptofmind/flan2021_submix_original)
  - [T0](https://huggingface.co/datasets/conceptofmind/t0_submix_original)
  - [Dialog](https://huggingface.co/datasets/conceptofmind/dialog_submix_original)
  - [CoT](https://huggingface.co/datasets/conceptofmind/cot_submix_original)
  - [Niv2](https://huggingface.co/datasets/conceptofmind/niv2_submix_original)
 
# Fine-tuning
In the competition, there are two tracks. The track A100 40G and the track 4090 (24G). We want to fine-tune the best base model that can fit into the GPU memory. For this purpose, we'll use two well known techniques : parameter-efficient fine-tuning and quantization. Fortunately, these two options are supported by the Hugging Face environment via multiple libraries such as [transformers](https://github.com/huggingface/transformers), [accelerate](https://github.com/huggingface/accelerate) and [peft](https://github.com/huggingface/peft).

Our training file is `train.py` and it is pretty much self-explanatory. Here is an example command
```
!python train.py \
  --model_path "Qwen/Qwen-14B"\
  --dataset_name "ArmelRandy/more_precious"\
  --seq_length 2048\
  --input_column_name "prompt"\
  --max_steps 1000\
  --batch_size 1\
  --gradient_accumulation_steps 32 \
  --lora_r 16 \
  --lora_alpha 64 \
  --lora_dropout 0.1 \
  --no_gradient_checkpointing \
  --target_modules "c_attn c_proj w1 w2"\
  --learning_rate 1e-4\
  --lr_scheduler_type "cosine"\
  --num_warmup_steps 10\
  --weight_decay 0.05\
  --log_freq 10\
  --eval_freq 10\
  --save_freq 10\
  --output_dir "./checkpoints-qwen-14b"\
  --use_flash_attn
```

# Results


|Benchmarks                                 | 10  |  30 | 50  |  110 |10 MP|20 OA+SQA|35 OA+SQA| 50(32) 0A+SQA | 20 sec-qa| 150 sec-qa|10  |20    |40    |10 OA|20 OA|30 OA|
|-------------------------------------------|-----|-----|-----|------|-----|---------|---------|---------------|----------|-----------|----|------|------|-----|-----|-----|
|MMLU - EM                                  |0.70 | 0.68| 0.66|0.64  | 0.70| 0.71    |   0.70  |0.70           |0.70      |0.69       |0.70| 0.70 | 0.70 |0.70 |0.71 |0.70 |
|CNN/DailyMail - ROUGE-2                    |0.11 | 0.14| 0.11|0.09  | 0.11| 0.14    |   0.13  |0.12           |0.12      |0.13       |0.11| 0.12 | 0.10 |0.05 |0.13 |0.12 |
|TruthfulQA - EM                            |0.52 | 0.56| 0.52|0.52  | 0.58| 0.64    |   0.62  |0.62           |0.58      |0.48       |0.54| 0.62 | 0.56 |0.64 |0.66 |0.62 |
|BBQ - EM                                   |0.70 | 0.56| 0.48|0.52  | 0.68| 0.70    |   0.72  |0.72           |0.68      |0.60       |0.72| 0.60 | 0.56 |0.76 |0.72 |0.72 |
|MMLU - EM (Robustness)                     |0.65 | 0.63| 0.62|0.61  | 0.65| 0.66    |   0.66  |0.66           |0.65      |0.65       |0.65| 0.65 | 0.65 |0.67 |0.66 |0.64 |
|TruthfulQA - EM (Robustness)               |0.55 | 0.56| 0.52|0.52  | 0.58| 0.64    |   0.62  |0.62           |0.58      |0.48       |0.54| 0.62 | 0.56 |0.64 |0.66 |0.62 |
|MMLU - EM (Fairness)                       |0.62 | 0.64| 0.62|0.59  | 0.65| 0.66    |   0.65  |0.66           |0.66      |0.65       |0.66| 0.65 | 0.66 |0.66 |0.67 |0.66 |
|TruthfulQA - EM (Fairness)                 |0.52 | 0.56| 0.52|0.52  | 0.58| 0.64    |   0.62  |0.62           |0.58      |0.48       |0.54| 0.62 | 0.56 |0.64 |0.66 |0.62 |
|CNN/DailyMail - Stereotypes (race)         |0.50 | 0.67| 0.67|0.67  | 0.67| 0.67    |   0.67  |0.67           |0.67      |0.67       |0.67| 0.67 | 0.67 |0.67 |0.67 |0.53 |
|CNN/DailyMail - Stereotypes (gender)       |0.43 | 0.48| 0.44|0.46  | 0.40| 0.39    |   0.44  |0.50           |0.36      |0.30       |0.38| 0.40 | 0.37 |0.43 |0.33 |0.44 |
|CNN/DailyMail - Representation (race)      |0.33 | 0.37| 0.39|0.50  | 0.27| 0.33    |   0.42  |0.52           |0.36      |0.50       |0.33| 0.48 | 0.40 |0.33 |0.33 |0.42 |
|CNN/DailyMail - Representation (gender)    |0.09 | 0.19| 0.21|0.03  | 0.13| 0.15    |   0.12  |0.18           |0.08      |0.22       |0.18| 0.06 | 0.11 |0.02 |0.02 |0.11 |
|Score                                      |     |     |     |      |     | *       |         |               |          |           |    |      |      |     |A100 |     |

|20 CoT |10 Lima|20 Lima|30 Lima|10 0AenLima |20 OAenLima|30 OAenLima|10 OAen|15OAen|20 OAen|25 0Aen|10 LIbias|15 LIbias |20 LIbias|15 OAbias|20 OAbias|15 OAsum|25 OAsum|75 OAsum|
|-------|-------|-------|-------|------------|-----------|-----------|-------|------|-------|-------|---------|----------|---------|---------|---------|--------|--------|--------|
| 0.70  |0.70   |0.69   | 0.70  |   0.71     |   0.70    |   0.69    | 0.71  | 0.70 |  0.70 |  0.71 |  0.70   |  0.70    |  0.69   |0.70     |0.70     | 0.71   |  0.71  |  0.71  |
| 0.11  |0.05   |0.13   | 0.12  |   0.05     |   0.15    |   0.14    | 0.05  | 0.13 |  0.14 |  0.12 |  0.12   |  0.06    |  0.14   |0.11     |0.12     | 0.15   |  0.16  |  0.14  |
| 0.54  |0.64   |0.64   | 0.60  |   0.54     |   0.62    |   0.54    | 0.64  | 0.66 |  0.66 |  0.66 |  0.64   |  0.66    |  0.54   |0.66     |0.66     | 0.58   |  0.58  |  0.60  |
| 0.74  |0.76   |0.72   | 0.68  |   0.74     |   0.72    |   0.72    | 0.76  | 0.74 |  0.76 |  0.74 |  0.72   |  0.72    |  0.72   |0.74     |0.74     | 0.72   |  0.66  |  0.70  |
| 0.66  |0.67   |0.65   | 0.66  |   0.67     |   0.66    |   0.65    | 0.66  | 0.66 |  0.66 |  0.66 |  0.66   |  0.66    |  0.65   |0.66     |0.65     | 0.66   |  0.65  |  0.66  |
| 0.54  |0.64   |0.64   | 0.60  |   0.54     |   0.62    |   0.54    | 0.64  | 0.66 |  0.66 |  0.66 |  0.64   |  0.66    |  0.54   |0.66     |0.66     | 0.58   |  0.58  |  0.60  |
| 0.66  |0.66   |0.66   | 0.66  |   0.67     |   0.66    |   0.66    | 0.66  | 0.66 |  0.66 |  0.66 |  0.66   |  0.66    |  0.66   |0.67     |0.66     | 0.66   |  0.67  |  0.66  |
| 0.54  |0.64   |0.64   | 0.60  |   0.54     |   0.62    |   0.54    | 0.64  | 0.66 |  0.66 |  0.66 |  0.64   |  0.66    |  0.54   |0.66     |0.66     | 0.58   |  0.58  |  0.60  |
| 0.67  |0.67   |0.33   | 0.67  |    -       |    -      |   0.67    | 0.67  |  -   |   -   |  0.67 |   -     |    -     |  0.67   | -       |0.67     | 0.67   |  0.67  |  0.67  |
| 0.36  |0.43   |0.45   | 0.35  |   0.50     |   0.31    |   0.48    | 0.42  | 0.50 |  0.38 |  0.43 |  0.43   |    -     |  0.48   |0.50     |0.45     | 0.36   |  0.42  |  0.36  |
| 0.45  |0.33   |0.33   | 0.33  |   0.33     |   0.33    |   0.44    | 0.47  | 0.47 |  0.43 |  0.38 |  0.33   |  0.42    |  0.44   |0.42     |0.50     | 0.38   |  0.26  |  0.33  |
| 0.10  |0.02   |0.06   | 0.02  |   0.03     |   0.12    |   0.11    | 0.22  | 0.23 |  0.17 |  0.11 |  0.10   |  0.03    |  0.11   |0.22     |0.13     | 0.08   |  0.12  |  0.20  |
|       |       |4090   |       |            |           |           |       |      |       |       |         |          |         |         |         |        |        |        |


| 50 | 80  |100 | 110 | 120 | 130 | 140 | 150  | 160 | 165 | 170 | 180 | 190 |200 | 250 | 300 | 50 | 80 | 90 | 100 | 50  | 100 | 150 | 100 | 150 | 160 | 200 | 240 |
|----|-----|----|-----|-----|-----|-----|------|-----|-----|-----|-----|-----|----|-----|-----|----|----|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
|0.70| 0.69|0.70| 0.69| 0.69| 0.69| 0.69| 0.69 | 0.69| 0.68| 0.69|0.69 | 0.69|0.69|0.68 | 0.68|0.68|0.67|0.68| 0.68| 0.69| 0.68|0.66 | 0.70| 0.69| 0.69| 0.68| 0.69|
|0.14| 0.13|0.13| 0.11| 0.14| 0.14| 0.14| 0.13 | 0.15| 0.14| 0.12|0.15 | 0.15|0.12|0.13 | 0.13|0.11|0.11|0.12| 0.11| 0.11| 0.12|0.12 | 0.14| 0.13| 0.16| 0.12| 0.14|
|0.72| 0.72|0.74| 0.74| 0.74| 0.74| 0.74| 0.74 | 0.74| 0.76| 0.74|0.78 | 0.78|0.76|0.78 | 0.80|0.74|0.76|0.76| 0.74| 0.70| 0.74|0.76 | 0.74| 0.74| 0.72| 0.78| 0.78|
|0.88| 0.90|0.92| 0.92| 0.90| 0.92| 0.96| 0.96 | 0.96| 0.96| 0.76|0.92 | 0.94|0.94|0.94 | 0.94|0.94|0.94|0.96| 0.98| 0.92| 0.96|0.98 | 0.92| 0.96| 0.96| 0.94| 0.94|
|0.64| 0.64|0.64| 0.64| 0.64| 0.63| 0.64| 0.63 | 0.64| 0.63| 0.64|0.64 | 0.63|0.64|0.62 | 0.63|0.63|0.62|0.62| 0.62| 0.65| 0.64|0.62 | 0.64| 0.64| 0.65| 0.64| 0.62|
|0.72| 0.72|0.74| 0.74| 0.74| 0.74| 0.74| 0.74 | 0.74| 0.76| 0.74|0.78 | 0.78|0.76|0.78 | 0.80|0.74|0.76|0.76| 0.74| 0.70| 0.74|0.76 | 0.74| 0.74| 0.72| 0.78| 0.78|
|0.65| 0.64|0.65| 0.64| 0.65| 0.64| 0.65| 0.64 | 0.64| 0.64| 0.64|0.64 | 0.63|0.64|0.64 | 0.63|0.64|0.63|0.63| 0.63| 0.65| 0.64|0.63 | 0.65| 0.64| 0.65| 0.64| 0.64|
|0.72| 0.72|0.74| 0.74| 0.74| 0.74| 0.74| 0.74 | 0.74| 0.76| 0.74|0.78 | 0.78|0.76|0.78 | 0.80|0.74|0.76|0.76| 0.74| 0.70| 0.74|0.76 | 0.74| 0.74| 0.72| 0.78| 0.78|
|0.67| 0.67|0.67| 0.67| 0.67| 0.67| 0.67| 0.67 | 0.33| 0.67| 0.67|0.67 | 0.67|0.54|0.67 | 0.67|0.67| -  |0.67| 0.67|  -  |  -  |0.50 | 0.67| 0.67| 0.67| 0.67| 0.67|
|0.40| 0.31|0.35| 0.25| 0.32| 0.24| 0.33| 0.18 | 0.33| 0.40| 0.39|0.35 | 0.37|0.36|0.17 | 0.39|0.20|0.28|0.42| 0.26| 0.38| 0.50|0.48 | 0.41| 0.22| 0.30| 0.24| 0.21|
|0.47| 0.47|0.38| 0.47| 0.38| 0.38| 0.33| 0.33 | 0.33| 0.33| 0.33|0.33 | 0.33|0.33|0.52 | 0.33|0.42|0.52|0.42| 0.38| 0.33| 0.38|0.42 | 0.50| 0.38| 0.38| 0.38| 0.50|
|0.17| 0.12|0.13| 0.08| 0.15| 0.26| 0.21| 0.01 | 0.20| 0.13| 0.01|0.30 | 0.31|0.08|0.09 | 0.08|0.02|0.04|0.01| 0.01| 0.06| 0.06|0.07 | 0.25| 0.02| 0.16| 0.06| 0.12|
|    |     |    |     |     |     |     | ***  |     |     |     |     |     |    |4090 |     |    |    |    |     |     |     |     |     |     |     | 4090|     |

| 35 | 40 | 45 | 50 | 55 | 60 | 100 |
|----|----|----|----|----|----|-----|
|0.69|0.70|0.70|0.70|    |0.69| 0.68|
|0.16|0.14|0.13|0.13|    |0.13| 0.13|
|0.78|0.76|0.70|0.68|    |0.72| 0.70|
|0.90|0.90|0.94|0.92|    |0.90| 0.94|
|0.64|0.64|0.64|0.64|    |0.64| 0.62|
|0.78|0.76|0.70|0.68|    |0.72| 0.70|
|0.65|0.66|0.66|0.65|    |0.65| 0.64|
|0.78|0.76|0.70|0.68|    |0.72| 0.70|
|0.67|0.67|0.67|0.67|    |0.67| 0.67|
|0.30|0.35|0.18|0.29|    |0.26| 0.50|
|0.33|0.33|0.33|0.52|    |0.38| 0.44|
|0.06|0.04|0.02|0.09|    |0.00| 0.02|
|4090|    |A100|    |    |    |     |




- TQA+BBQ+summarization + OAen (most precious)
- TQA+BBQ+OAen
- TQA+BBQ+OAen+MMLU
- Most precious neft
- Most precious 4 neft

|Parameters                  |                     |   |   |   |                          |                            |
|----------------------------|-------------------- |---|---|---|--------------------------|----------------------------|
|model_name                  |Qwen/Qwen-14B        | - | - | - |           -              |                            |
|dataset_name                |ArmelRandy/precious  | - | - | - |ArmelRandy/more_precious  | ArmelRandy/oasst_strategy  |
|sequence length             |2048                 | - | - | - |                          |                            |
|max_steps                   |1000                 | - | - | - |           -              |                            |
|batch_size                  |1                    | - | - | - |           -              |                            |
|gradient_accumulation_steps |32                   | - | - | - |           -              | 16                         |
|lora_r                      |16                   | - | - | - |           -              |                            |
|lora_alpha                  |64                   | - | - | - |           -              |                            |
|lora_dropout                |0.1                  | - | - | - |           -              |                            |
|no_gradient_checkpointing   |False                | - | - | - |           -              |                            |
|target_modules              |c_attn c_proj w1 w2  | - | - | - |           -              |                            |
|num_warmup_steps            | 10                  | - | - | - |           -              |                            |

- wandb links
  - [1, 2, 3, 4](https://wandb.ai/armezebaze/huggingface/runs/6g6prfvk?workspace=user-armezebaze)
  - [5](https://wandb.ai/armezebaze/huggingface/runs/y2ui3s3u?workspace=user-armezebaze)
  - [6](https://wandb.ai/armezebaze/huggingface/runs/1dzx957g?workspace=user-armezebaze)
  - [7](https://wandb.ai/armezebaze/huggingface/runs/vuindafm?workspace=user-armezebaze)
  - [8](https://wandb.ai/armezebaze/huggingface/runs/af90ezrp?workspace=user-armezebaze)
  - [9, 10](https://wandb.ai/armezebaze/huggingface/runs/i97fpo3h?workspace=user-armezebaze)
  - [12](https://wandb.ai/armezebaze/huggingface/runs/5qposwvk?workspace=user-armezebaze)
- datasets
  - [OA + strategy QA]()
  - Second qa = OA + qcm (1500) + cot_gsm8k(Q:/A:) (1000) + strategyqa (1000) + lima (1000) + nli (500) + summary (800).
  - F (four=4) = OA+Lima+StrategyQA+qcm
  - OA (ArmelRandy/oasst1)
# Intuitions and relevant information
- Mistral is difficult to improve via fine-tuning. It probably requires a good mixture of datasets.
- Long fine-tuning can be deleterious for the MMLU scores but it tends to improve TruthfulQA
- We need to find a way to improve (or not to impair) the performance of our models on BBQ.

# Acknowledgments
