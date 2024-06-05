# NeurIPS Large Language Model Efficiency Challenge

## Introduction
This is the submission of team PassionGPT for the [NeurIPS 2023 Large Language Model Efficiency Challenge](https://llm-efficiency-challenge.github.io/)


## Team Information
- Team name: PAssionGPT
- Team Lead: Yongquan Lai (ranchlai@163.com)
- Team Members: Yongquan Lai, Li Kang, Jing Wang, Kaihe Xu
- Affiliation:  PingAn Group


## Model Description

We use [Qwen-14B](https://huggingface.co/Qwen/Qwen-14B) for our submission. The model is trained with GPTQ-4bit + LoRA method. 

The quantized model is found in [huggingface](https://huggingface.co/kl1996/qwen-gptq). We will release the quantize code soon.

The model is quantized to 4-bit (at training stage within the 24-h limit) and trained with LoRA method. The model is trained with around 7000 to 15000 steps with weight decay 0.3. The model is trained with 1 4090 GPU in about 6 hours. We use the allowed dataset for training. Details will be released soon.

- Dockerfile1: [Qwen-v12a-7000](https://huggingface.co/kl1996/qwen-v12a-7000), using data our data v12a checkpoint-7000
- Dockerfile2: [Qwen-v12a-15000](https://huggingface.co/kl1996/qwen-v12a-15000), using data our data v12a checkpoint-15000
- Dockerfile3: [Qwen-v13a-9000](https://huggingface.co/kl1996/qwen-v13a-9000), using data our data v13a checkpoint-9000


## How to run inference: Quick Start
To make the evaluation easy, we follow the examples in the [official repo](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/tree/master/sample-submissions/llama_recipes). 

Simply run the following commands, one by one, to test our THREE submissions.
For each command, the docker will be running on port 8080. 
```bash
# first submission
>>> bash run1.sh
# second submission
>>> bash run2.sh
# thrid submission
>>> bash run3.sh
```


In case when you want to run the docker manually, you can run the following:
```bash
>>> sub=1 # or 2, 3
>>> docker build -f ./Dockerfile$sub -t qwen_recipes_inference .  
>>> docker run -it --gpus "device=0" -p 8080:80 qwen_recipes_inference # without /bin/bash, the docker will run the server
```


## Test

To test the inference docker we can run this query:

```bash
>>> curl -X POST -H "Content-Type: application/json" -d '{"text": "What is the capital of france? "}' http://localhost:8080/tokenize
>>> curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the capital of france? "}' http://localhost:8080/process


```


## NOTES
To pass the HELM test, we have a little code to extract answers from the output of the model
- Add "." to gsm8k outputs
- Parse choices from A to D etc

These steps are not needed for normal chatting scenarios, in which cases you can use the `/chat` endpoint directly.
```bash
>>> curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the capital of france? "}' http://localhost:8080/chat
>>> {"text":"Paris is the capital of france"}
```

## Acknowledgement
Thanks the ORGANIZERS for holding this challenge. We have learned a lot from this challenge.
