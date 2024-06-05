# NeurIPS Large Language Model Efficiency Challenge

## Introduction
This is the submission of team PassionGPT for the [NeurIPS 2023 Large Language Model Efficiency Challenge](https://llm-efficiency-challenge.github.io/)


## Team Information
- Team name: PAssionGPT
- Team Lead: Yongquan Lai (ranchlai@163.com)
- Team Members: Yongquan Lai, Li Kang, Jing Wang, Kaihe Xu
- Affiliation:  PingAn Group


## Model Description

We use [LLama2-70B-base](https://huggingface.co/meta-llama/Llama-2-70b-hf) for our submission. The model is trained with GPTQ-4bit + LoRA method. 

The quantized model is found in [huggingface](https://huggingface.co/kl1996/llama2-70b-gptq-4bit). We will release the quantize code later.

The model is quantized to 4-bit (at training stage within the 24-h limit) and trained with LoRA method. The model is trained with around 6500 to 8500 steps with weight decay 0.3. The model is trained with 1 A100 GPU in about 15 hours. We use the allowed dataset for training. Details will be released later.

- Dockerfile1: [llama-v12a-6500-7500-merged](https://huggingface.co/kl1996/llama-v12a-6500-7500-merged), using data v12a, merge checkpoint-6500/7000/7500, 
- Dockerfile2: [llama-v13a-6500](https://huggingface.co/kl1996/llama-v13a-6500), using data v13a, checkpoint-6500
- Dockerfile3: [llama-v12-8500](https://huggingface.co/kl1996/llama-v12-8500), using data v12a, checkpoint-8500


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
>>> docker build -f ./Dockerfile$sub -t llama_recipes_inference . # this will be slow since we need to download the quantized 70B model 
>>> docker run --gpus "device=0" -p 8080:80 --rm -ti llama_recipes_inference
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
>>> curl -X POST -H "Content-Type: application/json" -d '{"prompt": "What is the capital of france? Tell me its history"}' http://localhost:8080/chat
>>> {"text":"The capital of France is Paris. Paris is the city with the highest population in France and one of the most populated cities in Europe. It is located in the north of France, in the region of île de France, on the Seine river banks.\n\nThe first inhabitants of Paris are believed to be Celtic tribes, that settled on the bank of the Seine river, around the 3rd century BC. There is no certain date when Paris was founded, but the earliest evidence of its existence dates back to the 1st century BC. In the 3rd century BC, the Romans conquered the region and created a city called Lutetia. The city of Lutetia was renamed to Paris in 265 CE, and became the capital of the Merovingian Kingdom in 508 CE. In the 9th century, Paris became the capital of France.\n\nDuring the French Revolution in 1789, the population of Paris rebelled against the monarchy, and overthrew King Louis XVI. Since then, Paris has been the center of many political and cultural movements. Throughout the 19th and 20th century, Paris was one of the leading centers of art, literature, and music.\n\nToday, Paris is one of the most popular tourist destinations in the world, attracting over 40 million visitors every year. The city is famous for its rich history, art, and culture. Some of the most popular tourist attractions in Paris include the Eiffel Tower, the Louvre museum, the Notre Dame Cathedral, and the Arc de Triomphe"}
```

## Acknowledgement
Thanks the ORGANIZERS for holding this challenge. We have learned a lot from this challenge.
