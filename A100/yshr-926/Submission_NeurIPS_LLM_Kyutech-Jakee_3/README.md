# Our Submission
This repository is prepared for the [NeurIPS Large Language Model Efficiency Challenge:
1 LLM + 1GPU + 1Day](https://llm-efficiency-challenge.github.io/index). We have created this repository based on the [neurips_llm_efficiency_challenge](https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge).

## Submissions
We submitted following three settings for A100 (student) track. 
* Dockerfile.1
* Dockerfile.2
* Dockerfile.3

## Ours
### Team name
Kyutech Jakee

### Team info
Our team is composed to students of Kyushu Instutitute of Technology (Kyutech).
* Hiroto Yoshihara 

    email : yoshihara.hiroto873@mail.kyutech.jp

* Mizuki Sakaguchi

    email : sakaguchi.mizuki739@mail.kyutech.jp

## Getting Started
Make sure you have recursively cloned the top this repository in order to get `NeurIPS_LLM_Kyutech-Jakee`.

❗ Make sure the repo is cloned with git submodule support either:

```sh
git clone --recurse-submodules ...
```

or if you cloned the repo but are missing the `NeurIPS_LLM_Kyutech-Jakee` folder

```sh
git submodule update --init --recursive
```

## Structure
* NeurIPS_LLM_Kyutech-Jakee/ 
    * `NeurIPS_LLM_Kyutech-Jakee` is a repository of [lit-gpt](https://github.com/Lightning-AI/lit-gpt) modified for us.
* main.py
    * The process/ and tokenize/ endpoints are defined here
* main_finetune.py
    * The process/ and tokenize/ endpoints are defined here for finetune
* helper.py
    * Applies logic on top of lit-gpt's generate in order to produce responses in accordance with the spec.
* api.py
    * Defines the pydantic classes for the FASTapi server
* download_ftmodel.py
    * Downloads the lora-finetuned parameters from HuggingFace.
* Dockerfiles
    * Definition of the image that will set-up the server used for submissions
* Dockerfile.finetune
    * Definition of the image that will finetune the Llama-2-13b in our settings and set-up the server used for submissions

## Build and run 
```sh
docker build -f Dockerfile -t Kyutech-Jakee_submission .
docker run --gpus all -p 8080:80 Kyutech-Jakee_submission
```
## Send requests
```sh
curl -X POST -H "Content-Type: application/json" -d '{"prompt": "The capital of france is "}' http://localhost:8080/process
```
