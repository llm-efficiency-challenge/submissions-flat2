# neurips_llm_efficiency_challenge

### Evaluation
We have 3 submission folders: dolly_51199iter-Llama-2-13b-hf, flanfull_dolly_500k_iter-Llama-2-7b-hf and flan15kdolly_51199iter-Llama-2-13b-hf. Each folder contains a Dockerfile and the code required to run it.

### Before Submission
* Add model name to line 557 to lit-gpt/lit-gpt/lit_gpt/config.py
* Modify lit-gpt/main.py line 42 to correct model name
* Modify Dockerfile line 19, 21 and 22 for model name
* Check Dockerfile works before submission
