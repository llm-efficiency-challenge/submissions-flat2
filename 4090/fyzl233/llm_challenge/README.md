# Submission for neurips_llm_efficiency_challenge

Our submission is based on https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge/tree/master/sample-submissions/llama_recipes .

# Introduction about our approach

Foundation model: Qwen-14B

PEFT: Qlora

Dataset: Databricks-Dolly-15 

# To DO
1. try more datasets: OASST1(form Qlora) + Flan_v2_subsample(from https://github.com/allenai/open-instruct/blob/main/scripts/prepare_train_data.sh)
2. In order to improve the Representation (race) and Stereotypes (race) indicators, we plan to refer to https://ar5iv.labs.arxiv.org/html/2211.09110#A3.SS5. Modify the dataset, balance the distribution of various races in the dataset, and fine tune again
3. More adjustments to hyperparameters

# Email
2019202165@ruc.edu.cn
