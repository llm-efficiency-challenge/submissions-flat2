# neurips_llm_efficiency_challenge
Submission to NeurIPS 2023 LLM Efficiency Challenge

## Repo Update
[2023/10/31] Before submission deadline, we suspected that one of the fine-tuning tasks might be at risk of data leakage. To investigate this, we temporarily adjusted the routing rules to avoid using the fine-tuned model for TruthfulQA evaluation. After examination, it became clear that the training data was valid (no data leakage) and the results were reproducible. After reporting to the organizer (@marksaroufim), we revert the relevant [one-line-of-code commit](https://github.com/MrJungle1/neurips_llm_efficiency_challenge/commit/c60fed1a683039c31f88b7a91ce59bc2d2727a4e) so the originally-trained default LoRA adapter is active for evaluation.

## Team Details
- Team Name: Kaiwu-SIAT

- Team Members
  - **Lvfang Tao**, Researcher at Tencent
  - **Linhang Cai**, Researcher at Tencent
  - **Jiang Chen**, Master's Student at Southern University of Science and Technology & Shenzhen Institute of Advanced Technology
  - **Cong Ma**, Master's Student at Shenzhen Institute of Advanced Technology
  - **Wenxi Zhu** (*Team Lead*), Engineering Manager at Tencent

- Mailing List
  - luistao [at] tencent.com
  - linhangcai [at] tencent.com
  - 1505396805 [at] qq.com
  - 2268845318 [at] qq.com
  - wenxizhu [at] tencent.com


## Evaluation
Please build **all** the `Dockerfile.inference` files in `submission_A100_*` and `submission_4090_*` and run evaluation with the images built.

**IMPORTANT NOTE:** The bias scores of the CNN-DailyMail abstraction task can be easily hacked by appending tokens of opposite gender-specific pronouns 
and racial-related surnames to the output generated by model to balance the distribution. 
The bias scores can be lowered to **near-zero**, while the ROGUE-2 score is **almost the same** as orginal. **See the image below for results.**

![企业微信截图_16983215704325](https://github.com/MrJungle1/neurips_llm_efficiency_challenge/assets/16898790/e8992088-621f-4b21-85ae-cf8d7c048d47)


We found this trick somewhat suspicous and may be considered as a form of cheating, so we didn't include this in the submission.

**If some other teams obtained better CNN-DailyMail bias scores than our actual submission by using similar tricks, and their results considered valid, we may (at least) share the same bias scores with them.**

> Note: We are busy testing all the dockerfiles for training, so they may not be available immediately.
>
> Please ping us if some given solutions can be prompted to the reproduction stage,
> so that we can update the repository to include all the necessary materials and directions.

If any problems encountered during the evaluation / reproduction stage, feel free to contact us via **email** or **GitHub issues**.

Thanks for organizing the amazing event!

## Dataset
The datasets used to finetune the base model can be found [**HERE**](https://huggingface.co/datasets/jiangchensiat/kaiwu-siat).

Note that all the `test.json` files are used as placeholders and are not related to the corpus we trained on. We didn't monitor test loss during training.