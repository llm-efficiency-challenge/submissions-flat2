# NeurIPS Submission #3
=======================
- my mail:1413786269@qq.com
- collaborator:Secbone(https://github.com/Secbone)  mail:secbone@gmail.com


### Instructions for Running

Note: Before running the Docker image, please ensure GPUs are available on your machine.
IMPORTANT: I have tested this on a machine with A100 40GB GPU.

Our model achieved a high score on the Openllm leaderboard, and you are encouraged to use it for reference.

### About Our Submission
We use llama.cpp and modified the code to inference our model.

### Methodology
We adopted the llama-30b-hf as our foundational model and integrated a novel technique to align the model more closely with user intents. The training duration was 23 hours, at the end of which we observed commendable results.

Our model boasts top scores in the arc/Hellaswag/MMLU categories. However, it's worth noting that our scores were relatively lower in the TruthfulQA test.