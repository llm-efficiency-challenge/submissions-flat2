# Submission

4090 track Dockerfile is in: llama_recipes_7b

A100 track Dockerfile is in:
1. llama_recipes 13b_rand
2. llama_recipes 13b



# Proposal

Email: timothylimyonglee@gmail.com

To efficiently finetune a pretrained LLM, there’s 2 areas of potential improvement I want to look into:

1. Data curation [Auto-Selection via Active Learning approaches]
2. Efficient Training Loop for LLM
   
Proposal 1: Active Learning for Pretrained LLM

Proposal 2: Improve training loops of open-source LLM

If time permitting, I also want to look into:

a) Tweaking the attention mechanism (vllm has an interesting optimisation on this) for training
b) Training on lower context length (1024) but inferring on higher context length (2048). I am particularly interested in context length extrapolation via interpolation.


## Data curation

As of any typical DL projects, up to a certain quantity, quality of the data is what contributes to higher performance. I think this is more true for finetuning on-top of a pre-trained LLM model. The findings/theme of the research papers below aligns with this hypothesis:

- https://arxiv.org/pdf/2305.11206.pdf
- https://arxiv.org/pdf/2306.11644.pdf


However, to curate high quality data, researchers typically have to manually select/generate the data given their understanding of what it means for their data to be of high quality and diverse.

My proposal is to take research ideas that has already been used in the vision space (years ago) regarding Active Learning and apply it LLM training for an automated curation/selection of data.  Findings of paper along the lines of my proposal:

- https://arxiv.org/pdf/1703.04977.pdf
- https://arxiv.org/pdf/1506.02142v6.pdf
- https://arxiv.org/pdf/2106.04972.pdf
- https://arxiv.org/pdf/1706.04599.pdf
- https://arxiv.org/pdf/1901.10609.pdf
- https://arxiv.org/pdf/1701.03551.pdf


For my previous work in Computer Vision, I have applied Active Learning approaches to object recognition models, I want to experiment this for LLM, some material that I have posted: https://medium.com/@timothylimyonglee/benefits-of-determining-the-uncertainty-in-deep-neural-networks-3eb95ee0d52c

Another area of exploration is using embeddings. In image, we cluster embeddings to find similar images and then filter them out so that we do not train on the same kind of data. Same can be done for LLM either using its own embeddings or a custom embedding model.

## Efficient Training Loop for LLM


I see a lot of inefficiencies with the current open-source training loops such as those proposed in lit-gpt (lightning team) or llama-recipes (meta team). 

I want to work on improving their code by:

- Making better and more efficient data loaders that are suitable for LLM training. Currently, a lot of batches wasting compute time while producing 0 loss (masked out/padded).
- Another feature I want to add is along the idea of progressive resizing that has been used in the image domain.
- Others as I go, experiments need to be done on whether improvement is expected such as masking input versus non-masking. Currently, alpaca does not mask the input. However, I believe that masking the input makes better sense. I need to run experiment to check this.
- Non-masking of inputs can lead to potentially further optimisation with last ffn layer applying to token after the input.




