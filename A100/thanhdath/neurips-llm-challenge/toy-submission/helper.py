from typing import Optional, Tuple, List

import torch

@torch.no_grad()
def response_generation(model, tokenizer, prompt, max_new_tokens=256, temperature=0.2, top_k=40):
    inputs = tokenizer(prompt, return_tensors="pt")
    for k in inputs:
        inputs[k] = inputs[k].to(model.device)

    generation_output = model.generate(
        **inputs,
        temperature=temperature,
#         top_p=0.75,
        top_k=top_k,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=max_new_tokens,
    )
        
    input_length = inputs['input_ids'].shape[1]
    output_tokens = generation_output.sequences[0][input_length:]
    log_prob = [generation_output.scores[0][0, i].item() for i in output_tokens]
    top_log_prob = [(i.item(), prob) for i, prob in zip(output_tokens, log_prob)]

    return output_tokens, log_prob, top_log_prob


@torch.no_grad()
def toysubmission_generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_returned_tokens: int,
    max_seq_length: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[int], List[float], List[Tuple[int, float]]]:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        idx: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        max_seq_length: The maximum sequence length allowed. Should be less or equal than the block size.
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.

    Returns:
        Tuple containing a list of token indexes, id of the top log probability, and the actual log probability of the
        selected token.
    """
    T = idx.size(0)
    assert max_returned_tokens > T
    device, dtype = idx.device, idx.dtype
    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(max_returned_tokens, dtype=dtype, device=device)
    empty[:T] = idx
    idx = empty
    input_pos = torch.arange(0, T, device=device)

    top_logprob = []
    logprob = []

    # generate up to a fixed number of tokens
    for _ in range(max_returned_tokens - T):
        x = idx.index_select(0, input_pos).view(1, -1)

        # forward
        logits = model(x, max_seq_length, input_pos)
        logits = logits[0, -1] / temperature

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < v[[-1]], -float("Inf"), logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)

        idx_next = torch.multinomial(probs, num_samples=1).to(dtype=dtype)

        # append the logprob of selected token
        logprob.append(torch.log(probs[idx_next]).item())

        # append th idx and logprob of top token
        top_logprob.append((torch.argmax(probs).item(), torch.log(probs).max().item()))

        # advance
        input_pos = input_pos[-1:] + 1

        # concatenate the new generation
        idx = idx.index_copy(0, input_pos, idx_next)

        # if <eos> token is triggered, return the output (stop generation)
        if idx_next == eos_id:
            return idx[:input_pos], logprob, top_logprob  # include the EOS token

    return idx, logprob, top_logprob
