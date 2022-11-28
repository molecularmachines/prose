import torch
from samplers.Sampler import Sampler


def hamming_distance(seqa: str, seqb: str) -> float:
    """
    Calculates the hamming distance between seqa and seqb.
    """
    assert len(seqa) == len(seqb)
    return sum([0 if x == seqb[i] else 1 for i, x in enumerate(seqa)]) / len(seqa)


def perplexity(seq: str, sampler: Sampler) -> float:
    """
    Calculates the perplexity of seq given a Sampler. 
    
    Matches the definition of Pseudo-Perplexity in ESM (compute_pppl function linked below): 
    https://github.com/facebookresearch/esm/blob/main/examples/variant-prediction/predict.py
    """
    data = [('0', seq)]

    _, _, batch_tokens = sampler.batch_converter(data)
    batch_tokens = batch_tokens.to(sampler.device)
    ppl = torch.tensor(0.0).to(sampler.device)
    for i in range(1, len(seq)-1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = sampler.MASK_TOKEN_IDX
        with torch.no_grad():
            log_probs = torch.log_softmax(sampler.model(batch_tokens_masked, repr_layers=[33],
                                                        return_contacts=False)["logits"], dim=-1)
        ppl += log_probs[0, i, batch_tokens[0, i]]

    return torch.exp(-ppl / len(seq))  
