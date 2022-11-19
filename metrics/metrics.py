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
    Calculates the perplexity of seq given a model and alphabet.
    """
    seq = [('1', seq)]
    batch_labels, batch_str, batch_tokens = sampler.batch_converter(seq)
    batch_tokens = batch_tokens.to(sampler.device)
    with torch.no_grad():
            results = sampler.model(batch_tokens, repr_layers=[33], return_contacts=False)
    logits = results["logits"]
    loss = torch.nn.CrossEntropyLoss()
    return torch.exp(loss(logits[0], batch_tokens[0])).item()
    