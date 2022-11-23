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
    data = [('0', seq)]+[(str(i + 1), sampler._mask_sequence(seq,[i])) for i in range(len(seq))]
    batch_labels, batch_str, batch_tokens = sampler.batch_converter(data)
    batch_tokens = batch_tokens.to(sampler.device)
    with torch.no_grad():
            results = sampler.model(batch_tokens, repr_layers=[33], return_contacts=False)
    logits = results["logits"]
    ppl = torch.tensor(0.0).to(sampler.device)
    for i in range(len(seq)):
        ppl += logits[i+1,i+1,batch_tokens[0,i+1]]
    return ppl / len(seq)
    