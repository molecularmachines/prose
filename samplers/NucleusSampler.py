import torch

import sys
sys.path.append("/Users/manu/Documents/prose")
import torch.nn.functional as F
from tqdm import tqdm
import esm
import random
from samplers.Sampler import Sampler


class NucleusSampler(Sampler):
    """
    Nucleus Sampler
    """

    def __init__(self, model, alphabet, config=None):
        super().__init__(model, alphabet, config)
        self.current_energy = None
        self.allowed_aa = "ACDEFGHIKLMNPQRSTVWY"
        self.mask_token = "<mask>"
        self.mask_token_id = self.alphabet.tok_to_idx['<mask>']
        req_fields = []
        self._validate_req_fields(config, req_fields)

    def untokenize_sequence(self,tokens):
      return [self.alphabet.all_toks[i.cpu().item()] for i in tokens.squeeze()]

    def propose_new_sequence(self, masked_tokens, pos, temp=1.0,top_p=0.9):
        with torch.no_grad():
            results = self.model(masked_tokens, repr_layers=[33], return_contacts=False)
        #Sub select logits for only valid characters - ignore eos mask etc. 
        #TODO: do above by downselecting top k and picking from that 
        logits = results['logits']
        if temp is not None:
          logits = logits/temp
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
       
        #Get the cummulative probabilities from the logits  - Apply softmax first logits -> probs
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_indices_after_cumprob = cumulative_probs >= top_p
        
         # Shift the indices to right to keep the first token to be false
        remove_indices_after_cumprob[..., 1:] = remove_indices_after_cumprob[..., :-1].clone()
        remove_indices_after_cumprob[..., 0] = 0

        indices_to_remove = remove_indices_after_cumprob.scatter(dim=-1, index=sorted_indices, src=remove_indices_after_cumprob)

        #smooth logiths - maybe not needed here
        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        logits[indices_to_remove] = -float('Inf')
        #Sample a new set of tokens
        next_token = torch.distributions.Categorical(logits=logits)
        new_tokens = next_token.sample()
        #TODO: Have to figure out a way to penalize picking repeated characters
        return new_tokens

    def step(self, sequence, sampling_order='next',k=5,block_size=2,temperature=1.0,max_length=300):
        """
        Inputs

        sequence(str) - One seed sequence to start from. Has to be a string not a list.
        sampling_oder(str) - Determines the order in which we are making a new sequence
        'next' - Is the next token predictor, pad the sequence upto max length with <mask> tokens and 
        generate
        'random' - randomly select a position to change.
        max_length(int) - This is to allow us to make sequences of variable lengths to a given seed sequence
        """
        predictions = []
        if not isinstance(sequence,str):
          raise Exception("Only pass one seed sequence")

        #sampling - 'all' if you want to generate an entirely new sequence
        batch = [(1,sequence)]
        labels,strs,seed_tokens = self.batch_converter(batch)
        seed_tokens = seed_tokens.to(self.device)

        #get the embedding of the seed sequence
        with torch.no_grad():
            results = self.model(seed_tokens, repr_layers=[33], return_contacts=False)
  
        tokens = seed_tokens.clone() #copy the seed tokens
        num_tokens = tokens.shape[1] 
        for i in range(k):
            masked_tokens = tokens.clone()
            random_position = random.choice(range(0,num_tokens-1))
            masked_tokens[:,random_position+1] = self.mask_token_id
            new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=random_position+1)
            masked_tokens = new_sequence_tokens
            tokens = masked_tokens.clone()
            predictions.append(self.untokenize_sequence(tokens))
        
        return {"output": predictions}

import json
config = json.load(open("/Users/manu/Documents/prose/config.json"))
esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
print(NucleusSampler(model=esm_model,alphabet=alphabet,config=config).step(sampling_order='random',sequence="MKVIF"))