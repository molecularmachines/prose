import torch
from tqdm import tqdm
import esm
from einops import repeat, rearrange
import random
from samplers.Sampler import Sampler


class GibbsSampler(Sampler):
    """
    Gibbs Sampler is a MCMC sampling technqiue for generating a sample from a hard to sample distribution by using 
    conditional distributions. 
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

    def propose_new_sequence(self, masked_tokens, pos, temp=1.0):
        with torch.no_grad():
            results = self.model(masked_tokens, repr_layers=[33], return_contacts=False)
        #Sub select logits for only valid characters - ignore eos mask etc. 
        #TODO: do above by downselecting top k and picking from that 
        logits = results['logits'][:,pos,:]
        if temp is not None:
          logits = logits/temp
        dist = torch.distributions.categorical.Categorical(logits=logits)
        new_tokens = dist.sample()
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
        
        if sampling_order == 'next':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens 
          for i in range(1,tokens.shape[1]-1):
              masked_tokens = tokens.clone()
              masked_tokens[:,i] = self.mask_token_id
              new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=i)
              masked_tokens[:,i] = new_sequence_tokens[0]
              tokens = masked_tokens.clone()
              predictions.append(self.untokenize_sequence(tokens))
        elif sampling_order == 'random':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens
          num_tokens = tokens.shape[1] 
          for i in range(k):
              masked_tokens = tokens.clone()
              random_position = random.choice(range(0,num_tokens-1))
              masked_tokens[:,random_position+1] = self.mask_token_id
              new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=random_position+1)
              masked_tokens[:,i] = new_sequence_tokens[0]
              tokens = masked_tokens.clone()
              predictions.append(self.untokenize_sequence(tokens))
        elif sampling_order == 'block':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens
          num_tokens = tokens.shape[1] 
          for i in range(k):
              masked_tokens = tokens.clone()
              random_position = random.choice(range(0,num_tokens-1))
              masked_tokens[:,random_position+1] = self.mask_token_id
              new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=random_position+1)
              masked_tokens[:,i] = new_sequence_tokens[0]
              tokens = masked_tokens.clone()
              predictions.append(self.untokenize_sequence(tokens))
        return {"output": predictions}