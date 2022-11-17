import torch
import torch.nn.functional as F
import random
import gin

from samplers.Sampler import Sampler

@gin.configurable
class GibbsSampler(Sampler):
    """
    Gibbs Sampler is a MCMC sampling technqiue for generating a sample from a hard to sample distribution by using 
    conditional distributions. 
    """
    def __init__(self, model, alphabet, block_size: int = gin.REQUIRED,start_at: int = gin.REQUIRED, temp: float = gin.REQUIRED,sampling_order:str=gin.REQUIRED):
        super().__init__(model, alphabet)

        self.mask_token_id = self.alphabet.tok_to_idx['<mask>']
        self.block_size = block_size
        self.temp = temp
        self.start_at = start_at
        self.sampling_order = sampling_order

    def untokenize_sequence(self,tokens):
      """
      Removes <cls and <eos> tokens
      """
      
      string_tokens = ""
      for i in tokens.squeeze():
        print(i.cpu().item())
        print(self.alphabet.all_toks[i.cpu().item()])
        string_tokens = string_tokens+self.alphabet.all_toks[i.cpu().item()]

      untokens = [self.alphabet.all_toks[i.cpu().item()] for i in tokens.squeeze()]
      return "".join(untokens[1:len(untokens)-1])

    def propose_new_sequence(self, masked_tokens, pos, temp=1.0):
        with torch.no_grad():
            results = self.model(masked_tokens, repr_layers=[33], return_contacts=False)
        #Sub select logits for only valid characters - ignore eos mask etc. 
        #TODO: do above by downselecting top k and picking from that 
        logits = results['logits'][:,pos,:]
        if temp is not None:
          logits = logits/temp
        dist = torch.distributions.categorical.Categorical(logits=logits)
        #accept the next best logit if the logit being predicted is actually not a allowed_aa
        new_tokens = dist.sample()
        print(self.alphabet.tok_to_idx)
        allowed_aa = [self.alphabet.tok_to_idx[k] for k in 'ACDEFGHIKLMNPQRSTVWY']
        # while new_tokens not in allowed_aa:
        #   new_tokens = dist.sample()
        #TODO: Have to figure out a way to penalize picking repeated characters
        return new_tokens

    def step(self, sequences):
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
        sequence = sequences[0]
        if not isinstance(sequence,str):
          raise Exception("Only pass one seed sequence")

        #sampling - 'all' if you want to generate an entirely new sequence
        batch = [(1,sequence)]
        labels,strs,seed_tokens = self.batch_converter(batch)
        seed_tokens = seed_tokens.to(self.device)
        
        #get the embedding of the seed sequence
        with torch.no_grad():
            results = self.model(seed_tokens, repr_layers=[33], return_contacts=False)
        
        if self.sampling_order == 'next':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens 
          for i in range(1,tokens.shape[1]-1):
              masked_tokens = tokens.clone()
              masked_tokens[:,i] = self.mask_token_id
              new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=i)
              masked_tokens[:,i] = new_sequence_tokens[0]
              tokens = masked_tokens.clone()
              predictions.append(self.untokenize_sequence(tokens))

        if self.sampling_order == 'random':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens
          num_tokens = tokens.shape[1] 
          for i in range(0,self.block_size):
              masked_tokens = tokens.clone()
              random_position = random.choice(range(0,num_tokens-1))

              masked_tokens[:,random_position+1] = self.mask_token_id
              new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=random_position+1)
              
              masked_tokens[:,random_position+i] = new_sequence_tokens[0]
              tokens = masked_tokens.clone()
              predictions.append(self.untokenize_sequence(tokens))

        elif self.sampling_order == 'block':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens
          num_tokens = tokens.shape[1] 
          for i in range(self.start_at,self.block_size):
              masked_tokens = tokens.clone() #make a copy
              select_position = self.start_at+i
              masked_tokens[:,select_position+1] = self.mask_token_id
              new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=select_position+1)
             
              masked_tokens[:,select_position+1] = new_sequence_tokens[0]
              tokens = masked_tokens.clone()
              predictions.append(self.untokenize_sequence(tokens))

        return predictions, {}
