import torch
import torch.nn.functional as F
import random
import gin

from samplers.Sampler import Sampler

@gin.configurable
class NucleusSampler(Sampler):
    """
    Nucleus Sampler
    """
    def __init__(self, model, alphabet, block_size: int = gin.REQUIRED,start_at: int = gin.REQUIRED, temp: float = gin.REQUIRED,top_p:float=gin.REQUIRED,sampling_order:str=gin.REQUIRED):
        super().__init__(model, alphabet)

        self.mask_token_id = self.alphabet.tok_to_idx['<mask>']
        self.block_size = block_size
        self.temp = temp
        self.start_at = start_at
        self.sampling_order = sampling_order

    def __str__(self):
        return f"nucleus-sampler[block_size={self.block_size},temp={self.temp},sampling_order={self.sampling_order}]"

    def untokenize_sequence(self,tokens):
      """
      Removes <cls and <eos> tokens
      """
      untokens = [self.alphabet.all_toks[i.cpu().item()] for i in tokens.squeeze()]
      return "".join(untokens[1:len(untokens)-1])

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
        return new_tokens[:,pos]

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
        sequence = sequences[0] #only going to take the first sequence - so one sample - no batches
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
        for i in range(self.start_at,self.block_size):
            masked_tokens = tokens.clone() #make a copy
            if self.sampling_order == 'random':
                select_position = random.choice(range(0,num_tokens-1)) #select a random position
            if self.sampling_order == 'next':
                select_position = self.start_at+i #select a random position
            masked_tokens[:,select_position+1] = self.mask_token_id
            new_sequence_tokens = self.propose_new_sequence(masked_tokens,pos=select_position+1)
            masked_tokens[:,select_position+1] = new_sequence_tokens
            tokens = masked_tokens.clone()
            predictions.append(self.untokenize_sequence(masked_tokens))
        
        return predictions
