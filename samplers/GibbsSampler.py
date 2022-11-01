import torch
from tqdm import tqdm
from einops import repeat, rearrange

from samplers.Sampler import Sampler


class GibbsSampler(Sampler):
    """
    Gibbs Sampler is a MCMC sampling technqiue for generating a sample from a hard to sample distribution by using 
    conditional distributions. 

    Algorithm

    1. 


    Inputs


    """

    def __init__(self, model, alphabet, config):
        super().__init__(model, alphabet, config)
        self.current_energy = None
        self.allowed_aa = "ACDEFGHIKLMNPQRSTVWY"
        req_fields = ["k", "temp"]
        self._validate_req_fields(config, req_fields)

    def sequence_tokenizer(self,sequence,max_len=300):
      #Check to be make sure the seed sequence is a sequence and not a list. 
      #Allan takes in multiple sequence - Gibbs/Nuclues/My Metroplis - Accept one 
      #sequence

      if not isinstance(sequence,str):
        raise Exception("Only pass one seed sequence")

      pad_to_length = max_len - len(sequence)
      uppercase_sequence = sequence.upper()

      #Check for only allowed characters in the sequence
      input_chars = {s for s in uppercase_sequence}
      valid_chars = {s for s in self.allowed_aa}

      if not input_chars.issubset(valid_chars):
          raise (Exception("Invalid input character: " + ",".join(input_chars-valid_chars)))

      batch = [(1, uppercase_sequence + "<mask>" * pad_to_length)]
      labels, strs, tokens = self.batch_converter(batch)
      return labels, strs, tokens

    def propose_new_sequence(self, batch_tokens, k=3, temp=1.0):
        return new_tokens, (forward_log_prob, backward_log_prob)

    def step(self, sequence, sampling_order='next'):
        #sampling - 'all' if you want to generate an entirely new sequence
        
        labels,strs,tokens = self.sequence_tokenizer(sequence)

        return {"output": predictions, "trials": trials, "energy": accepted_energies}
