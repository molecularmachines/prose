import torch
import torch.nn.functional as F
import random
import gin

from samplers.Sampler import Sampler

@gin.configurable
class MetropolisSampler(Sampler):

    def __init__(self, model, alphabet, block_size: int = gin.REQUIRED,start_at: int = gin.REQUIRED, temp: float = gin.REQUIRED,energy_type:str=gin.REQUIRED,sampling_order:str=gin.REQUIRED):
        super().__init__(model, alphabet)

        self.mask_token_id = self.alphabet.tok_to_idx['<mask>']
        self.block_size = block_size
        self.energy_type = energy_type
        self.temp = temp
        self.start_at = start_at
        self.sampling_order = sampling_order
        
    def compute_sequence_energy(self,tokens):
        sequence_energy = 0
        sequence_length_with_end_tokens = len(tokens[0])-1
        #skip first and last token
        for idx in range(1,sequence_length_with_end_tokens):
            batch_tokens = tokens.clone()
            batch_tokens[:,idx]=self.mask_token_id
           
            with torch.no_grad():
              results = self.model(batch_tokens)
              logits = results['logits']
              logits_post_softmax = torch.log_softmax(logits,2)
              # print(logits)
              sum_logits = torch.sum(logits,2)
              sum_post_softmax = torch.sum(logits_post_softmax,2)
              # print(sum_logits,"sum")
              if self.energy_type:
                sequence_energy = sequence_energy+sum_logits[:,idx]
              else:
                #defaulting to local normalized
                sequence_energy = sequence_energy+sum_post_softmax[:,idx]
        return sequence_energy

    def untokenize_sequence(self,tokens):
      """
      Removes <cls and <eos> tokens
      """
      untokens = [self.alphabet.all_toks[i.cpu().item()] for i in tokens.squeeze()]
      return "".join(untokens[1:len(untokens)-1])

    def propose_new_sequence(self,masked_tokens,pre_masked,pos,energy_old):
        #Propose a new sequence - sequentially first

        with torch.no_grad():
            results = self.model(masked_tokens, repr_layers=[33], return_contacts=False)
        
        logits = results['logits']

        allowed_aa = [self.alphabet.tok_to_idx[k] for k in 'ACDEFGHIKLMNPQRSTVY']
        disallowed_aa = [i for i in range(33) if i not in allowed_aa]
        # logits[:,:,disallowed_aa]=-float('Inf')
        
        mlm_conditional = torch.distributions.Categorical(logits=logits)
        w_o_q = mlm_conditional.log_prob(masked_tokens)[:,pos]
        token_from_mlm_conditional = mlm_conditional.sample()
        w_o_n = mlm_conditional.log_prob(token_from_mlm_conditional)[:,pos]
        energy_new = self.compute_sequence_energy(token_from_mlm_conditional)
        acceptance_probability = torch.min(torch.Tensor([1,torch.exp(energy_old-energy_new)*torch.exp(w_o_q-w_o_n)]))
        
        u = random.uniform(0,1)
        if u<=acceptance_probability:
            tokens = token_from_mlm_conditional
        else:
            tokens = pre_masked
        
        return tokens
        
    def step(self,sequences):

        predictions = []
        sequence = sequences[0]
        if not isinstance(sequence,str):
          raise Exception("Only pass one seed sequence")

        batch = [(1,sequence)]
        labels,strs,seed_tokens = self.batch_converter(batch)
        seed_tokens = seed_tokens.to(self.device)

        #get the embedding of the seed sequence
        with torch.no_grad():
            results = self.model(seed_tokens, repr_layers=[33], return_contacts=False)

        if self.sampling_order == 'random':
          #Next token sampler - First token is a CLS token so will ignore that one
          tokens = seed_tokens.clone() #copy the seed tokens
          num_tokens = tokens.shape[1] 
          for i in range(0,self.block_size):
            masked_tokens = tokens.clone()
            e_o = self.compute_sequence_energy(masked_tokens)
            random_position = random.choice(range(0,num_tokens-1))
            masked_tokens[:,random_position+1] = self.mask_token_id
            new_sequence_tokens = self.propose_new_sequence(masked_tokens,tokens,random_position+1,e_o)
            masked_tokens = new_sequence_tokens.clone()
            tokens = masked_tokens.clone()
            predictions.append(self.untokenize_sequence(masked_tokens))
#       
        print(predictions)
        return predictions, {}








   
