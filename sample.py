import torch
import torch.nn.functional as F
import esm
from typing import List
import random
from functools import reduce
from einops import repeat, rearrange
from tqdm import tqdm


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# load ESM2
print("loading ESM model...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()
print("done loading model")
model = model.to(device)


# constants
MASK_TOKEN = "<mask>"
MASK_TOKEN_IDX = alphabet.tok_to_idx['<mask>']


class Sampler:
    """
    Abstract class for each sampling method.
    Results of sampler should be stored in a dictionary
    under the 'output' key.
    """

    def sample(self, sequences: List[str], k: int = 3, **kwargs) -> dict:
        return {"output": ""}

    @staticmethod
    def _mask_sequence(sequence: str, indices: List[int]) -> str:
        # seq_len = len(sequence)
        # assert reduce(lambda head, index: head & (0 <= index < seq_len), True), f"indices must be non-negative and smaller than sequence size"
        seq_lst = list(sequence)
        for index in indices:
            seq_lst[index] = MASK_TOKEN
        return "".join(seq_lst)

    @staticmethod
    def _mask_sequence_randomly(sequence: str, k: int) -> str:
        seq_len = len(sequence)
        assert k < seq_len, f"masking variable ({k}) must be smaller than sequence length ({seq_len})"
        indices = random.sample([i for i in range(seq_len)], k)
        return Sampler._mask_sequence(sequence, indices)


class VanillaSampler(Sampler):

    def sample(self, sequences, k, **kwargs):
        # data in correct ESM format
        masked_sequences = [self._mask_sequence_randomly(seq, k) for seq in sequences]
        data = [(str(i + 1), masked_sequences[i]) for i in range(len(sequences))]
        batch_labels, batch_str, batch_tokens = batch_converter(data)

        # ESM inference
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        # retrieve logits from model and sample accordingly
        logits = results['logits']
        predictions = []

        top_probs = torch.argmax(logits, dim=2)
        for top in top_probs:
            top_indices = [int(p) for p in top]
            top_tokens = [alphabet.all_toks[i] for i in top_indices]
            predictions.append("".join(top_tokens[1:-1]))

        return {"output": predictions}


class MetropolisHastingsSampler(Sampler):

    @staticmethod
    def compute_sequence_energy(batch_tokens, method='raw', temp=1.0):
        batch_size, sequence_length = batch_tokens.shape[:2]

        num_tokens = len(alphabet.tok_to_idx)
        selector_buffer = repeat(torch.arange(0, num_tokens).to(device), 'l -> b l', b=batch_size)

        total_energy = torch.zeros((batch_size, ), device=device)
        for index in tqdm(range(1, sequence_length)):
            masked_tokens = batch_tokens.clone()
            masked_tokens[:, index] = MASK_TOKEN_IDX
            with torch.no_grad():
                results = model(masked_tokens, repr_layers=[33], return_contacts=False)
            logits = results['logits'][:, index] / temp
            token_selector = (selector_buffer == rearrange(batch_tokens[:, index], 'b -> b ()'))
            total_energy = total_energy + logits[token_selector]

        return total_energy

    @staticmethod
    def propose_new_sequence(batch_tokens, k=3, temp=1.0):
        batch_size, sequence_length = batch_tokens.shape[:2]
        choice = torch.stack([
            torch.randperm(sequence_length)[:k] for _ in range(batch_size)]).to(device)

        num_tokens = len(alphabet.tok_to_idx)
        selector_buffer = rearrange(torch.arange(0, sequence_length).to(device), 's -> () s ()')
        logits_selector = selector_buffer == rearrange(choice, 'b k -> b () k')
        sequence_selector = logits_selector.sum(-1).bool()

        original_tokens = rearrange(batch_tokens[sequence_selector], '(b k) -> b k', b=batch_size, k=k)
        new_tokens = batch_tokens.clone()
        new_tokens[sequence_selector] = MASK_TOKEN_IDX

        with torch.no_grad():
            results = model(new_tokens, repr_layers=[33], return_contacts=False)
        logits = results['logits'] / temp

        # retrieve logits from model and sample accordingly
        logits = (rearrange(logits, 'b s l -> b s () l') *
                  rearrange(logits_selector, 'b s k -> b s k ()')).sum(1)
        logits[..., :len(alphabet.prepend_toks)] = -torch.finfo(logits.dtype).max
        logits[..., (len(alphabet.prepend_toks) + 20):] = -torch.finfo(logits.dtype).max
        
        transition_prob = torch.distributions.Categorical(logits=logits)
        sampled_tokens = transition_prob.sample()
        
        forward_log_prob, backward_log_prob = list(map(transition_prob.log_prob,
                                                      (sampled_tokens, original_tokens)))

        new_tokens[sequence_selector] = rearrange(sampled_tokens, 'b k -> (b k)')

        return new_tokens, (forward_log_prob, backward_log_prob)

    def sample(self, sequences, k=3, current_energy=None, temp=1.0):
        data = [(str(i + 1), sequences[i]) for i in range(len(sequences))]
        _, __, tokens = batch_converter(data)
        tokens = tokens.to(device)

        batch_size, seq_len = tokens.shape[:2]
        
        # compute current sequence_energy
        if current_energy is None:
            current_energy = self.compute_sequence_energy(tokens, temp=temp)

        accepted, trials = torch.zeros((batch_size,), device=device).bool(), torch.zeros((batch_size, ), device=device)
        accepted_tokens = torch.zeros_like(tokens, device=device).long()
        accepted_energies = torch.zeros((batch_size,), device=device).float() 

        while not accepted.any():
            new_tokens, (forward, backward) = self.propose_new_sequence(tokens[~accepted], temp=temp, k=k)

            proposal_energy = self.compute_sequence_energy(new_tokens, temp=temp)
            acceptance_prob = torch.exp(-(proposal_energy - current_energy) + backward.sum(-1) - forward.sum(-1))
            acceptance_prob = torch.clamp(acceptance_prob, max=1)

            u = torch.rand((new_tokens.shape[0], ), device=device)

            accepted_tokens[~accepted] = new_tokens * rearrange(u < acceptance_prob, 'b -> b ()').long() 
            accepted_energies[~accepted] = proposal_energy * (u < acceptance_prob).float()
            trials[~accepted] = trials[~accepted] + (u < acceptance_prob).int()
            accepted[~accepted] = accepted[~accepted] | (u < acceptance_prob)


        predictions = []
        for tokens in accepted_tokens:
            tokens = [alphabet.all_toks[i.cpu().item()] for i in tokens]
            predictions.append("".join(tokens[1:-1]))
       
        print(predictions)

        return {"output": predictions, "trials": trials, "energy": accepted_energies}

ESM_ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

class MMetropolisHastingsSampler(Sampler):
    @staticmethod
    def sequence_tokenizer(sequence,max_len=300):
      if not isinstance(sequence,str):
        raise (Exception("Pass sequence as a string"))

      pad_length = max_len - len(sequence)
      upper_sequence = sequence.upper()

      #Check for only allowed characters in the sequence
      input_chars = {s for s in upper_sequence}
      valid_chars = {s for s in ESM_ALLOWED_AMINO_ACIDS}

      if not input_chars.issubset(valid_chars):
          raise (Exception("Invalid input character: " + ",".join(input_chars-valid_chars)))

      batch = [(1, upper_sequence + "<mask>" * pad_length)]
      labels, strs, tokens = batch_converter(batch)
      return labels, strs, tokens
  
    @staticmethod
    def compute_sequence_energy(tokens,list_of_available_tokens,energy_type='raw'):
        sequence_energy = 0
        sequence_length_with_end_tokens = len(tokens[0])-1
        #skip first and last token
        for idx in range(1,sequence_length_with_end_tokens):
            batch_tokens = tokens.clone()
            batch_tokens[:,idx]=MASK_TOKEN_IDX
           
            with torch.no_grad():
              results = model(batch_tokens)
              logits = results['logits']
              logits_post_softmax = torch.log_softmax(logits,2)
              # print(logits)
              sum_logits = torch.sum(logits,2)
              sum_post_softmax = torch.sum(logits_post_softmax,2)
              # print(sum_logits,"sum")
              if energy_type == 'raw':
                sequence_energy = sequence_energy+sum_logits[:,idx]
              else:
                #defaulting to local normalized
                sequence_energy = sequence_energy+sum_post_softmax[:,idx]
        return sequence_energy

    def sample(self, sequence, epochs=3, current_energy=None):
        labels,strs,tokens = self.sequence_tokenizer(sequence,max_len=len(sequence))
        original_seq_tokens = tokens.clone()
        list_of_available_tokens = alphabet.tok_to_idx
        idx_to_tok = {v: k for k, v in list_of_available_tokens.items()}
        proposal_outputs = []
        for epoch in range(epochs):
          #Selecting sequentially but to move randomly
          # seq_ids = [x for x in range(1,len(tokens[0])-1)]
          # seq_ids.pop(random.randrange(len(seq_ids)))
          tokens = original_seq_tokens.clone()
          for idx in range(1,len(tokens[0])-1):
            old_sequence_tokens = tokens.clone()
            energy_old = self.compute_sequence_energy(tokens,list_of_available_tokens,energy_type='raw')
            #Propose a new sequence - sequentially first
            
            old_token_id = idx_to_tok[int(tokens[:,idx].cpu().numpy())]
            old_sequence_tokens[:,idx] = MASK_TOKEN_IDX
            with torch.no_grad():
              results = model(old_sequence_tokens, repr_layers=[33], return_contacts=False)
              logits = results['logits']
              #Fun stuff -???? 
              #Followed this tack over flow for sampling from the conditional distribution
              mlm_conditional = torch.distributions.Categorical(logits=logits)
              w_o_q = mlm_conditional.log_prob(old_sequence_tokens)[:,idx]
              token_from_mlm_conditional = mlm_conditional.sample()
              w_o_n = mlm_conditional.log_prob(token_from_mlm_conditional)[:,idx]
              # probability
              energy_new = self.compute_sequence_energy(token_from_mlm_conditional,list_of_available_tokens,energy_type='raw')
              acceptance_probability = torch.min(torch.Tensor([1,torch.exp(energy_old-energy_new)*torch.exp(w_o_q-w_o_n)]))
              u = random.uniform(0,1)
              if u<=acceptance_probability:
                tokens = token_from_mlm_conditional
              else:
                tokens = old_sequence_tokens.clone()

            #tokens to amino_acid
            chars = []
            for j in range(1,len(tokens[0])-1):
              selxt_idx = tokens[0][j].cpu()
              
              chars.append(idx_to_tok[int(selxt_idx.detach().cpu().numpy())])

            proposal_outputs.append("".join([x for x in chars]))
            print("".join([x for x in chars]))
            

        #tokens 
        return proposal_outputs


class NucleusSampler(Sampler):
    @staticmethod
    def sequence_tokenizer(sequence,max_len=300):

      if not isinstance(sequence,str):
        raise (Exception("Pass sequence as a string"))

      pad_length = max_len - len(sequence)
      upper_sequence = sequence.upper()

      #Check for only allowed characters in the sequence
      input_chars = {s for s in upper_sequence}
      valid_chars = {s for s in ESM_ALLOWED_AMINO_ACIDS}

      if not input_chars.issubset(valid_chars):
          raise (Exception("Invalid input character: " + ",".join(input_chars-valid_chars)))

      batch = [(1, upper_sequence + "<mask>" * pad_length)]
      labels, strs, tokens = batch_converter(batch)
      return labels, strs, tokens

    def sample(self, sequence,history=True,temperature=1.0,top_p=0.9):
      labels,strs,tokens = self.sequence_tokenizer(sequence,max_len=len(sequence))
      #Get logits for the sequence seed
      results = model(tokens.to(device), repr_layers=[33], return_contacts=False)
      logits = results['logits']
      logits = logits[:, -1, :] / temperature
      sorted_logits, sorted_indices = torch.sort(logits, descending=True)
      cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
      print(cumulative_probs)
      sorted_indices_to_remove = cumulative_probs >= top_p
      # Shift the indices to the right to keep also the first token above the threshold
      sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
      sorted_indices_to_remove[..., 0] = 0
	
      indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
      logits[indices_to_remove] = 0
#I am not sure if categorical normalizes automatically
      logits = logits - logits.logsumexp(dim=-1, keepdim=True)
      logits[indices_to_remove] = -float('Inf')
      next_token = torch.distributions.Categorical(logits=logits)
      print(next_token.sample())
      list_of_available_tokens = alphabet.tok_to_idx
      idx_to_tok = {v: k for k, v in list_of_available_tokens.items()}
      return idx_to_tok[int(next_token.sample().detach().cpu().numpy())]

