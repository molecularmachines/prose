import gin
import torch
import random
from tqdm import tqdm
from einops import repeat, rearrange

from samplers.Sampler import Sampler


@gin.configurable
class MetropolisHastingsSampler(Sampler):
    def __init__(
        self, model, alphabet, k: int = gin.REQUIRED, temp: float = gin.REQUIRED
    ):
        super().__init__(model, alphabet)
        self.current_energy = None
        self.k = k
        self.temp = temp

    def __str__(self):
        return f"metropolis-hastings-sampler[k={self.k},temp={self.temp}]"

    def compute_sequence_energy(self, batch_tokens, method="raw", temp=1.0):
        batch_size, sequence_length = batch_tokens.shape[:2]

        num_tokens = len(self.alphabet.tok_to_idx)
        selector_buffer = repeat(
            torch.arange(0, num_tokens).to(self.device), "l -> b l", b=batch_size
        )

        total_energy = torch.zeros((batch_size,), device=self.device)
        for index in tqdm(range(1, sequence_length)):
            masked_tokens = batch_tokens.clone()
            masked_tokens[:, index] = self.MASK_TOKEN_IDX
            with torch.no_grad():
                results = self.model(
                    masked_tokens, repr_layers=[33], return_contacts=False
                )
            logits = results["logits"][:, index] / temp
            token_selector = selector_buffer == rearrange(
                batch_tokens[:, index], "b -> b ()"
            )
            total_energy = total_energy + logits[token_selector]

        return total_energy

    def propose_new_sequence(self, batch_tokens, mask_indices, k=3, temp=1.0):
        batch_size, sequence_length = batch_tokens.shape[:2]
        choices = [torch.tensor(random.sample(mask_indices, k=len(mask_indices))[:k]) for _ in range(batch_size)]
        choice = torch.stack(choices).to(self.device)

        selector_buffer = rearrange(
            torch.arange(0, sequence_length).to(self.device), "s -> () s ()"
        )
        logits_selector = selector_buffer == rearrange(choice, "b k -> b () k")
        sequence_selector = logits_selector.sum(-1).bool()

        original_tokens = rearrange(
            batch_tokens[sequence_selector], "(b k) -> b k", b=batch_size, k=k
        )
        new_tokens = batch_tokens.clone()
        new_tokens[sequence_selector] = self.MASK_TOKEN_IDX

        with torch.no_grad():
            results = self.model(new_tokens, repr_layers=[33], return_contacts=False)
        logits = results["logits"] / temp

        # retrieve logits from model and sample accordingly
        logits = (
            rearrange(logits, "b s l -> b s () l")
            * rearrange(logits_selector, "b s k -> b s k ()")
        ).sum(1)
        logits[..., : len(self.alphabet.prepend_toks)] = -torch.finfo(logits.dtype).max
        logits[..., (len(self.alphabet.prepend_toks) + 20) :] = -torch.finfo(
            logits.dtype
        ).max

        transition_prob = torch.distributions.Categorical(logits=logits)
        sampled_tokens = transition_prob.sample()

        forward_log_prob, backward_log_prob = list(
            map(transition_prob.log_prob, (sampled_tokens, original_tokens))
        )

        new_tokens[sequence_selector] = rearrange(sampled_tokens, "b k -> (b k)")

        return new_tokens, (forward_log_prob, backward_log_prob)

    def step(self, sequences):
        current_energy = self.current_energy
        k = self.k
        temp = self.temp
        data = [(str(i + 1), sequences[i]) for i in range(len(sequences))]
        _, __, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        batch_size, seq_len = tokens.shape[:2]
        mask_indices = self.get_mask_indices(sequences[0])
        # compute current sequence_energy
        if current_energy is None:
            current_energy = self.compute_sequence_energy(tokens, temp=temp)

        accepted, trials = torch.zeros(
            (batch_size,), device=self.device
        ).bool(), torch.zeros((batch_size,), device=self.device)
        accepted_tokens = torch.zeros_like(tokens, device=self.device).long()
        accepted_energies = torch.zeros((batch_size,), device=self.device).float()

        while not accepted.any():
            new_tokens, (forward, backward) = self.propose_new_sequence(
                tokens[~accepted], mask_indices, temp=temp, k=k
            )

            proposal_energy = self.compute_sequence_energy(new_tokens, temp=temp)
            acceptance_prob = torch.exp(
                -(proposal_energy - current_energy) + backward.sum(-1) - forward.sum(-1)
            )
            acceptance_prob = torch.clamp(acceptance_prob, max=1)

            u = torch.rand((new_tokens.shape[0],), device=self.device)

            accepted_tokens[~accepted] = (
                new_tokens * rearrange(u < acceptance_prob, "b -> b ()").long()
            )
            accepted_energies[~accepted] = (
                proposal_energy * (u < acceptance_prob).float()
            )
            trials[~accepted] = trials[~accepted] + (u < acceptance_prob).int()
            accepted[~accepted] = accepted[~accepted] | (u < acceptance_prob)

        predictions = []
        for tokens in accepted_tokens:
            tokens = [self.alphabet.all_toks[i.cpu().item()] for i in tokens]
            predictions.append("".join(tokens[1:-1]))

        print(predictions)
        self.current_energy = accepted_energies

        return predictions, {"trials": trials, "energy": accepted_energies}
