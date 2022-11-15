import torch
import gin

from samplers.Sampler import Sampler


@gin.configurable
class VanillaSampler(Sampler):
    def __init__(self, model, alphabet, k: int = gin.REQUIRED):
        super().__init__(model, alphabet)
        self.k = k

    def __str__(self):
        return f"vanilla-sampler[k={self.k}]"

    def step(self, sequences):
        # data in correct ESM format
        masked_sequences = [
            self._mask_sequence_randomly(seq, self.k) for seq in sequences
        ]
        data = [(str(i + 1), masked_sequences[i]) for i in range(len(sequences))]
        batch_labels, batch_str, batch_tokens = self.batch_converter(data)

        # ESM inference
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

        # retrieve logits from model and sample accordingly
        logits = results["logits"]
        predictions = []

        top_probs = torch.argmax(logits, dim=2)
        for top in top_probs:
            top_indices = [int(p) for p in top]
            top_tokens = [self.alphabet.all_toks[i] for i in top_indices]
            predictions.append("".join(top_tokens[1:-1]))

        return {"output": predictions}
