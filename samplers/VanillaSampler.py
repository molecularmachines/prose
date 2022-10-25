import torch

from samplers.Sampler import Sampler


class VanillaSampler(Sampler):

    def __init__(self, model, alphabet, config):
        super().__init__(model, alphabet, config)
        req_fields = ["k"]
        self._validate_req_fields(config, req_fields)

    def step(self, sequences):
        # data in correct ESM format
        masked_sequences = [self._mask_sequence_randomly(seq, self.k) for seq in sequences]
        data = [(str(i + 1), masked_sequences[i]) for i in range(len(sequences))]
        batch_labels, batch_str, batch_tokens = self.batch_converter(data)

        # ESM inference
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)

        # retrieve logits from model and sample accordingly
        logits = results['logits']
        predictions = []

        top_probs = torch.argmax(logits, dim=2)
        for top in top_probs:
            top_indices = [int(p) for p in top]
            top_tokens = [self.alphabet.all_toks[i] for i in top_indices]
            predictions.append("".join(top_tokens[1:-1]))

        return {"output": predictions}
