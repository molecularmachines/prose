import torch
from typing import List
import random


class Sampler:
    """
    Abstract class for each sampling method.
    Results of sampler should be stored in a dictionary
    under the 'output' key.
    """

    MASK_TOKEN = "<mask>"

    def __init__(self, model, alphabet, config):

        # initialize sampler config and model
        self.model = model
        self.alphabet = alphabet
        self.config = config
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        # load model to correct device
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        # constants
        self.MASK_TOKEN_IDX = self.alphabet.tok_to_idx[Sampler.MASK_TOKEN]

    def _validate_req_fields(self, config: dict, req_fields: List[str]):
        for field in req_fields:
            assert field in config.keys(), f"Field {field} has to exist in config"
        for k, v in config.items():
            setattr(self, k, v)

    def step(self, sequences: List[str]) -> dict:
        return {"output": ""}

    @staticmethod
    def _mask_sequence(sequence: str, indices: List[int]) -> str:
        seq_lst = list(sequence)
        for index in indices:
            seq_lst[index] = Sampler.MASK_TOKEN
        return "".join(seq_lst)

    @staticmethod
    def _mask_sequence_randomly(sequence: str, k: int) -> str:
        seq_len = len(sequence)
        assert k < seq_len, f"masking variable ({k}) must be smaller than sequence length ({seq_len})"
        indices = random.sample([i for i in range(seq_len)], k)
        return Sampler._mask_sequence(sequence, indices)
