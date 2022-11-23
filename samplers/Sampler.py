import torch
from typing import List, Tuple
import random


class Sampler:
    """
    Abstract class for each sampling method.
    Results of sampler should be stored in a dictionary
    under the 'output' key.
    """

    MASK_TOKEN = "<mask>"

    def __init__(self, model, alphabet):
        # initialize sampler config and model
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

        # load model to correct device
        self.device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)

        # constants
        self.MASK_TOKEN_IDX = self.alphabet.tok_to_idx[Sampler.MASK_TOKEN]


    def step(self, sequences: List[str]) -> Tuple[str, dict]:
        return "", dict()

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
        indices = random.sample(range(seq_len), k)
        return Sampler._mask_sequence(sequence, indices)
