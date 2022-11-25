import torch
from typing import List, Tuple
import random
import gin


@gin.configurable
class Sampler:
    """
    Abstract class for each sampling method.
    Results of sampler should be stored in a dictionary
    under the 'output' key.
    """

    MASK_TOKEN = "<mask>"

    def __init__(self, model, alphabet, freeze: gin.REQUIRED):
        # initialize sampler config and model
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = self.alphabet.get_batch_converter()
        self.freeze = freeze

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

    def get_mask_indices(self, sequence: str) -> List[int]:
        # add 1 because sequences start with <cls>
        mask_start_idx = sequence.find(self.freeze) + 1
        mask_end_idx = mask_start_idx + len(self.freeze)

        # if freeze sequence not in sequence or not provided
        if mask_start_idx == -1 or not self.freeze:
            # add 1 because sequences start with <cls>
            return [i + 1 for i in range(len(sequence))]

        start_segment = [i for i in range(1, mask_start_idx)]
        end_segment = [i for i in range(mask_end_idx, len(sequence) + 1)]
        return start_segment + end_segment

    @staticmethod
    def _mask_sequence_randomly(sequence: str, k: int) -> str:
        seq_len = len(sequence)
        assert k < seq_len, f"masking variable ({k}) must be smaller than sequence length ({seq_len})"
        indices = random.sample(range(seq_len), k)
        return Sampler._mask_sequence(sequence, indices)
