import os
from typing import Dict


def hamming_distance(seqa: str, seqb: str) -> Dict:
    """
    Calculates the hamming distance between seqa and seqb.
    """
    assert len(seqa) == len(seqb)
    return sum([0 if x == seqb[i] else 1 for i, x in enumerate(seqa)]) / len(seqa)

