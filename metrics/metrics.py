import json
import os
from absl import app
from absl import flags
from absl import logging
from typing import Dict

FLAGS = flags.FLAGS
flags.DEFINE_string("results_dir", None, "Path to results directory")
flags.mark_flag_as_required("results_dir")

RESULTS_FILE = "results.json"
METRICS_FILE = "metrics.json"


def hamming_distances(results) -> Dict:
    """
    Calculates the hamming distance of resulted sequences.
    Result file contains the original sequences alongside the
    resulted sequences per sampler. Each sampler has a list of
    sequences for each step.
    """

    def _calc_hamming(a, b) -> float:
        return sum([0 if x == b[i] else 1 for i, x in enumerate(a)]) / len(a)

    distances = dict()
    original_sequences = results.get("original_sequences")

    # for each sampler in results JSON calculate all sequences hamming
    for s in results.get("samplers").keys():
        sampler = results.get("samplers").get(s)
        sampler_seqs = sampler.get("res_sequences")  # [n_steps, n_seqs]
        sampler_distances = []

        # sampler sequences are organized per sample step
        for step_sequences in sampler_seqs:
            step_distances = []

            # calculate hamming of each sequence in the step against original
            for i in range(len(step_sequences)):
                d = _calc_hamming(original_sequences[i], step_sequences[i])
                step_distances.append(d)
            sampler_distances.append(step_distances)
        distances[s] = sampler_distances

    return distances


def main(argv):
    del argv
    # read results JSON file
    results_path = os.path.join(FLAGS.results_dir, RESULTS_FILE)
    results = json.load(open(results_path))

    # calculate all metrics
    metrics = dict()
    metrics["hamming"] = hamming_distances(results)

    # save metrics to metrics JSON file
    logging.info(f"Logging metrics file to {METRICS_FILE}")
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    app.run(main)
