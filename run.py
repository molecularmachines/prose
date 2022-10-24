import os
import json
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm

from sample import VanillaSampler, MetropolisHastingsSampler
from biotite.sequence.io import fasta
from omegafold.__main__ import main as fold
from omegafold.__main__ import model

# samplers
samplers = {
    "vanilla": VanillaSampler(),
    "metropolis": MetropolisHastingsSampler()
}
sampler_names = list(samplers.keys())

# experiment program run arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("input", None, "Path to fasta sequences file")
flags.DEFINE_string("output", "out", "Path to output directory")
flags.DEFINE_integer("n", 10,
                     "Number of steps to run the sampler")
flags.DEFINE_float("temp", 1.0,
                    "Temperature of sampler")
flags.DEFINE_integer("k", 1,
                     "Number of sampling residues per step")
flags.DEFINE_integer('fold_every', 10,
                     'Number of steps between fold metric collection')
flags.DEFINE_list("method", None,
                  "Sampling method. Must be of: {sampler_names}")

# required arguments
flags.mark_flag_as_required("input")
flags.mark_flag_as_required("output")

# constants
RESULTS_FILE = "results.json"


def main(argv):
    del argv  # Unused.

    # load omegafold to memory
    omegafold_model = model()

    # choose sampling methods
    user_samplers = sampler_names if FLAGS.method is None else FLAGS.method

    # prepare output directory
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    # read fasta file
    fasta_file = fasta.FastaFile.read(FLAGS.input)
    fasta_sequences = fasta.get_sequences(fasta_file)
    sequences = list(fasta_sequences.values())
    sequences = [str(s) for s in sequences]
    names = list(fasta_sequences.keys())

    # run all samplers
    res_json = {"original_sequences": sequences}
    for s in user_samplers:
        logging.info(f"sampling with method: {s}")
        sampler = samplers[s]
        res_sequences = []
        confidences = []
        current_energy = None

        # sample n times from each sampler
        for i in tqdm(range(FLAGS.n)):
            # perform sampler forward
            sampled_seqs = sampler.sample(sequences, k=FLAGS.k, current_energy=current_energy, temp=FLAGS.temp)
            output_sequences = sampled_seqs.get('output')
            current_energy = sampled_seqs.get('energy')
            res_sequences += output_sequences

            if (i % FLAGS.fold_every == 0):
                # construct fasta file for folding
                step_fasta_file_name = f"{s}_{i+1}.fasta"
                step_fasta_file_path = os.path.join(FLAGS.output, step_fasta_file_name)
                step_fasta_file = fasta.FastaFile()
                res_names = [f"{name}|{s}|STEP {i+1}" for name in names]
                for j, res_name in enumerate(res_names):
                    step_fasta_file[res_name] = res_sequences[j]
                step_fasta_file.write(step_fasta_file_path)

                # fold fasta with OmegaFold
                logging.info(f"Folding step {i+1}/{FLAGS.n}")
                fold_out = fold(omegafold_model, step_fasta_file_path, FLAGS.output)
                confidence = [x['confidence_overall'] for x in fold_out]
                confidences.append(confidence)

            # next step in trajectory from current step
            sequences = output_sequences

        # construct result for sampler
        logging.info(f"logging results for sampler: {s}")
        sampler_json = {"res_sequences": res_sequences,
                        "confidences": confidences}
        res_json[s] = sampler_json

    # write results to file
    OUTPUT_FILE = os.path.join(FLAGS.output, RESULTS_FILE)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(res_json, f)
    logging.info(f'results have been written to {OUTPUT_FILE}')


if __name__ == '__main__':
    app.run(main)
