import os
import json
import esm
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from time import gmtime, strftime

from samplers import VanillaSampler, MetropolisHastingsSampler
from biotite.sequence.io import fasta
from omegafold.__main__ import main as fold
from omegafold.__main__ import model


# constants
RESULTS_FILE = "results.json"
CONFIG_FILE = "config.json"
N_STEPS_DEFAULT = 100
K_DEFAULT = 1
FOLD_EVERY_DEFAULT = 5

# experiment program run arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("input", None, "Path to fasta sequences file")
flags.DEFINE_string("output", "out", "Path to output directory")
flags.DEFINE_string("config", CONFIG_FILE, "Path to config json file")

# required arguments
flags.mark_flag_as_required("input")

samplers = {
    "vanilla": VanillaSampler,
    "metropolis": MetropolisHastingsSampler
}


def init_config(config_file: str) -> dict:
    # read config file
    logging.info(f"reading config file: {config_file}")
    config = json.load(open(config_file))

    # validate required fields
    req_fields = ["n_steps", "samplers"]
    for field in req_fields:
        assert field in config.keys(), f"Config file must contain {field}"

    # store defaults to config
    t = strftime('%Y-%m-%d-%H-%M-%S', gmtime())
    config["name"] = config.get("name", t)
    config["date"] = t
    config["fold_every"] = config.get("fold_every", FOLD_EVERY_DEFAULT)
    config["n_steps"] = config.get("n_steps", N_STEPS_DEFAULT)
    config["k"] = config.get("k", K_DEFAULT)

    return config


def main(argv):
    del argv  # Unused.

    # load ESM to memory
    logging.info("loading ESM2")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    logging.info("done loading")

    # initialize config from file
    config = init_config(FLAGS.config)

    # load omegafold to memory
    omegafold_model = model()

    # choose sampling methods
    user_sampler_names = list(config.get("samplers").keys())
    user_samplers = []
    for sampler in user_sampler_names:
        sampler_config = config["samplers"][sampler]
        s = samplers[sampler](esm_model, alphabet, sampler_config)
        user_samplers.append(s)

    # prepare output directory
    if not os.path.exists(FLAGS.output):
        os.makedirs(FLAGS.output)

    # read input fasta file
    fasta_file = fasta.FastaFile.read(FLAGS.input)
    fasta_sequences = fasta.get_sequences(fasta_file)
    sequences = list(fasta_sequences.values())
    sequences = [str(s) for s in sequences]
    names = list(fasta_sequences.keys())

    # run all samplers
    res_json = {"original_sequences": sequences}
    for idx, sampler in enumerate(user_samplers):
        sampler_name = user_sampler_names[idx]
        logging.info(f"sampling with method: {sampler_name}")
        res_sequences = []
        confidences = []

        # sample n times from each sampler
        n_steps = config.get("n_steps")
        fold_every = config.get("fold_every")
        for i in tqdm(range(n_steps)):
            # perform sampler forward
            sampled_seqs = sampler.step(sequences)
            output_sequences = sampled_seqs.get('output')
            res_sequences += output_sequences

            if (i % fold_every == 0):
                # construct fasta file for folding
                step_fasta_file_name = f"{sampler_name}_{i+1}.fasta"
                step_fasta_file_path = os.path.join(FLAGS.output, step_fasta_file_name)
                step_fasta_file = fasta.FastaFile()
                res_names = [f"{name}|{s}|STEP {i+1}" for name in names]
                for j, res_name in enumerate(res_names):
                    step_fasta_file[res_name] = res_sequences[j]
                step_fasta_file.write(step_fasta_file_path)

                # fold fasta with OmegaFold
                logging.info(f"Folding step {i+1}/{n_steps}")
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
    RUN_CONFIG_FILE = os.path.join(FLAGS.output, CONFIG_FILE)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(res_json, f)
    logging.info(f'results have been written to {OUTPUT_FILE}')

    with open(RUN_CONFIG_FILE, "w") as f:
        json.dump(config, f)
    logging.info(f'config has been written to {CONFIG_FILE}')


if __name__ == '__main__':
    app.run(main)
