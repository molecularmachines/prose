import os
import gin
import json
import esm
from absl import app
from absl import flags
from absl import logging
from tqdm import tqdm
from time import gmtime, strftime
import torch.nn as nn
from samplers import Sampler
from omegafold.__main__ import main as fold
from omegafold.__main__ import model
from aim import Run as Register
from aim import Text, Figure
from typing import List, Tuple
from pathlib import Path
import numpy as np
from metrics.metrics import hamming_distance
from utils import gin_config_to_dict, load_fasta_file, save_fasta_file

from samplers import (
    VanillaSampler,
    MetropolisHastingsSampler, 
    NucleusSampler
)



CONFIG_FILE = "config.gin"

# experiment program run arguments
FLAGS = flags.FLAGS
flags.DEFINE_string("config", CONFIG_FILE, "Path to config json file")




@gin.configurable
def run(
    start_fasta: str,
    sampler: Sampler,
    n_steps: int = gin.REQUIRED,
    fold_every: int = gin.REQUIRED,
    experiment: str = gin.REQUIRED,
    repo: str = gin.REQUIRED,
):
    logging.info(f"sampling with : {sampler}")
    sequences, names = load_fasta_file(start_fasta)

    # load ESM to memory
    logging.info("loading ESM2")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    sampler = sampler(esm_model, alphabet)

    # load omegafold to memory
    logging.info("loading Omegafold")
    folder = model()

    # set up Aim run where we keep track of metrics
    register = Register(experiment=experiment, repo=repo)
    register["hparams"] = gin_config_to_dict(gin.config._OPERATIVE_CONFIG)

    # save files in the same path as Aim, using the hash as dir
    register_dir = str(Path(repo) / register.hash)
    os.makedirs(register_dir, exist_ok=False)
    logging.info(f'Saving Structures to {register_dir}') 

    res_sequences = []

    # sample n times from each sampler
    for step in tqdm(range(n_steps)):
        # perform sampler forward
        output_sequences, sample_metrics = sampler.step(sequences)
        res_sequences.append(output_sequences)
        
        for key, value in sample_metrics.items():
            register.track(value, name=key, step=step)
        register.track(
            Text(output_sequences[0]), name="sequence", step=step)
        
        hamming = hamming_distance(res_sequences[0][0], output_sequences[0])
        register.track(hamming, name='hamming_distance', step=step)

        if step % fold_every == 0:
            # construct fasta file for folding
            step_fasta_file_name = f"{sampler}_{step+1}.fasta"
            step_fasta_file_path = os.path.join(register_dir, step_fasta_file_name)
            step_names = [f"{name}|{sampler}|STEP {step+1}" for name in names]
            save_fasta_file(output_sequences, step_names, step_fasta_file_path)

            # fold fasta with OmegaFold
            logging.info(f"Folding step {step+1}/{n_steps}")
            fold_out = fold(folder, step_fasta_file_path, register_dir)
            confidence = [x["confidence_overall"] for x in fold_out]
            register.track(
                sum(confidence) / len(confidence), name="mean_confidence", step=step
            )
        # next step in trajectory from current step
        sequences = output_sequences

    return register


def main(argv):
    del argv  # Unused.

    # parse config file
    gin.parse_config_file(FLAGS.config)
    register = run()


if __name__ == "__main__":
    app.run(main)
