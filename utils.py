from typing import List, Tuple

from biotite.sequence.io import fasta

def load_fasta_file(input: str) -> List[Tuple[str]]:
    # read input fasta file
    fasta_file = fasta.FastaFile.read(input)
    fasta_sequences = fasta.get_sequences(fasta_file)
    sequences = list(fasta_sequences.values())
    sequences = [str(s) for s in sequences]
    names = list(fasta_sequences.keys())
    return sequences, names

def save_fasta_file(sequences, names, save_path):
    step_fasta_file = fasta.FastaFile() 
    for j, res_name in enumerate(names):
        step_fasta_file[res_name] = sequences[j]
    step_fasta_file.write(save_path)

def gin_config_to_dict(gin_config: dict):
    """
    Originally from https://github.com/google/gin-config/issues/154
    Parses the gin configuration to a dictionary. Useful for logging to e.g. W&B
    :param gin_config: the gin's config dictionary. Can be obtained by gin.config._OPERATIVE_CONFIG
    :return: the parsed (mainly: cleaned) dictionary
    """
    data = {}
    for key in gin_config.keys():
        name = key[1].split(".")[1]
        values = gin_config[key]
        for k, v in values.items():
            data[".".join([name, k])] = str(v)
    return data
