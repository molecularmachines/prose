import os
from biotite.sequence.io.fasta import FastaFile
from random import choices

peptide = "PPPRPPK"
peptide_name = "Grb2 SH3 Binding Domain"
protein_length = 60

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'
}

residues = list(restype_1to3.keys())
random_picks = choices(residues, k=(protein_length - len(peptide)))
random_protein = "".join(random_picks)
currdir = os.path.dirname(os.path.realpath(__file__))
exp_dir = os.path.join(currdir, "experiments")
os.makedirs(exp_dir, exist_ok=True)

for i in range(len(random_protein)):
    sequence_name = f"{peptide_name}|POSITION {i+1}"
    sequence = random_protein[:i] + peptide + random_protein[i:]
    out_file = FastaFile()
    out_file[sequence_name] = sequence
    out_file.write(os.path.join(exp_dir, f"peptide_{i+1}.fasta"))
