# Evolutron
Evolving Novel Protein Sequences through Language Model Sampling

## Install
```sh
$ cd evolutron
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Run example
To run 10 steps of sampler with the example fasta file run:
```sh
$ python run.py --input examples/example.fasta --output out --n 10 --fold_every 2
```
