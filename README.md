# ProSE
**Pro**tein **S**equence **E**volver

## Install
```sh
$ cd prose
$ python -m venv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
```

## Run example
To run 10 steps of sampler with the example fasta file run:
```sh
$ python run.py --input examples/example.fasta --output out --n 10 --fold_every 2
```
