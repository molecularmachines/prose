# ProSE
**Pro**tein **S**equence **E**volver

## Install
```sh
$ cd prose
$ python -m venv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
```

## Run Example
To run an experiment an input FASTA file, an output directory and a config file are required.
```sh
$ python run.py --input examples/example.fasta --output out --config config.json
```

The `samplers` dictionary in `config.json` determines which samplers run in the experiment. 
Each attribute in the sampler dictionary will be available as a sampler's class member.
