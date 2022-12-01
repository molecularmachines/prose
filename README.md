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

We manage experiments with [gin-config](https://github.com/google/gin-config). Refer to `config.gin` for the default configuration. To run with default settings:
```sh
$ python run.py
```

To run with specific configuration: 
```sh
$ python run.py --config.gin
```

### Run Configuration

```py
# (in config.gin)
run.start_fasta = './examples/2HBB.fasta' # start sequence fasta
run.experiment = "dev" # name of experiment, used for grouping runs
run.repo = '~/expts/prose/' # file path to save output files and metrics
run.sampler = @MetropolisHastingsSampler # class constructor of sampler to use (note the @) 
run.n_steps = 100 # trajectory length
run.fold_every = 5 # frequency of structure prediction
```


### Interface
To fetch experiments, forward the remote port to a local as you access remote:

```sh
$ ssh -L 16006:127.0.0.1:16006 -K allanc@matlaber14.media.mit.edu
```

Then, inside remote, access the directory where the experiment was saved (`run.repo` in `config.gin`) and run:

```sh
$ aim up -h 127.0.0.1 -p 16006
```

### App

Access the matlaber as above (specifying ports and forwarding). Then run the app with:

```
streamlit run app.py --server.port 16006
```


Access the [Aim UI](https://aimstack.readthedocs.io/en/latest/ui/overview.html) locally through any browser at `localhost:16006`
