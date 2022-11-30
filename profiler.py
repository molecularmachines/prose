import numpy as np
import pandas as pd
import os
from aim import Run, Repo

from absl import app
from absl import flags
from pathlib import Path

import streamlit as st
from stmol import showmol

import py3Dmol

FLAGS = flags.FLAGS
flags.DEFINE_string('repo', '~/expts/prose/', 'path to aim repository')
# count = st_autorefresh(interval=1000, limit=100, key="fizzbuzzcounter")
st.set_page_config(page_title='Prose Trajectory')



repo = '~/expts/prose'
repo_path = os.path.expanduser(repo)
repo = Repo(repo_path)

all_runs = repo.query_runs("")

avail_hashes = []
for run in all_runs.iter_runs():
    avail_hashes.append(run.run.hash)

selected_run = filter(lambda run: run.run.hash == selected_hash, all_runs)

structure = st.selectbox(label="Select Run", options=avail_hashes)

repo_dir = Path(repo_path) / structure
for file_name in os.listdir(str(repo_dir)):
    if file_name.endswith(".pdb"):
        step = file_name[file_name.rindex('|') + 1:file_name.index('.pdb')]

        view = py3Dmol.view()
        with open(str(repo_dir / file_name)) as file:
            system = "".join([x for x in file])
            view.addModel(system)
        view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
        view.zoomTo()
        with st.expander(step):
            showmol(view, height=500, width=800)






