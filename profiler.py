import numpy as np
import streamlit as st
import pandas as pd
import os
from aim import Run, Repo

from absl import app
from absl import flags
from pathlib import Path

from stmol import showmol
import plotly.graph_objects as go
from streamlit_option_menu import option_menu


import py3Dmol

FLAGS = flags.FLAGS
flags.DEFINE_string('repo', '~/expts/prose/', 'path to aim repository')
# count = st_autorefresh(interval=1000, limit=100, key="fizzbuzzcounter")


st.set_page_config(
    page_title='ProSE',
    layout='wide',
    page_icon='https://cdn-icons-png.flaticon.com/512/4773/4773245.png'
)

repo = '~/expts/prose'
repo_path = os.path.expanduser(repo)
repo = Repo(repo_path)

all_runs = repo.query_runs("")

avail_hashes = []
for run in all_runs.iter_runs():
    avail_hashes.append(run.run.hash)

with st.sidebar:
    selected = option_menu("ProSE", ["Analyze", 'Launch'], 
        icons=['clipboard', 'lightning-charge'], menu_icon="null", default_index=1)


        
col1, col2 = st.columns(2)

with col1:
    st.header('Analyze', anchor=None)
    selected_hash = st.selectbox(label="Select Run", options=avail_hashes)
    run_dir = Path(repo_path) / selected_hash
    run = repo.get_run(selected_hash)


    system_metrics, metrics = [], []
    for metric in run.metrics():
        value = metric.values.sparse_numpy()[1]
        tupl = (metric.name, value)
        if metric.name.startswith('__'):
            system_metrics.append(tupl)
        else:
            metrics.append(tupl) 


    def plot_metrics(metrics_list):
        for name, value in metrics_list:
            with st.expander(name, expanded=True):    
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=value, name=name,
                                    line_shape='linear'))
                fig.update_layout(
                        margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                )
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                st.plotly_chart(fig)



    tab1, tab2, tab3 = st.tabs(["Metrics", "System Metrics", "Weights"])

    with tab2:
        plot_metrics(system_metrics)

    with tab1:
        plot_metrics(metrics)


with col2: 
    st.subheader('Structures', anchor=None)
    for file_name in os.listdir(str(run_dir)):
        if file_name.endswith(".pdb"):
            step = file_name[file_name.rindex('|') + 1:file_name.index('.pdb')]

            view = py3Dmol.view()
            with open(str(run_dir / file_name)) as file:
                system = "".join([x for x in file])
                view.addModel(system)
            view.setStyle({'model': -1}, {"cartoon": {'color': 'spectrum'}})
            view.zoomTo()
            view.setBackgroundColor(0x00000000, 0);
            with st.expander(step, expanded=True):
                st.components.v1.html(view._make_html(), height=500,width=500)






