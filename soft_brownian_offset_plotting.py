#!/usr/bin/python
# -*- coding: utf-8 -*-

"""Soft Brownian Offset - Plotting

"""

import pandas as pd

from umap import UMAP

# import plotly.express as px
from plotly.subplots import make_subplots

import plotly.graph_objects as go

import itertools
import numpy as np
from sbo import soft_brownian_offset

data_initial = pd.read_csv('ADFANet_Shuffled_LabelOK.csv').drop(columns=[
    'label']).to_numpy()[0:500]
n_colrow = 2
d_min = np.linspace(.25, .45, n_colrow)
softness = np.linspace(0, 1, n_colrow)

fig = make_subplots(rows=n_colrow,
                    cols=n_colrow,
                    subplot_titles=[f"Dmin: {d_min_}, Softness: {soft_}" for (
                        i, (d_min_, soft_)) in enumerate(itertools.product(d_min, softness))]
                    )

transform_color = np.vectorize(lambda x: (1 if x == 'ood' else (2 if x
                               == 'normal' else 3)))

for (i, (d_min_, softness_)) in enumerate(itertools.product(d_min, softness)):
    (row, col) = (i // n_colrow + 1, i % n_colrow + 1)
    d_off_ = d_min_ * .7

    data = data_initial
    data_ood = soft_brownian_offset(data, d_min_, d_off_,
                                    softness=softness_, n_samples=600)

    data = np.concatenate((data, data_ood))

    labels = np.concatenate((pd.read_csv(
        'ADFANet_Shuffled_LabelOK.csv'
    )[0:500].label, ['ood' for x in range(600)]))

    umap_2d = UMAP(n_components=2, init='random', random_state=0,
                   n_neighbors=50, min_dist=.0)
    proj_2d = umap_2d.fit_transform(data)

    fig.add_trace(go.Scatter(
        x=proj_2d[:, 0],
        y=proj_2d[:, 1],
        mode='markers',
        marker=dict(
            color=transform_color(labels),
            colorscale='Viridis',
            showscale=True
        )), row=row, col=col)

    # fig_2d = px.scatter(proj_2d, x=0, y=1, color=labels,
    #                     labels={'color': 'label'})
    # fig_2d.show()

fig.show()
