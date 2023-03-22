#!/usr/bin/python

"""Soft Brownian Offset - Plotting"""

import pandas as pd

from umap import UMAP

# import plotly.express as px
from plotly.subplots import make_subplots

import plotly.graph_objects as go

import itertools
import numpy as np
from sbo import soft_brownian_offset

csv_data = pd.read_csv('datasets/ADFANet_Shuffled_LabelOK.csv')

# n_normal_samples = 500
# n_ood_samples = 600

# the number of normal samples are 1.5% of the initial data (for performance)
n_normal_samples = int(len(csv_data) * .015)

# the number ood samples are 110% of the initial data
n_ood_samples = n_normal_samples + int(n_normal_samples * .1)

# reduce the number of sample data
csv_data = csv_data[0:n_normal_samples]


# Initialize legend values
number_of_normal_samples = len(csv_data[csv_data['label'] == "normal"])
number_of_attacks_samples = len(csv_data[csv_data['label'] != "normal"])

# Remove the label column
data_initial = csv_data.drop(columns=['label']).to_numpy()

# Number of columns for the plot
n_colrow = 4
d_min = np.linspace(.25, .45, n_colrow)
softness = np.linspace(0, 1, n_colrow)

fig = make_subplots(rows=n_colrow,
                    cols=n_colrow,
                    subplot_titles=[f"Dmin: {d_min_:.2f}, Softness: {soft_:.2f}"
                                    for (
                                        i, (d_min_, soft_)) in enumerate(
                                        itertools.product(d_min, softness))])
fig.update_layout(
    title_text=f"Total Samples: {n_normal_samples+n_ood_samples}, Normal samples: {number_of_normal_samples}, Attack samples: {number_of_attacks_samples}, OOD samples: {n_ood_samples}")

# Create different colors for the labels (yellow for other, purple for OOD and green for normal)
transform_color = np.vectorize(lambda x: (1 if x == 'ood' else (2 if x
                               == 'normal' else 3)))

for (i, (d_min_, softness_)) in enumerate(itertools.product(d_min, softness)):
    (row, col) = (i // n_colrow + 1, i % n_colrow + 1)
    d_off_ = d_min_ * .7

    # tmp variable to store the data
    data = data_initial

    # Run the soft brownian offset algorithm
    data_ood = soft_brownian_offset(data, d_min_, d_off_,
                                    softness=softness_,
                                    n_samples=n_ood_samples)

    data = np.concatenate((data, data_ood))
    # Merge the initial data with the OOD data

    labels = np.concatenate((csv_data.label, [
                            'ood' for x in range(n_ood_samples)]))
    # Merge the initial labels with the OOD labels

    umap_2d = UMAP(n_components=2, init='random', random_state=0,
                   n_neighbors=70, min_dist=.8)
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
