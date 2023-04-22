#!/usr/bin/python

"""Test Data generation SBO"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


if not os.path.exists("images"):
    os.mkdir("images")
if not os.path.exists("html"):
    os.mkdir("html")

from umap import UMAP

# import plotly.express as px
from plotly.subplots import make_subplots

import plotly.graph_objects as go

import itertools
import numpy as np
from sbo import soft_brownian_offset


def maximum_absolute_scaling(df):
    # copy the dataframe
    df_scaled = df.copy()
    # apply maximum absolute scaling
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled


def robust_scaling(df):
    # copy the dataframe
    df_robust = df.copy()
    # apply robust scaling
    for column in df_robust.columns:
        df_robust[column] = (df_robust[column] - df_robust[column].median()) / (
            df_robust[column].quantile(0.75) - df_robust[column].quantile(0.25))
    return df_robust


def gen_test(input_data, filename):

    print("Generating data for: " + filename)
    n_normal_samples = 50000
    # n_normal_samples = int(len(input_data))

    n_ood_samples = int(n_normal_samples * 0.25)

# reduce the number of sample data
    input_data = input_data[0:n_normal_samples]
    input_data.replace([np.inf, -np.inf], -1, inplace=True)
    input_data.replace(np.nan, -1, inplace=True)

# Select only attack samples
    data_used = input_data[input_data['label'] != "normal"]

# Initialize legend values
    number_of_normal_samples = len(input_data[input_data['label'] == "normal"])
    number_of_attacks_samples = len(
        input_data[input_data['label'] != "normal"])

# Remove the label column and remve NaN values
    data_initial = data_used.drop(columns=['label', 'Timestamp'])
    # Replace inf values with -1

    # scaler = StandardScaler()
    # scaler.fit(data_initial)
    # data_initial = scaler.transform(data_initial)

    data_initial = data_initial.to_numpy()

# Number of columns for the plot
    n_colrow = 2
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

        # Merge the initial data with the OOD data and normal data
        data = np.concatenate((data, data_ood, input_data[input_data['label'] == "normal"].drop(
            columns=['label', 'Timestamp']).to_numpy()))

        # Merge the initial labels with the OOD labels
        labels = np.concatenate((data_used.label, [
            'ood' for x in range(n_ood_samples)], input_data[input_data['label'] == "normal"].label))

        umap_2d = UMAP(n_components=2, init='random', random_state=0,
                       n_neighbors=70, min_dist=.8)
        proj_2d = umap_2d.fit_transform(data)

        fig.add_trace(go.Scatter(
            x=proj_2d[:, 0],
            y=proj_2d[:, 1],
            mode='markers',
            showlegend=False,
            marker=dict(
                color=transform_color(labels),
                colorscale='Viridis',
                showscale=False
            )), row=row, col=col)

        # fig_2d = px.scatter(proj_2d, x=0, y=1, color=labels,
        #                     labels={'color': 'label'})
        # fig_2d.show()

    # fig.show()

    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Normal',
        marker=dict(
            size=10,
            color="rgb(35,144,139)",
            colorscale='Viridis',
        )))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='Attack',
        marker=dict(
            size=10,
            color="rgb(253,231,37)",
            colorscale='Viridis',
        )))
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        name='OOD',
        marker=dict(
            size=10,
            color="rgb(69,13,84)",
            colorscale='Viridis',
        )))
    # fig.show()
    fig.write_image("images/" + filename + '.pdf')
    fig.write_html("html/" + filename + '.html')


input_data = pd.read_csv('datasets/CICIDS18_Shuffled_Reduced.csv')
gen_test(input_data, 'CICIDS18_attack_only_25k')
