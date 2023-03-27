#!/usr/bin/python

"""Soft Brownian Offset - Plotting"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from umap import UMAP

# import plotly.express as px
from plotly.subplots import make_subplots

import plotly.graph_objects as go

import itertools
import numpy as np
from sbo import soft_brownian_offset

model = xgb.XGBClassifier()


input_data = pd.read_csv('datasets/ADFANet_Shuffled_LabelOK.csv')

# n_normal_samples = 500
# n_ood_samples = 600

# the number of normal samples are 1.5% of the initial data (for performance)
# n_normal_samples = int(len(csv_data) * .015)
n_normal_samples = int(len(input_data) * .015)

# the number ood samples are 110% of the initial data
# n_ood_samples = n_normal_samples + int(n_normal_samples * .1)
n_ood_samples = 2000

# reduce the number of sample data
input_data = input_data[0:n_normal_samples]

# Select only attack samples
data_used = input_data[input_data['label'] != "normal"]

# Initialize legend values
number_of_normal_samples = len(input_data[input_data['label'] == "normal"])
number_of_attacks_samples = len(input_data[input_data['label'] != "normal"])

# Remove the label column
data_initial = data_used.drop(columns=['label']).to_numpy()

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
    X_without_ood = data
    y_without_ood = data_used.label

    # Run the soft brownian offset algorithm
    data_ood = soft_brownian_offset(data, d_min_, d_off_,
                                    softness=softness_,
                                    n_samples=n_ood_samples)

    # Merge the initial data with the OOD data and normal data
    data = np.concatenate((data, data_ood, input_data[input_data['label'] == "normal"].drop(
        columns=['label']).to_numpy()))

    # For comprension
    X_with_ood = data

    # Merge the initial labels with the OOD labels
    labels = np.concatenate((data_used.label, [
        'ood' for x in range(n_ood_samples)], input_data[input_data['label'] == "normal"].label))

    labels_for_model = np.concatenate((['attack' for x in range(len(data_used))], [
        'attack' for x in range(n_ood_samples)], input_data[input_data['label'] == "normal"].label))

    # Normalize the labels for the model
    le = LabelEncoder()
    le.fit(labels_for_model)
    y_with_ood = le.transform(labels_for_model)

    # Normalize the labels for the model
    le = LabelEncoder()
    le.fit(data_used.label)
    y_without_ood = le.transform(data_used.label)

    # Separate the data
    X_with_ood_train, X_with_ood_test, y_with_ood_train, y_with_ood_test = train_test_split(
        X_with_ood, y_with_ood, test_size=0.5, random_state=123)

    X_without_ood_train, X_without_ood_test, y_without_ood_train, y_without_ood_test = train_test_split(
        X_without_ood, y_without_ood, test_size=0.5, random_state=123)

    model = xgb.XGBClassifier()
    print(len(X_with_ood_train))
    print(len(y_with_ood_train))
    # Train the model
    model.fit(X_with_ood_train, y_with_ood_train)

    y_with_ood_pred = model.predict(X_without_ood_test)
    accuracy_with_ood = accuracy_score(y_without_ood_test, y_with_ood_pred)
    metthews_with_ood = matthews_corrcoef(y_without_ood_test, y_with_ood_pred)

    model.fit(X_without_ood_train, y_without_ood_train)

    y_without_ood_pred = model.predict(X_without_ood_test)
    accuracy_without_ood = accuracy_score(
        y_without_ood_test, y_without_ood_pred)
    metthews_without_ood = matthews_corrcoef(
        y_without_ood_test, y_without_ood_pred)

    print("WITH OOD Accuracy: {:.2f}\tMetthews: {:.2f}".format(
        accuracy_with_ood, metthews_with_ood))
    print("WITHOUT OOD Accuracy: {:.2f}\tMetthews: {:.2f}".format(
        accuracy_without_ood, metthews_without_ood))

    # umap_2d = UMAP(n_components=2, init='random', random_state=0,
    #                n_neighbors=70, min_dist=.8)
    # proj_2d = umap_2d.fit_transform(data)
    #
    # fig.add_trace(go.Scatter(
    #     x=proj_2d[:, 0],
    #     y=proj_2d[:, 1],
    #     mode='markers',
    #     marker=dict(
    #         color=transform_color(labels),
    #         colorscale='Viridis',
    #         showscale=True
    #     )), row=row, col=col)

# fig.show()
