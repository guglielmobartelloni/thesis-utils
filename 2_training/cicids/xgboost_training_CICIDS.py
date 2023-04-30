#!/usr/bin/python

"""Soft Brownian Offset - Plotting"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import itertools
import numpy as np
from sbo import soft_brownian_offset

# Create different colors for the labels (yellow for other, purple for OOD and green for normal)
transform_color = np.vectorize(lambda x: (1 if x == 'ood' else (2 if x
                               == 'normal' else 3)))


def train_and_save_model(input_data, n_samples, n_ood_samples):

    # reduce the number of sample data
    input_data = input_data[0:n_samples]
    input_data.replace([np.inf, -np.inf], -1, inplace=True)
    input_data.replace(np.nan, -1, inplace=True)
# input_data.drop(columns=['Date_first_seen'], inplace=True)

# One hot encode the data without the labels
    labels = input_data['label']
    input_data.drop(columns=['label', 'Timestamp'], inplace=True)

    input_data = pd.get_dummies(input_data)
    input_data['label'] = labels


# Select only attack samples
    attacks_packets_types = ['Bot', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP',
                             'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'FTP-BruteForce',
                             'Infilteration', 'SSH-Bruteforce']
# attacks_packets_types = ['1b', 'mailbomb', 'neptune', 'other', 'portsweep']
    attacks_packets = input_data[input_data['label'] != "normal"]
# Replace the attack labels with the attack type
    attacks_packets.replace(attacks_packets_types, 'attack', inplace=True)

# Select only normal Samples
    normal_packets = input_data[input_data['label'] == "normal"]


# Initialize legend values
    # number_of_normal_samples = len(input_data[input_data['label'] == "normal"])
    # number_of_attacks_samples = len(
    #     input_data[input_data['label'] != "normal"])

# Number of columns for the plot
    n_colrow = 1
    d_min = np.linspace(.25, .45, n_colrow)
    softness = np.linspace(0, 1, n_colrow)

    for (i, (d_min_, softness_)) in enumerate(itertools.product(d_min, softness)):
        (row, col) = (i // n_colrow + 1, i % n_colrow + 1)
        d_off_ = d_min_ * .7

        # tmp variable to store the data removing the labels
        data_i = attacks_packets.drop(columns=['label']).to_numpy()

        # Run the soft brownian offset algorithm
        data_ood = soft_brownian_offset(data_i, d_min_, d_off_,
                                        softness=softness_,
                                        n_samples=n_ood_samples)

        # Merge the initial data with the OOD data and normal data
        data_i = np.concatenate((data_i, data_ood, normal_packets.drop(
            columns=['label']).to_numpy()))

        # Merge the initial labels with the OOD labels
        labels = np.concatenate((attacks_packets.label, [
            'attack' for x in range(n_ood_samples)], normal_packets.label))

        # Normalize the labels for the model
        one_hot_encoder = OneHotEncoder(sparse=False)
        y_with_ood = one_hot_encoder.fit_transform(labels.reshape(-1, 1))
        print(y_with_ood)
        X_with_ood = data_i

        # Separate the data
        X_with_ood_train, X_with_ood_test, y_with_ood_train, y_with_ood_test = train_test_split(
            X_with_ood, y_with_ood, test_size=0.2, random_state=123)

        model = xgb.XGBClassifier()
        # Train the model
        model.fit(X_with_ood_train, y_with_ood_train)

        y_with_ood_pred = model.predict(X_with_ood_test)

        model_path = f'./models/xgboost_{d_min_}_{softness_}_CICIDS18_with_100k.json'

        # Save the model for later use
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        accuracy_with_ood = accuracy_score(y_with_ood_test, y_with_ood_pred)
        metthews_with_ood = matthews_corrcoef(y_with_ood_test.argmax(
            axis=1), y_with_ood_pred.argmax(axis=1))

        return accuracy_with_ood, metthews_with_ood


input_data = pd.read_csv('./datasets/CICIDS18_Shuffled_Reduced.csv')
n_samples = len(input_data)
n_ood_samples = 100000
train_and_save_model(input_data, n_samples, 100000)
