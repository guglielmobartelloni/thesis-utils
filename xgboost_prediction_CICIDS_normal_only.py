#!/usr/bin/python

"""Soft Brownian Offset - Plotting"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Create different colors for the labels (yellow for other, purple for OOD and green for normal)
transform_color = np.vectorize(lambda x: (1 if x == 'ood' else (2 if x
                               == 'normal' else 3)))


input_data = pd.read_csv('./datasets/CICIDS18_Shuffled_Reduced.csv')

# the number of normal samples are 1.5% of the initial data (for performance)
n_samples = len(input_data)
# n_samples = int(len(input_data) * .055)

# the number ood samples are 110% of the initial data
# DON'T TOUCH THIS VALUE cause we don't want to generate data in this case
n_ood_samples = 1

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
print(input_data.head())


# Select only attack samples
# attacks_packets_types = ['bruteForce', 'dos', 'pingScan', 'portScan']
attacks_packets_types = ['Bot', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP',
                         'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'FTP-BruteForce',
                         'Infilteration', 'SSH-Bruteforce']
attacks_packets = input_data[input_data['label'] != "normal"]
# Replace the attack labels with the attack type
attacks_packets.replace(attacks_packets_types, 'attack', inplace=True)

# Select only normal Samples
normal_packets = input_data[input_data['label'] == "normal"]


# Initialize legend values
number_of_normal_samples = len(input_data[input_data['label'] == "normal"])
number_of_attacks_samples = len(input_data[input_data['label'] != "normal"])


# tmp variable to store the data removing the labels
data_i = attacks_packets.drop(columns=['label']).to_numpy()


# Merge the initial data with the OOD data and normal data
data_i = np.concatenate((data_i,normal_packets.drop(
    columns=['label']).to_numpy()))

# Merge the initial labels with the OOD labels
# labels = np.concatenate((attacks_packets.label, [
#     'ood' for x in range(n_ood_samples)], normal_packets.label))
labels = np.concatenate((attacks_packets.label, normal_packets.label))

print(labels)

# Normalize the labels for the model
one_hot_encoder = OneHotEncoder(sparse_output=False)
y_with_ood = one_hot_encoder.fit_transform(labels.reshape(-1, 1))
print(y_with_ood)
X_with_ood = data_i

# Separate the data
X_with_ood_train, X_with_ood_test, y_with_ood_train, y_with_ood_test = train_test_split(
    X_with_ood, y_with_ood, test_size=0.2, random_state=123)

model = xgb.XGBClassifier()
# Train the model
# model.fit(X_with_ood_train, y_with_ood_train)
model.load_model('./models/xgboost_0.45_0.3333333333333333_CICIDS18_with_100k_only_normal.json')

y_with_ood_pred = model.predict(X_with_ood_test)

# Save the model for later use
accuracy_with_ood = accuracy_score(y_with_ood_test, y_with_ood_pred)
metthews_with_ood = matthews_corrcoef(y_with_ood_test.argmax( # type: ignore
    axis=1), y_with_ood_pred.argmax(axis=1)) # type: ignore

print("WITHOUT OOD Accuracy {:.4f}\tMetthews: {:.4f}".format(
    accuracy_with_ood, metthews_with_ood))
