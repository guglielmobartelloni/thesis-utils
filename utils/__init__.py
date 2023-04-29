import numpy as np
from sbo import soft_brownian_offset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


def preprocess_cicids(input_data, n_samples):

    # reduce the number of sample data
    input_data = input_data[0:n_samples]
    input_data.drop(columns=['Timestamp'], inplace=True)
    input_data.replace([np.inf, -np.inf], -1, inplace=True)
    input_data.replace(np.nan, -1, inplace=True)
    # Select only attack samples
    attacks_packets_types = ['Bot', 'DDOS attack-HOIC', 'DDOS attack-LOIC-UDP',
                             'DoS attacks-Hulk', 'DoS attacks-SlowHTTPTest', 'FTP-BruteForce',
                             'Infilteration', 'SSH-Bruteforce']
    # attacks_packets_types = ['1b', 'mailbomb', 'neptune', 'other', 'portsweep']
    attacks_packets = input_data[input_data['label'] != "normal"]
    # Replace the attack labels with the attack type
    attacks_packets.replace(attacks_packets_types, 'attack', inplace=True)
    normal_packets = input_data[input_data['label'] == "normal"]

    return (normal_packets, attacks_packets)


def preprocess_adfa(input_data, n_normal_samples):

    n_samples = n_normal_samples

    # reduce the number of sample data
    input_data = input_data[0:n_samples]
    # input_data.drop(columns=['Date_first_seen'], inplace=True)

    # One hot encode the data without the labels
    input_data.drop(columns=['label'], inplace=True)

    attacks_packets_types = ['1b', 'mailbomb', 'neptune', 'other', 'portsweep']
    attacks_packets = input_data[input_data['label'] != "normal"]
    # Replace the attack labels with the attack type
    attacks_packets.replace(attacks_packets_types, 'attack', inplace=True)

    normal_packets = input_data[input_data['label'] == "normal"]

    return (normal_packets, attacks_packets)


def generate_ood(starting_data, n_ood_samples, d_min, softness):
    d_off = d_min * .7
    return soft_brownian_offset(starting_data.drop(columns=['label']).to_numpy(), d_min, d_off,
                                softness=softness,
                                n_samples=n_ood_samples)


# Merge the initial data with the OOD data and normal data
def merge_data(normal_data, attacks_data, ood_data):
    return (np.concatenate((normal_data.drop(columns=['label']), attacks_data.drop(columns=['label']), ood_data)), merge_labels(normal_data, attacks_data, ood_data))

# Merge the initial labels with the OOD labels


def merge_labels(normal_data, attacks_data, ood_data):
    return np.concatenate((normal_data.label, attacks_data.label, ['attack' for x in range(len(ood_data))]))


def train_and_save_model(data,labels, model_path):
    # Normalize the labels for the model
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.reshape(-1, 1))
    X = data_i

    # Separate the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)

    model = xgb.XGBClassifier()
    # Train the model
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)

    # Save the model
    model.save_model(model_path)
