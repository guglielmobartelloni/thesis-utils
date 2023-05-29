import numpy as np
import csv
from sbo import soft_brownian_offset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
import xgboost as xgb
from os import listdir
from os.path import join


def preprocess_cicids(input_data, n_samples):
    # reduce the number of sample data
    input_data = input_data[0:n_samples]
    input_data.drop(columns=["Timestamp"], inplace=True)
    input_data.replace([np.inf, -np.inf], -1, inplace=True)
    input_data.replace(np.nan, -1, inplace=True)
    # Select only attack samples
    attacks_packets_types = [
        "Bot",
        "DDOS attack-HOIC",
        "DDOS attack-LOIC-UDP",
        "DoS attacks-Hulk",
        "DoS attacks-SlowHTTPTest",
        "FTP-BruteForce",
        "Infilteration",
        "SSH-Bruteforce",
    ]
    attacks_packets = input_data[input_data["label"] != "normal"]
    # Replace the attack labels with the attack type
    attacks_packets.replace(attacks_packets_types, "attack", inplace=True)
    normal_packets = input_data[input_data["label"] == "normal"]

    return (normal_packets, attacks_packets)


def preprocess_adfa(input_data, n_normal_samples):
    n_samples = n_normal_samples

    # reduce the number of sample data
    input_data = input_data[0:n_samples]

    attacks_packets_types = ["1b", "mailbomb", "neptune", "other", "portsweep"]
    attacks_packets = input_data[input_data["label"] != "normal"]
    # Replace the attack labels with the attack type
    attacks_packets.replace(attacks_packets_types, "attack", inplace=True)

    normal_packets = input_data[input_data["label"] == "normal"]

    return (normal_packets, attacks_packets)


def generate_ood(starting_data, n_ood_samples, d_min, softness):
    d_off = d_min * 0.7
    return soft_brownian_offset(
        starting_data.drop(columns=["label"]).to_numpy(),
        d_min,
        d_off,
        softness=softness,
        n_samples=n_ood_samples,
    )


# Merge the initial data with the OOD data and normal data
def merge_data_with_ood(normal_data, attacks_data, ood_data):
    return (
        np.concatenate(
            (
                normal_data.drop(columns=["label"]),
                attacks_data.drop(columns=["label"]),
                ood_data,
            )
        ),
        merge_labels(normal_data, attacks_data, ood_data),
    )


def merge_data_normal_only_with_ood(normal_data, ood_data):
    return (
        np.concatenate((normal_data.drop(columns=["label"]), ood_data)),
        merge_normal_only_labels(normal_data, ood_data),
    )


def merge_data(normal_data, attacks_data):
    merged_data = np.concatenate(
        (normal_data.drop(columns=["label"]), attacks_data.drop(columns=["label"]))
    )
    merged_labels = np.concatenate((normal_data.label, attacks_data.label))
    return (merged_data, merged_labels)


def merge_normal_only_labels(normal_data, ood_data):
    return np.concatenate((normal_data.label, ["attack" for x in range(len(ood_data))]))


def merge_labels(normal_data, attacks_data, ood_data):
    return np.concatenate(
        (
            normal_data.label,
            attacks_data.label,
            ["attack" for x in range(len(ood_data))],
        )
    )


def train_and_save_model(data, labels, test_size, model_path):
    # Normalize the labels for the model
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels.reshape(-1, 1))
    X = data
    
    # Separate the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=123
    )

    model = xgb.XGBClassifier()
    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    model.save_model(model_path)

    y_predicted = model.predict(X_test)
    return matthews_corrcoef(y_test, y_predicted)


def is_json(f):
    return f.endswith(".json")


def get_models(models_path):
    return [
        join(models_path, f)
        for f in listdir(models_path)
        if is_json(join(models_path, f))
    ]


def test_model(model_path, data_x, data_y):
    encoder = LabelEncoder()
    data_y = encoder.fit_transform(data_y.reshape(-1, 1))
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    y_predicted = model.predict(data_x)
    return matthews_corrcoef(data_y, y_predicted)


def save_in_csv(filename, results):
    with open(filename, "w", newline="") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["Model", "Mode", "Starting Samples", "OOD Samples", "Metric"])
        for result in results:
            writer.writerow(
                [
                    result,
                    result.split("_")[1],
                    result.split("_")[2],
                    result.split("_")[3],
                    results[result],
                ]
            )
