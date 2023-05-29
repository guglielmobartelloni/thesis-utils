#!/usr/bin/env python3
import warnings
from utils import (
    preprocess_adfa,
    preprocess_cicids,
    generate_ood,
    merge_data_with_ood,
    train_and_save_model,
    get_models,
    test_model,
    merge_data,
    save_in_csv,
    merge_data_normal_only_with_ood,
)
import pandas as pd
import numpy as np
import csv

# N+(N+AV OOD)
# N+AV+(N + AV OOD)
# N+N OOD
# N+AV+N OOD (FATTO)
# N+AV OOD
# N+AV+AV OOD (FATTO)

warnings.filterwarnings("ignore")

type = "av_ood"


def training_ADFA(normal_samples, attack_samples, N, N_ood, d_min, softness):
    ood_samples = generate_ood(
        attack_samples, N_ood, d_min, softness
    )

    merged_samples, labels = merge_data_normal_only_with_ood(
        normal_samples, ood_samples
    )

    # Check if the number of samples are right
    # assert len(merged_samples) == N + N_ood and len(labels) == N + N_ood

    model_path = f"./results/models/ADFA/adfa_{type}_addr_{N}_{N_ood}.json"
    print("Training model...")
    print(train_and_save_model(merged_samples, labels, 0.0001, model_path))
    print(f"Model saved in {model_path}")


def training_CICIDS():
    ood_samples = generate_ood(
        pd.concat((normal_samples, attack_samples)), N_ood, d_min, softness
    )

    merged_samples, labels = merge_data_normal_only_with_ood(
        normal_samples, ood_samples
    )

    # Check if the number of samples are right
    # assert len(merged_samples) == N + N_ood and len(labels) == N + N_ood

    model_path = f"./results/models/CICIDS/cicids_{type}_{N}_{N_ood}.json"
    print("Training model...")
    print(train_and_save_model(merged_samples, labels, 0.5, model_path))
    print(f"Model saved in {model_path}")


def testing(models_path="./results/models/CICIDS"):
    models = get_models(models_path)
    test_data, test_labels = merge_data(normal_samples, attack_samples)
    assert len(test_data) == N and len(test_labels) == N
    results = dict()

    for model in models:
        # For every model make ten tests and take the average
        metric_sum = 0
        for i in range(10):
            model_metric = test_model(model, test_data, test_labels)
            metric_sum += model_metric
        metric_sum /= 10
        results[model] = metric_sum

    # Sort the results
    results = dict(sorted(results.items(), key=lambda x: x[1]))
    # Save the results
    save_in_csv(f"./results/adfa_results_n_{type}-odd.csv", results)


cicids_dataset = pd.read_csv("./datasets/CICIDS18_Shuffled_Reduced.csv")
adfa_dataset = pd.read_csv("./datasets/ADFANet_Shuffled_LabelOK.csv")
N = len(adfa_dataset)
d_min = 0.25
softness = 0.0

normal_samples, attack_samples = preprocess_adfa(adfa_dataset, N)

for number_samples in [0.25 * N, 0.5 * N, 0.75 * N, 0.95 * N]:
    for number_ood in [
        1,
        number_samples * 0.5,
        number_samples * 2,
        number_samples * 3,
        number_samples * 4,
    ]:
        training_ADFA(
            normal_samples,
            attack_samples,
            int(number_samples),
            int(number_ood),
            d_min,
            softness,
        )
        models_path = "./results/models/ADFA/"
        testing(models_path)

# training_CICIDS()
