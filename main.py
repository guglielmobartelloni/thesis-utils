#!/usr/bin/env python3

from utils import preprocess_adfa, preprocess_cicids, generate_ood, merge_data_with_ood, train_and_save_model, get_models, test_model, merge_data
import pandas as pd
import numpy as np


# N+(N+AV OOD)
# N+AV+(N + AV OOD)
# N+N OOD
# N+AV+N OOD (FATTO)
# N+AV OOD
# N+AV+AV OOD (FATTO)


cicids_dataset = pd.read_csv('./datasets/CICIDS18_Shuffled_Reduced.csv')
N = int(len(cicids_dataset) * 0.5)
N = 154996
N_ood = int(N * 0.25)
N_ood = 1
d_min = 0.25
softness = 0.0

normal_samples, attack_samples = preprocess_cicids(cicids_dataset, N)


def training():
    ood_samples = generate_ood(attack_samples, N_ood, d_min, softness)

    merged_samples, labels = merge_data_with_ood(
        normal_samples, attack_samples, ood_samples)

    # Check if the number of samples are right
    assert len(merged_samples) == N + N_ood and len(labels) == N + N_ood

    model_path = f'./results/models/cicids_attack_{N}_{N_ood}.json'
    print("Training model...")
    print(train_and_save_model(merged_samples, labels, 0.5, model_path))
    print(f"Model saved in {model_path}")


def testing():
    models_path = './results/models/'
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
    for result in results:
        print(f"{results[result]} - {result}")


# training()
testing()
