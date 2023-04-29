#!/usr/bin/env python3

from utils import preprocess_adfa, preprocess_cicids, generate_ood, merge_data, train_and_save_model
import pandas as pd
import numpy as np


# N+(N+AV OOD) 
# N+AV+(N + AV OOD)
# N+N OOD 
# N+AV+N OOD (FATTO)
# N+AV OOD 
# N+AV+AV OOD (FATTO)


cicids_dataset = pd.read_csv('./datasets/CICIDS18_Shuffled_Reduced.csv')
N = len(cicids_dataset)
N_ood = 50000

normal_samples, attack_samples = preprocess_cicids(cicids_dataset, N)

ood_samples = generate_ood(attack_samples, N_ood, 0.25, 0.0)

merged_samples, labels= merge_data(normal_samples, attack_samples, ood_samples)

# Check if the number of samples are right
assert len(merged_samples) == N + N_ood and len(labels) == N + N_ood

model_path= f'./results/models/cicids_{N}_{N_ood}.json'
print("Training model...")
print(train_and_save_model(merged_samples, labels, 0.5, './results/models/cicids_1000_1000_0.25.json'))
print(f"Model saved in {model_path}")



