import numpy as np
from sbo import soft_brownian_offset

def preprocess_cicids(input_data, n_samples):

    # reduce the number of sample data
    input_data = input_data[0:n_samples]
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

def generate_ood(starting_data, n_ood_samples, d_min,softness):
    d_off = d_min * .7
    return soft_brownian_offset(starting_data, d_min, d_off,
                                softness=softness,
                                n_samples=n_ood_samples)


