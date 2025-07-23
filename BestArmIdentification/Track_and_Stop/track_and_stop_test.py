import json
import numpy as np
from track_and_stop import track_and_stop_discrete
from track_and_stop_continuous import track_and_stop_continuous
from track_and_stop_binned import track_and_stop_binned

gradients = [0.004688906442461286, 2.083805193420233e-17, 5.05756708090395e-19, -1.2102806638436691e-18,
             -8.594900056643715e-20, 0.021915848151867676, 2.65349356349251e-18, -0.0036377622290232553,
             -2.853001338733163e-21, 6.958198181146069e-21, 2.0719496978687343e-36, -2.023173962877157e-19,
             6.496668210232166e-19, -2.7108356604678235e-22, -0.009195623156032194, 0.0055578609270090125,
             -5.8772091262655924e-21, 1.1406230046284987e-18, -1.118706581321763e-18, -2.949766604006259e-22]


if __name__ == "__main__":

    # Load JSON file
    with open("eigenvalue_data_merged.json", "r") as f:
        data = json.load(f)

    reward_fns = []

    # Create reward functions
    for entry in data:
        eigenvalues = np.array(entry["eigenvalues"])
        probabilities = np.array(entry["probabilities"])
        probabilities_normalized = probabilities / probabilities.sum()
        reward_fns.append(
            lambda eigv=eigenvalues, prob=probabilities_normalized: np.random.choice(eigv, p=prob))

    magnitudes = np.abs(gradients)
    max_magnitude = np.max(magnitudes)
    max_indices = np.where(magnitudes == max_magnitude)[0]
    best_arm, emp_dists, counts, t = track_and_stop_binned(reward_fns,
                                                             delta=0.05,
                                                             n_bins=300,
                                                             seed=42)
    print(f"Identified best arm: {best_arm}")
    print(f"Total rounds: {t}")
    print(f"Expected best: {max_indices}")
