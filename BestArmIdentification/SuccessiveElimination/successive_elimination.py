import json
import numpy as np
from best_arm_identification import best_arm_successive_elimination

gradients = [0.004688906442461286, 2.083805193420233e-17, 5.05756708090395e-19, -1.2102806638436691e-18,
             -8.594900056643715e-20, 0.021915848151867676, 2.65349356349251e-18, -0.0036377622290232553,
             -2.853001338733163e-21, 6.958198181146069e-21, 2.0719496978687343e-36, -2.023173962877157e-19,
             6.496668210232166e-19, -2.7108356604678235e-22, -0.009195623156032194, 0.0055578609270090125,
             -5.8772091262655924e-21, 1.1406230046284987e-18, -1.118706581321763e-18, -2.949766604006259e-22]


if __name__ == "__main__":

    # Load JSON file
    with open("Data/eigenvalue_data_merged.json", "r") as f:
        data = json.load(f)

    reward_fns = []

    # Create reward functions
    for entry in data:
        eigenvalues = np.array(entry["eigenvalues"])
        probabilities = np.array(entry["probabilities"])
        probabilities_normalized = probabilities / probabilities.sum()
        reward_fns.append(
            lambda eigv=eigenvalues, prob=probabilities_normalized: np.random.choice(eigv, p=prob))
    print(f"Total Number of arms: {len(reward_fns)}")
    magnitudes = np.abs(gradients)
    max_magnitude = np.max(magnitudes)
    max_indices = np.where(magnitudes == max_magnitude)[0]
    np.random.seed(42)
    best_arm = (
        best_arm_successive_elimination(reward_fns, delta=0.05, max_rounds=1000000))
    print(f"Identified best arm: {best_arm}")
    print(f"Expected best: {max_indices}")
