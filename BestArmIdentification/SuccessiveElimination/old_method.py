import numpy as np
from collections import defaultdict


def estimate_shots_for_gen_idx(variances_by_generator, gen_idx, target_accuracy=0.001):
    fragment_variances = variances_by_generator[gen_idx]
    total_variance = sum(fragment_variances)
    required_shots = int(np.ceil(total_variance / target_accuracy**2))
    return required_shots



if __name__ == "__main__":
    eigen_data = np.load("eigen_info.npz", allow_pickle=True)
    allocation_data = np.load("allocation_info.npz", allow_pickle=True)

    eigen_info = [eigen_data[key].item() for key in eigen_data]
    allocation_info = [allocation_data[key].item() for key in allocation_data]

    # Step 2: Group fragments by generator_idx
    fragments_by_generator = defaultdict(list)
    variances_by_generator = defaultdict(list)
    expectation_values_dict = defaultdict(list)

    for entry in eigen_info:
        gen_idx = entry['generator_idx']
        eigenvalues = np.array(entry['eigenvalues'])
        probabilities = np.array(entry['probabilities'])

        # Mean (expected value)
        mean = np.sum(probabilities * eigenvalues)

        # Expected value of square
        mean_sq = np.sum(probabilities * eigenvalues ** 2)

        # Variance
        variance = mean_sq - mean ** 2

        # Store per-fragment variance and mean
        variances_by_generator[gen_idx].append(variance)
        expectation_values_dict[gen_idx].append(mean)

    total_shots = 0
    mean_list = []
    for gen_idx, expectation_values in expectation_values_dict.items():
        required_shots = estimate_shots_for_gen_idx(variances_by_generator, 0, target_accuracy=0.001)
        mean_list.append(sum(expectation_values_dict[gen_idx]))
        total_shots += required_shots
    print(f"Required total shots: {total_shots}")
    print(f"Mean list: {mean_list}")
    mean_array = np.array(mean_list)
    top_two_indices = np.argsort(np.abs(mean_array))[-2:][
                      ::-1]  # descending order
    top_two_values = mean_array[top_two_indices]

    print(f"Top two indices: {top_two_indices}")
    print(f"Top two values: {top_two_values}")

    print(f"Difference: {abs(top_two_values[0]) - abs(top_two_values[1])}")
