import numpy as np
from collections import defaultdict
from best_arm_identification import integer_allocations_one_guaranteed, get_rewards_from_fragments


def calculate_variance(eigenvalues, probabilities):
    import numpy as np

    eigenvalues = np.array(eigenvalues)
    probabilities = np.array(probabilities)

    mean = np.sum(probabilities * eigenvalues)
    mean_squared = np.sum(probabilities * eigenvalues**2)

    variance = mean_squared - mean**2
    return variance


if __name__ == '__main__':
    which_arm = 0
    eigen_data = np.load("eigen_info.npz", allow_pickle=True)
    allocation_data = np.load("allocation_info.npz", allow_pickle=True)

    eigen_info = [eigen_data[key].item() for key in eigen_data]
    allocation_info = [allocation_data[key].item() for key in allocation_data]

    # Step 2: Group fragments by generator_idx
    fragments_by_generator = defaultdict(list)
    expectation_values_dict = defaultdict(float)
    for entry in eigen_info:
        gen_idx = entry['generator_idx']
        eigenvalues = entry['eigenvalues']
        probabilities = entry['probabilities']

        variance = calculate_variance(eigenvalues, probabilities)
        print(f"Variance: {variance} for generator: {gen_idx}")

        mean = np.sum(eigenvalues * probabilities)
        expectation_values_dict[gen_idx] += mean

        # Create a fragment function that captures the current eigenvalues and probabilities
        def create_fragment(eigv, prob):
            return lambda: np.random.choice(eigv, p=prob)

        fragments_by_generator[gen_idx].append(create_fragment(eigenvalues, probabilities))

    # Step 3: Rebuild reward_fns and allocations
    reward_fns = []
    allocations = []
    expectation_values = []

    # Keep generator_idx order consistent with allocation_info
    for alloc_entry in allocation_info:
        gen_idx = alloc_entry['generator_idx']
        allocation = alloc_entry['allocation']

        reward_fns.append(fragments_by_generator[gen_idx])
        allocations.append(allocation)
        expectation_values.append(expectation_values_dict[gen_idx])

    print(
        f"Expectation values <[H, G]> but with selected fragments: {expectation_values[which_arm]}")

    fifth_arm = reward_fns[which_arm]

    total_allocation = 1000000

    uniform_allocation = [1 / len(allocations[which_arm]) for _ in range(len(allocations[which_arm]))]
    actual_allocation = integer_allocations_one_guaranteed(allocations[which_arm],
                                                           total_allocation)
    # actual_allocation = [166, 166, 166, 166, 166, 166]
    print(f"Actual allocation: {actual_allocation}")
    rewards = get_rewards_from_fragments(fifth_arm, actual_allocation)

    print(f"Rewards: {rewards}")
    print(f"Expected exact value: {expectation_values[which_arm]}")
    print(f"Difference: {rewards - expectation_values[which_arm]}")
    print(f"Relative error: {abs(rewards - expectation_values[which_arm]) / abs(expectation_values[which_arm]) * 100:.2f}%")


