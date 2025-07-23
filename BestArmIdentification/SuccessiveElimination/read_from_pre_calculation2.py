import numpy as np
import json
from collections import defaultdict
from best_arm_identification import best_arm_successive_elimination, successive_elimination_var_considered
# from parallel_successive_var_elimination import successive_elimination_var_considered


if __name__ == '__main__':
    result_store = []
    mol= '_h4st2'
    with open(f"bai_results_7_9{mol}.txt", "w") as f:
        for trial_idx in range(20):
            # Load data from JSON files instead of npz
            with open(f"eigen_info_h4st2.json", "r") as eigen_file:
                eigen_info = json.load(eigen_file)

            with open(f"allocation_info_h4st2.json", "r") as alloc_file:
                allocation_info = json.load(alloc_file)

            print(f"Loaded {len(eigen_info)} eigen entries and {len(allocation_info)} allocation entries")

            # Step 2: Group fragments by generator_idx
            fragments_by_generator = defaultdict(list)
            expectation_values_dict = defaultdict(float)
            for entry in eigen_info:
                gen_idx = entry['generator_idx']
                eigenvalues = np.array(entry['eigenvalues'])  # Convert back to numpy array
                probabilities = np.array(entry['probabilities'])  # Convert back to numpy array

                mean = np.sum(eigenvalues * probabilities)
                expectation_values_dict[gen_idx] += mean

                fragments_by_generator[gen_idx].append(
                    lambda eigv=eigenvalues, prob=probabilities: np.random.choice(eigv, p=prob)
                )

            # Step 3: Rebuild reward_fns and allocations
            reward_fns = []
            allocations = []
            expectation_values = []

            # Keep generator_idx order consistent with allocation_info
            for alloc_entry in allocation_info:
                gen_idx = alloc_entry['generator_idx']
                allocation = np.array(alloc_entry['allocation'])  # Convert back to numpy array

                reward_fns.append(fragments_by_generator[gen_idx])
                allocations.append(allocation)
                expectation_values.append(expectation_values_dict[gen_idx])
            print(len(reward_fns[0]))
            print(
                f"Expectation values <[H, G]> but with selected fragments: {expectation_values}")
            magnitudes = np.abs(expectation_values)
            max_magnitude = np.max(magnitudes)
            max_indices = np.where(magnitudes == max_magnitude)[0]
            # sample_test_results = sample_test(reward_fns[-1], allocations[-1])
            # print(f"Test results: {sample_test_results}")
            # exit()
            best_arm, total_samples = successive_elimination_var_considered(reward_fns, allocations,
                                                             max_rounds=1000,
                                                             delta=0.05)

            # print(f"Identified best arm: {best_arm}")
            print(f"Expected best: {max_indices}")
            match = best_arm == max_indices[0]
            print(f"Does match: {match}")

            result_store.append(match)

            f.write(f"Trial {trial_idx + 1}:\n")
            f.write(f"  Max indices: {max_indices.tolist()}\n")
            f.write(f"  Best arm: {best_arm}\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Match: {match}\n\n")

    print(result_store)

    # with open("gradient_analysis_results_lih.txt", "a") as f:
    #     f.write(f"Identified best arm: {best_arm}\n")
    #     f.write(f"Expected best (max magnitude indices): {max_indices}\n")
    #
    #     f.write("-" * 40 + "\n")  # Separator line
