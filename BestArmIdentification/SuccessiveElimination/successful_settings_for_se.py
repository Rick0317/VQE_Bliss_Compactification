from best_arm_identification import integer_allocations_one_guaranteed, get_rewards_from_fragments
import numpy as np

def successive_elimination_var_considered_H4(arms, allocations, delta=0.05, samples_per_round=1, max_rounds=100000):
    """
    The following setting allowed the algorithm to find the best arm 5 in 86 rounds
    total of 860000 samples.

    :param arms: List of the list
    of fragments (probability distributions) of the operator
    :param allocations: List of the list of allocations of samples for the operator
    :param delta:
    :param samples_per_round:
    :param max_rounds:
    :return:
    """
    K = len(arms)
    active_arms = list(range(K))
    estimates = np.zeros(K)
    pulls = np.zeros(K)
    rounds = 0

    while len(active_arms) > 1 and rounds < max_rounds:
        rounds += 1
        for i in active_arms:
            total_allocation = 10000
            fragments = arms[i]
            actual_allocation = integer_allocations_one_guaranteed(allocations[i], total_allocation)
            # actual_allocation = [166, 166, 166, 166, 166, 166]
            rewards = get_rewards_from_fragments(fragments, actual_allocation)
            pulls[i] += 5000
            estimates[i] += rewards

        # means = estimates / np.maximum(pulls, 1)
        means = estimates / rounds
        print(f"Means estimates: {means}")
        print(f"Active arms: {active_arms}")
        max_mean = max(abs(means[active_arms]))

        radius = np.sqrt(
            np.log(0.001 * len(active_arms) * (pulls ** 2) / delta) / pulls)

        print(f"{rounds} / {max_rounds} / {len(active_arms)}, Radius: {radius}")

        new_active_arms = []
        for i in active_arms:
            if abs(means[i]) + radius[i] >= max_mean - radius[i]:
                new_active_arms.append(i)
        active_arms = new_active_arms


    # means = estimates / np.maximum(pulls, 1)
    means = estimates / rounds
    print(f"Active arms: {active_arms}")
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    print(f"Round: {rounds}, Mean Estimates: {means}")
    print(f"Pulls from each arm: {pulls}")
    return best_arm
