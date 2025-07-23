import numpy as np
import math

def best_arm_successive_elimination(arms, delta=0.05, samples_per_round=1, max_rounds=1000000):
    """
    SuccessiveElimination algorithm that corresponds to Algorithm 3 in
    "Action Elimination and Stopping Conditions for the Multi-Armed bandit and
    Reinforcement Learning Problems by E. Even-Dar, S. Mannor, Y. Mansour"
    :param arms:
    :param delta:
    :param samples_per_round:
    :param max_rounds:
    :return:
    """
    K = len(arms)
    n = samples_per_round
    active_arms = list(range(K))
    estimates = np.zeros(K)
    pulls = np.zeros(K)
    rounds = 0

    while len(active_arms) > 1 and rounds < max_rounds:
        rounds += 1
        for i in active_arms:
            rewards = [arms[i]() for _ in range(n)]
            pulls[i] += n
            estimates[i] += sum(rewards)

        means = estimates / np.maximum(pulls, 1)
        max_mean = max(abs(means[active_arms]))

        # radius = np.sqrt(2 * np.log(4 * len(active_arms) * pulls.max() / delta) / pulls)

        radius = np.sqrt(np.log(4 * len(active_arms) * (pulls ** 2) / delta) / pulls)

        new_active_arms = []
        for i in active_arms:
            if abs(means[i]) + radius[i] >= max_mean - radius[i]:
                new_active_arms.append(i)
        active_arms = new_active_arms

    # If more than one arm remains, pick the one with the highest empirical mean
    means = estimates / np.maximum(pulls, 1)
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    print(f"Total rounds: {rounds}")
    return best_arm



def get_rewards_from_fragments(fragments, allocation):
    """
    Given the fragments, allocate samplings to each distribution based on
    the variance of each expectation value of the fragments.

    :param fragments: A list of probability distributions that corresponds to
    the decomposition of the operator.
    :param allocation: List of allocations for each fragment
    :return: The estimated expectation value of the sum of all fragments
    """
    rewards = 0

    for index, frag in enumerate(fragments):
        if allocation[index] != 0:
            each_reward = [frag() for _ in range(allocation[index])]

            # Estimate <F1>
            rewards += sum(each_reward) / allocation[index]

    return rewards



def integer_allocations(probabilities, total_allocation):
    """
    Allocate integers summing to total_allocation, proportionally to probabilities.
    """
    raw_allocations = [p * total_allocation for p in probabilities]
    int_parts = [math.floor(x) for x in raw_allocations]
    remainders = [x - int_part for x, int_part in zip(raw_allocations, int_parts)]

    remainder_total = total_allocation - sum(int_parts)  # Use exact int difference

    # Distribute the remaining units to those with largest remainders
    indices_by_remainder = sorted(range(len(remainders)), key=lambda i: -remainders[i])
    for i in indices_by_remainder[:remainder_total]:
        int_parts[i] += 1

    return int_parts


def integer_allocations_one_guaranteed(probabilities, total_allocation):
    """
    Allocate integers summing to total_allocation, proportionally to probabilities,
    ensuring at least one sample is assigned to each index.
    """
    n = len(probabilities)
    if total_allocation < n:
        raise ValueError("total_allocation must be at least equal to the number of indices")

    # Step 1: Reserve 1 allocation for each index
    allocation = [1] * n
    remaining_allocation = total_allocation - n

    # Step 2: Compute proportional allocation for the remaining
    raw_allocations = [p * remaining_allocation for p in probabilities]
    int_parts = [math.floor(x) for x in raw_allocations]
    remainders = [x - int_part for x, int_part in zip(raw_allocations, int_parts)]

    # Add the floor part to the base allocation of 1
    for i in range(n):
        allocation[i] += int_parts[i]

    # Step 3: Distribute remaining units to highest remainders
    units_to_distribute = remaining_allocation - sum(int_parts)
    indices_by_remainder = sorted(range(n), key=lambda i: -remainders[i])
    for i in indices_by_remainder[:units_to_distribute]:
        allocation[i] += 1

    return allocation

def successive_elimination_var_considered(arms, allocations, delta=0.05, samples_per_round=1, max_rounds=100000):
    """
    SuccessiveElimination algorithm that takes into account of fragmentation

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
    total_allocation_history = np.zeros(K)
    rounds = 0

    while len(active_arms) > 1 and rounds < max_rounds:
        rounds += 1
        for i in active_arms:
            if rounds == 1:
                total_allocation = 3000000
                total_allocation_history[i] += total_allocation
                fragments = arms[i]

                # uniform_distribution = [1 / len(fragments) for _ in range(len(fragments))]
                actual_allocation = integer_allocations_one_guaranteed(
                    allocations[i], total_allocation)
                # actual_allocation = integer_allocations_one_guaranteed(
                #     uniform_distribution, total_allocation)
                rewards = get_rewards_from_fragments(fragments,
                                                     actual_allocation)
                pulls[i] += 30000
                estimates[i] += rewards
            else:
                total_allocation = 10000
                total_allocation_history[i] += total_allocation
                fragments = arms[i]
                # uniform_distribution = [1 / len(fragments) for _ in
                #                         range(5000len(fragments))]
                actual_allocation = integer_allocations_one_guaranteed(allocations[i], total_allocation)
                # actual_allocation = integer_allocations_one_guaranteed(
                #     uniform_distribution, total_allocation)
                # actual_allocation = [166, 166, 166, 166, 166, 166]
                rewards = get_rewards_from_fragments(fragments, actual_allocation)
                pulls[i] += 5000
                estimates[i] = (estimates[i] * total_allocation_history[i] + rewards * total_allocation) / (total_allocation_history[i] + total_allocation)

        # means = estimates / np.maximum(pulls, 1)
        means = estimates
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
    return best_arm, sum(total_allocation_history)



if __name__ == '__main__':
    integer_allocations()
