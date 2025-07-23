import numpy as np
import math
from joblib import Parallel, delayed
import time


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


def process_arm_sequential(i, arms, allocations, rounds, estimates, total_allocation_history):
    """
    Process a single arm for threading.

    :param i: Arm index
    :param arms: List of arms (fragments)
    :param allocations: List of allocations
    :param rounds: Current round number
    :param estimates: Current estimates array
    :param total_allocation_history: Current allocation history array
    :return: Tuple containing (i, rewards, total_allocation, pulls_increment)
    """
    if rounds == 1:
        print(f"Processing arm {i}")
        total_allocation = 1000000
        fragments = arms[i]
        actual_allocation = integer_allocations_one_guaranteed(
            allocations[i], total_allocation)
        rewards = get_rewards_from_fragments(fragments, actual_allocation)
        pulls_increment = 10000
        print(f"Finished processing arm {i}")
    else:
        total_allocation = 10000
        fragments = arms[i]
        actual_allocation = integer_allocations_one_guaranteed(allocations[i], total_allocation)
        rewards = get_rewards_from_fragments(fragments, actual_allocation)
        pulls_increment = 5000

    return i, rewards, total_allocation, pulls_increment


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

                # Use joblib for parallel processing of arms (handles pickling issues better)
        results = Parallel(n_jobs=-1)(
            delayed(process_arm_sequential)(i, arms, allocations, rounds, estimates, total_allocation_history)
            for i in active_arms
        )

        # Update estimates and pulls based on results
        for i, rewards, total_allocation, pulls_increment in results:
            total_allocation_history[i] += total_allocation
            if rounds == 1:
                estimates[i] += rewards
            else:
                estimates[i] = (estimates[i] * (total_allocation_history[i] - total_allocation) + rewards * total_allocation) / total_allocation_history[i]
            pulls[i] += pulls_increment

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

