from best_arm_identification import integer_allocations, get_rewards_from_fragments


def sample_test(fragments, allocations):
    """
    By sampling many times, estimate the mean of the set of probability distributions
    :param fragments: A list of probability distributions that you can sample from.
    :return:
    """
    rounds = 0
    estimate = 0
    mean_estimate = 0
    while rounds < 10000:
        rounds += 1
        total_allocation = 1000
        # actual_allocation = integer_allocations(allocations, total_allocation)
        actual_allocation = [333, 333, 334]
        rewards = get_rewards_from_fragments(fragments, actual_allocation)
        estimate += rewards
        mean_estimate = estimate / rounds
        print(f"Intermediate mean estimate: {mean_estimate} at round {rounds}")

    return mean_estimate
