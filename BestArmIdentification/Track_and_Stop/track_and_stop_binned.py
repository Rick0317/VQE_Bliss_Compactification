import numpy as np
from scipy.optimize import minimize
import cvxpy as cp


def create_bins(samples, n_bins=50):
    """
    Create bins for discretizing continuous values.
    Uses equal-width binning over the range of all samples.

    bins is a list of numbers from the minimum possible value and the maximum
    possible values. It represents all the possible values the entire set of
    distributions can take.
    """
    all_samples = np.concatenate(samples)
    min_val = np.min(all_samples)
    max_val = np.max(all_samples)
    # Add small padding to avoid edge cases
    padding = (max_val - min_val) * 0.01
    bins = np.linspace(min_val - padding, max_val + padding, n_bins + 1)
    return bins


def empirical_distribution_binned(samples, bins, alpha=1e-3):
    """Return the empirical distribution over the given bins."""
    counts, _ = np.histogram(samples, bins=bins, density=False)
    counts = counts.astype(np.float64)
    counts += alpha
    probs = counts / counts.sum()
    return probs


def kl_discrete(p, q, eps=1e-12):
    """
    Compute KL divergence between two discrete distributions.
    Unless the two distributions have special forms, the KL divergence is
    asymmetric. KL(p || q) != KL(q || p) in general.
    """
    p = np.array(p) + eps
    q = np.array(q) + eps
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))


def kl_bernoulli(p, q):
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def compute_optimal_allocation_binned(all_samples, bins):
    """
    all_samples: list of 1D numpy arrays, one per arm
    bins: array of bin edges for discretization

    From the empirical distributions, find the one with largest mean
    then, Use KL divergence to find
    """
    K = len(all_samples)

    # Compute empirical distributions and means
    emp_dists = []
    emp_means = []
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for samples in all_samples:
        dist = empirical_distribution_binned(samples, bins)
        emp_dists.append(dist)
        # Compute mean using bin centers
        emp_means.append(np.sum(dist * bin_centers))

    emp_means = np.array(emp_means)
    best = np.argmax(abs(emp_means))

    def objective(w):
        vals = []
        for i in range(K):
            if i == best:
                continue
            # KL(P_best || P_i)
            d1 = kl_discrete(emp_dists[best], emp_dists[i])
            # KL(P_i || P_best)
            d2 = kl_discrete(emp_dists[i], emp_dists[best])
            info = w[best] * d1 + w[i] * d2
            if info <= 0 or np.isnan(info) or np.isinf(info):
                return 1e6
            vals.append(1 / info)
        return max(vals)

    w0 = np.ones(K) / K
    bounds = [(1e-6, 1.0) for _ in range(K)]
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    res = minimize(objective, w0, method='SLSQP', bounds=bounds,
                   constraints=cons)

    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)

    return res.x


def original_optimal_allocation(all_samples, bins):
    """
    The nested optimization which was proposed in the Track-and-Stop paper.
    :param all_samples:
    :param bins:
    :return:
    """

    K = len(all_samples)

    # Compute empirical distributions and means
    emp_dists = []
    emp_means = []
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for samples in all_samples:
        dist = empirical_distribution_binned(samples, bins)
        emp_dists.append(dist)
        # Compute mean using bin centers
        emp_means.append(np.sum(dist * bin_centers))

    emp_means = np.array(emp_means)
    best = np.argmax(abs(emp_means))

    def alt_constraints(lambda_):
        return lambda_[best] - max([lambda_[i] for i in range(K) if i != best])

    def inner_objective(lambda_, w, mu):
        return sum(w[a] * kl_bernoulli(mu[a], lambda_[a]) for a in range(len(mu)))

    def outer_objective(w, mu):
        def inner(lambda_):
            return inner_objective(lambda_, w, mu)

        # Initial guess: same as mu
        lambda0 = np.ones(K) / K

        # Alt constraint
        cons = {
            'type': 'ineq',
            'fun': lambda lambda_: -alt_constraints(lambda_)
        }

        res = minimize(inner, lambda0, constraints=cons,
                       bounds=[(1e-3, 1 - 1e-3)] * K)
        return res.fun

    w0 = np.ones(K) / K
    bounds = [(1e-6, 1.0)] * K
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    print(f"Empirical Means: {emp_means}")

    res = minimize(lambda w: -outer_objective(w, emp_means), w0, bounds=bounds,
                   constraints=cons)

    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)

    return res.x


def glr_binned(emp_dists, counts, best_arm, bin_centers):
    """
    Compute a simplified but conservative GLR statistic for binned distributions.
    """
    K = len(emp_dists)
    best_glr = float('inf')

    # Get the best arm's distribution and count
    best_dist = emp_dists[best_arm]
    N_best = counts[best_arm]

    for i in range(K):
        if i == best_arm:
            continue

        # Get current (i) arm's distribution and count
        curr_dist = emp_dists[i]
        N_i = counts[i]

        # mu_ab = (N_best / ) * best_dist + (N_i / (N_i + N_best)) * curr_dist

        # Compute KL divergence in both directions
        d1 = kl_discrete(best_dist, curr_dist)  # KL(best || i)
        d2 = kl_discrete(curr_dist, best_dist)  # KL(i || best)

        # Weight the KL divergences by sample counts
        # Use sqrt of counts to make the statistic grow more slowly
        weighted_kl = np.sqrt(N_best) * d1 + np.sqrt(N_i) * d2

        # Add a term that penalizes mean differences, but make it grow more slowly
        best_mean = np.sum(best_dist * bin_centers)
        curr_mean = np.sum(curr_dist * bin_centers)
        mean_diff = abs(best_mean - curr_mean)
        # Scale the mean difference penalty by the minimum of the two counts
        mean_diff_penalty = mean_diff * np.sqrt(min(N_best, N_i))

        # Combine KL divergence and mean difference
        glr_val = weighted_kl + mean_diff_penalty
        best_glr = min(best_glr, glr_val)

    return best_glr


def glr(emp_dists, counts, best_arm, common_support):
    K = len(emp_dists)
    best_glr = float('inf')

    # Get the best arm's distribution and count
    best_dist = emp_dists[best_arm]
    N_best = counts[best_arm]

    min_glr = float('inf')

    for i in range(K):
        if i == best_arm:
            continue
        curr_dist = emp_dists[i]
        N_i = counts[i]

        qa = cp.Variable(len(common_support))
        qb = cp.Variable(len(common_support))

        constraints = [
            cp.sum(qa) == 1,
            cp.sum(qb) == 1,
            qa >= 1e-10,
            qb >= 1e-10,
            common_support @ qa >= common_support @ qb
        ]

        objective = cp.Maximize(
            N_best * cp.sum(cp.multiply(best_dist, cp.log(qa))) +
            N_i * cp.sum(cp.multiply(curr_dist, cp.log(qb)))
        )

        prob = cp.Problem(objective, constraints)
        prob.solve()

        logL_constrained = N_best * np.sum(best_dist * np.log(best_dist) + N_i * np.sum(curr_dist * np.log(curr_dist)))

        glr_value = prob.value - logL_constrained

        min_glr = min(min_glr, glr_value)

    return min_glr


def track_and_stop_binned(arms, delta, n_bins=50, max_rounds=100000, seed=None):
    np.random.seed(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    sample_lists = [np.array([], dtype=float) for _ in range(K)]
    t = 0
    initial_allocation = 10
    min_samples_per_arm = 30  # Minimum samples required before considering stopping

    # Initial sampling
    for i in range(K):
        samples = np.array([arms[i]() for _ in range(initial_allocation)])
        sample_lists[i] = np.concatenate([sample_lists[i], samples])
        counts[i] += initial_allocation
        t += initial_allocation

    # Create initial bins
    bins = create_bins(sample_lists, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    while t < max_rounds:
        bins = create_bins(sample_lists, n_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Compute empirical distributions over bins
        emp_dists = []
        for i in range(K):
            emp_dists.append(empirical_distribution_binned(sample_lists[i], bins))

        # Compute empirical means using bin centers
        emp_means = [np.sum(d * bin_centers) for d in emp_dists]
        best_arm = np.argmax(abs(np.array(emp_means)))

        if t % 2 == 0:
            print(f"\nRound {t}:")
            print(f"Empirical Means: {[f'{m:.3f}' for m in emp_means]}")
            print(f"Current best arm: {best_arm}")
            print(f"Sample counts: {counts}")

        try:
            allocation = compute_optimal_allocation_binned(sample_lists, bins)
            if t % 2 == 0:
                print(f"Optimal allocation: {[f'{a:.3f}' for a in allocation]}")
        except Exception as e:
            print(f"Warning: Allocation computation failed: {e}")
            allocation = np.ones(K) / K

        # Sampling strategy
        proportions = counts / counts.sum()
        ratios = proportions / (allocation + 1e-12)
        if np.random.random() < 0.7:
            under_sampled = np.argmin(ratios)
        else:
            under_sampled = np.random.randint(K)

        # Sample and update
        sample = arms[under_sampled]()
        sample_lists[under_sampled] = np.append(sample_lists[under_sampled], sample)
        counts[under_sampled] += 1
        t += 1

        # Only check stopping condition if we have enough samples
        if np.min(counts) >= min_samples_per_arm:
            # Compute stopping condition with more conservative threshold
            threshold = np.log((t + 1) * (K - 1) / delta)

            try:
                glr = glr_binned(emp_dists, counts, best_arm, bin_centers)
                if t % 2 == 0:
                    print(f"GLR statistic: {glr:.3f}")
                    print(f"Threshold: {threshold:.3f}")
            except Exception as e:
                print(f"Warning: GLR computation failed: {e}")
                continue

            if glr > threshold:
                print(f"\nStopping at round {t}")
                print(f"Final empirical means: {[f'{m:.3f}' for m in emp_means]}")
                print(f"Final sample counts: {counts}")
                break

    return best_arm, emp_dists, counts, t


if __name__ == "__main__":
    n_states = 5
    arm_distributions = [
        np.array([0.1, 0.2, 0.3, 0.2, 0.2]),
        np.array([0.15, 0.2, 0.2, 0.25, 0.2]),
        np.array([0.05, 0.05, 0.8, 0.05, 0.05]),
        np.array([0.25, 0.25, 0.25, 0.15, 0.10]),
        np.array([0.05, 0.15, 0.1, 0.3, 0.4]),
        np.array([0.4, 0.3, 0.1, 0.1, 0.1]),
        np.array([0.2, 0.1, 0.1, 0.3, 0.3]),
        np.array([0.3, 0.1, 0.2, 0.2, 0.2]),
        np.array([0.1, 0.1, 0.1, 0.3, 0.4]),
        np.array([0.18, 0.22, 0.2, 0.2, 0.2])
    ]
    arms = [lambda dist=dist: np.random.choice(len(dist), p=dist) for dist in arm_distributions]
    # Compute true expected value of each arm
    expected_values = [np.sum(dist * np.arange(n_states)) for dist in arm_distributions]
    expected_best_arm = int(np.argmax(expected_values))

    best_arm, emp_dists, counts, t = track_and_stop_binned(arms, delta=0.05, n_bins=50, seed=42)

    print(f"Identified best arm: {best_arm}")
    print(f"Expected best arm (true): {expected_best_arm}")
    print(f"Expected values: {np.round(expected_values, 3)}")
    print(f"Number of pulls: {counts}")
    print(f"Total rounds: {t}")
