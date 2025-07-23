import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def empirical_distribution(samples, support, alpha=1e-3):
    """Return the empirical distribution over the given support."""
    counts = np.array([np.sum(samples == s) for s in support], dtype=np.float64)
    counts += alpha
    probs = counts / counts.sum()
    return probs


def kl_discrete(p, q, eps=1e-12):
    p = np.array(p) + eps
    q = np.array(q) + eps
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))


def compute_optimal_allocation(all_samples):
    """
    all_samples: list of 1D numpy arrays, one per arm,
                 each array contains observed values (real values, not just [0,1])
    """
    K = len(all_samples)

    # union of all observable values
    combined_support = sorted(set(np.concatenate(all_samples)))

    # 2. Compute empirical distributions and means
    emp_dists = []
    emp_means = []
    for samples in all_samples:
        dist = empirical_distribution(samples, combined_support)
        emp_dists.append(dist)
        emp_means.append(np.mean(samples))
    emp_means = np.array(emp_means)

    best = np.argmax(emp_means)

    def objective(w):
        vals = []
        for i in range(K):
            if i == best:
                continue
            # KL(P_best || P_i)
            d1 = kl_discrete(emp_dists[best], emp_dists[i])
            # KL(P_i || P_best)
            d2 = kl_discrete(emp_dists[i], emp_dists[best])
            # Info: How different the probability distribution of P_best is from P_i
            info = w[best] * d1 + w[i] * d2
            if info <= 0 or np.isnan(info) or np.isinf(info):
                # Penalize invalid or zero info to keep optimizer in safe region
                return 1e6

            # How similar the distributions are.
            vals.append(1 / info)

        # We want sampling allocations that will distinguish similar distributions
        return max(vals)

    # Initial Allocation guess
    w0 = np.ones(K) / K
    bounds = [(1e-6, 1.0) for _ in range(K)]
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    res = minimize(objective, w0, method='SLSQP', bounds=bounds,
                   constraints=cons)

    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)

    return res.x


def glr_true(emp_dists, counts, b):
    """
    Compute GLR(b) = min_{a ≠ b} inf over {P_i} s.t. mu_a >= mu_i ∀i ≠ a of ∑ N_i * KL(emp_i || P_i)
    """
    K = len(emp_dists)
    n_states = len(emp_dists[0])
    support = np.arange(n_states)
    best_glr = float('inf')

    def unpack(x):
        return [x[i * n_states:(i + 1) * n_states] for i in range(K)]

    def likelihood(x):
        dists = unpack(x)
        return sum(counts[i] * kl_discrete(emp_dists[i], dists[i]) for i in range(K))

    def dist_constraints(x):
        sum_to_1 = [{'type': 'eq', 'fun': lambda x, i=i: np.sum(
            x[i * n_states:(i + 1) * n_states]) - 1}
                    for i in range(K)]
        return sum_to_1


    def mean_constraints(x, best_idx):
        # Enforce mu_best >= mu_i for all i ≠ best_idx
        def constraint(x, i):
            dists = unpack(x)
            mu_best = np.dot(support, dists[best_idx])
            mu_i = np.dot(support, dists[i])
            return mu_best - mu_i
        return [{'type': 'ineq', 'fun': lambda x, i=i: constraint(x, i)} for i in range(K) if i != best_idx]

    # Initialization: uniform distributions
    x0 = np.concatenate([np.ones(n_states) / n_states for _ in range(K)])
    bounds = [(1e-9, 1.0)] * (K * n_states)

    res_H0 = minimize(
        likelihood,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=dist_constraints(x0) + mean_constraints(x0, b),
        options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
    )

    if not res_H0.success:
        raise RuntimeError(f"H0 optimization failed: {res_H0.message}")
    ll_null = res_H0.fun

    ll_alt = float('inf')
    for a in range(K):
        if a == b:
            continue

        res_H1 = minimize(
            likelihood,
            x0,
            bounds=bounds,
            constraints=dist_constraints(x0) + mean_constraints(x0, a),
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-6, 'disp': False}
        )

        if res_H1.success:
            ll_alt = min(ll_alt, res_H1.fun)
        else:
            print(f"Warning: optimization failed for a = {a}: {res.message}")

    return ll_alt - ll_null


def track_and_stop_discrete(arms, delta, max_rounds=100000, seed=None):
    np.random.seed(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    sample_lists = [np.array([], dtype=float) for _ in range(K)]
    t = 0
    initial_allocation = 10  # Reduced from 50 to allow more adaptive sampling

    # Initial sampling
    for i in range(K):
        samples = np.array([arms[i]() for _ in range(initial_allocation)])
        sample_lists[i] = np.concatenate([sample_lists[i], samples])
        counts[i] += initial_allocation
        t += initial_allocation

    while t < max_rounds:
        # Build combined support from all samples seen so far
        combined_support = np.array(sorted(set(np.concatenate(sample_lists))))

        # Compute empirical distributions over combined support
        emp_dists = []
        for i in range(K):
            emp_dists.append(
                empirical_distribution(sample_lists[i], combined_support))

        # Compute empirical means over combined support
        emp_means = [np.sum(d * combined_support) for d in emp_dists]
        best_arm = np.argmax(abs(np.array(emp_means)))

        if t % 2 == 0:  # Print status every 100 rounds
            print(f"\nRound {t}:")
            print(f"Empirical Means: {[f'{m:.3f}' for m in emp_means]}")
            print(f"Current best arm: {best_arm}")
            print(f"Sample counts: {counts}")

        # Compute optimal allocation based on empirical distributions
        try:
            allocation = compute_optimal_allocation(sample_lists)
            if t % 2 == 0:
                print(f"Optimal allocation: {[f'{a:.3f}' for a in allocation]}")
        except Exception as e:
            print(f"Warning: Allocation computation failed: {e}")
            # Fallback to uniform allocation if optimization fails
            allocation = np.ones(K) / K

        # Modified sampling strategy
        proportions = counts / counts.sum()
        # Find arms that are under-sampled relative to allocation
        ratios = proportions / (allocation + 1e-12)
        # Sample from the most under-sampled arm with probability 0.7
        # Otherwise sample uniformly from all arms
        if np.random.random() < 0.7:
            under_sampled = np.argmin(ratios)
        else:
            under_sampled = np.random.randint(K)

        # Sample the under sampled one.
        sample = arms[under_sampled]()
        sample_lists[under_sampled] = np.append(sample_lists[under_sampled], sample)
        counts[under_sampled] += 1
        t += 1

        # Compute stopping condition threshold
        threshold = np.log(2 * t * (K - 1) / delta)

        # Compute GLR statistic based on empirical means and counts
        try:
            glr = glr_true(emp_dists, counts, best_arm)
            print(f"GLR: {glr}")
            if t % 2 == 0:
                print(f"GLR statistic: {glr:.3f}")
                print(f"Threshold: {threshold:.3f}")
        except Exception as e:
            print(f"Warning: GLR computation failed: {e}")
            continue  # Skip stopping check if GLR computation fails

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
    expected_values = [np.sum(dist * np.arange(n_states)) for dist in
                       arm_distributions]
    expected_best_arm = int(np.argmax(expected_values))

    best_arm, emp_dists, counts, t = track_and_stop_discrete(arms,
                                                             delta=0.05,
                                                             seed=42)

    print(f"Identified best arm: {best_arm}")
    print(f"Expected best arm (true): {expected_best_arm}")
    print(f"Expected values: {np.round(expected_values, 3)}")
    print(f"Empirical distributions:")
    for i, dist in enumerate(emp_dists):
        print(f"  Arm {i}: {dist.round(3)}")
    print(f"Number of pulls: {counts}")
    print(f"Total rounds: {t}")
