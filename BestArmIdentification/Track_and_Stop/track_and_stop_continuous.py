import numpy as np
from scipy.stats import gaussian_kde
from scipy.integrate import quad
from scipy.optimize import minimize


def kl_continuous(p_kde, q_kde, x_min, x_max):
    """Compute KL divergence between two continuous distributions using KDE."""
    def integrand(x):
        p_val = p_kde(x)
        q_val = q_kde(x)
        # Avoid log(0) by using small epsilon
        eps = 1e-10
        if p_val < eps or q_val < eps:
            return 0
        return p_val * np.log(p_val / q_val)

    # Integrate over the domain
    result, _ = quad(integrand, x_min, x_max)
    return result


def compute_optimal_allocation_continuous(all_samples):
    """
    all_samples: list of 1D numpy arrays, one per arm,
                 each array contains observed values (continuous)
    """
    K = len(all_samples)

    # Compute KDEs and means for each arm
    kdes = []
    emp_means = []
    x_min = float('inf')
    x_max = float('-inf')

    for samples in all_samples:
        kde = gaussian_kde(samples)
        kdes.append(kde)
        emp_means.append(np.mean(samples))
        x_min = min(x_min, np.min(samples))
        x_max = max(x_max, np.max(samples))

    emp_means = np.array(emp_means)
    best = np.argmax(emp_means)

    def objective(w):
        vals = []
        for i in range(K):
            if i == best:
                continue
            # KL(P_best || P_i)
            d1 = kl_continuous(kdes[best], kdes[i], x_min, x_max)
            # KL(P_i || P_best)
            d2 = kl_continuous(kdes[i], kdes[best], x_min, x_max)
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


def glr_continuous(kdes, counts, best_arm, x_min, x_max):
    """
    Compute GLR statistic for continuous distributions using KDE.
    """
    K = len(kdes)
    best_glr = float('inf')

    for a in range(K):
        if a == best_arm:
            continue

        def objective(x):
            # x is a vector of K-1 parameters for KDE bandwidths
            # (excluding the best arm which we keep fixed)
            total_kl = 0
            for i in range(K):
                if i == best_arm:
                    # Use original KDE for best arm
                    p_kde = kdes[i]
                else:
                    # Create new KDE with modified bandwidth
                    idx = i if i < best_arm else i - 1
                    new_kde = gaussian_kde(kdes[i].dataset, bw_method=x[idx])
                    p_kde = new_kde

                # Compute KL divergence
                kl = kl_continuous(p_kde, kdes[i], x_min, x_max)
                total_kl += counts[i] * kl
            return total_kl

        # Initial bandwidths
        x0 = np.ones(K-1) * 0.1
        bounds = [(0.01, 1.0) for _ in range(K-1)]

        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        if res.success:
            best_glr = min(best_glr, res.fun)

    return best_glr


def track_and_stop_continuous(arms, delta, max_rounds=100000, seed=None):
    np.random.seed(seed)
    K = len(arms)
    counts = np.zeros(K, dtype=int)
    sample_lists = [np.array([], dtype=float) for _ in range(K)]
    t = 0
    initial_allocation = 10

    # Initial sampling
    for i in range(K):
        samples = np.array([arms[i]() for _ in range(initial_allocation)])
        sample_lists[i] = np.concatenate([sample_lists[i], samples])
        counts[i] += initial_allocation
        t += initial_allocation

    # Compute initial domain bounds
    x_min = float('inf')
    x_max = float('-inf')
    for samples in sample_lists:
        x_min = min(x_min, np.min(samples))
        x_max = max(x_max, np.max(samples))

    while t < max_rounds:
        # Compute KDEs for each arm
        kdes = [gaussian_kde(samples) for samples in sample_lists]

        # Compute empirical means
        emp_means = [np.mean(samples) for samples in sample_lists]
        best_arm = np.argmax(emp_means)

        if t % 2 == 0:
            print(f"\nRound {t}:")
            print(f"Empirical Means: {[f'{m:.3f}' for m in emp_means]}")
            print(f"Current best arm: {best_arm}")
            print(f"Sample counts: {counts}")

        try:
            allocation = compute_optimal_allocation_continuous(sample_lists)
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

        # Update domain bounds if needed
        x_min = min(x_min, sample)
        x_max = max(x_max, sample)

        # Compute stopping condition
        threshold = np.log((t + 1) * (K - 1) / delta) + np.log(t)

        try:
            glr = glr_continuous(kdes, counts, best_arm, x_min, x_max)
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

    return best_arm, kdes, counts, t


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
    arms = [lambda dist=dist: np.random.choice(len(dist), p=dist) for dist in
            arm_distributions]
    # Compute true expected value of each arm
    expected_values = [np.sum(dist * np.arange(n_states)) for dist in
                       arm_distributions]
    expected_best_arm = int(np.argmax(expected_values))

    best_arm, kdes, counts, t = track_and_stop_continuous(arms, delta=0.05, seed=42)

    print(f"Identified best arm: {best_arm}")
    print(f"Expected best arm (true): {expected_best_arm}")
    print(f"Expected values: {np.round(expected_values, 3)}")
    print(f"Number of pulls: {counts}")
    print(f"Total rounds: {t}")
