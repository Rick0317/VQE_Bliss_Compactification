import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def kl_divergence(p, q):
    # KL divergence for Bernoulli distributions
    p = np.clip(p, 1e-12, 1 - 1e-12)
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def kl_bernoulli(p, q):
    """KL divergence between two Bernoulli distributions."""
    eps = 1e-12  # avoid log(0)
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def compute_optimal_allocation(emp_means):
    K = len(emp_means)
    best = np.argmax(emp_means)

    def objective(w):
        # Max of inverse information across all suboptimal arms
        vals = []
        for i in range(K):
            if i == best:
                continue
            d1 = kl_bernoulli(emp_means[best], emp_means[i])
            d2 = kl_bernoulli(emp_means[i], emp_means[best])
            info = w[best] * d1 + w[i] * d2
            vals.append(1 / info)
        return max(vals)

    # Start from uniform allocation
    w0 = np.ones(K) / K
    bounds = [(1e-6, 1.0) for _ in range(K)]
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    res = minimize(objective, w0, method='SLSQP', bounds=bounds,
                   constraints=cons)

    if not res.success:
        raise RuntimeError("Optimization failed: " + res.message)

    return res.x


def generalized_likelihood_ratio(emp_means, counts):
    # For Bernoulli arms, use the GLR statistic
    K = len(emp_means)
    best = np.argmax(emp_means)
    glr = np.inf
    for i in range(K):
        if i == best:
            continue
        # Test H0: mu_best <= mu_i vs H1: mu_best > mu_i
        mu1 = emp_means[best]
        mu2 = emp_means[i]
        n1 = counts[best]
        n2 = counts[i]
        # GLR for Bernoulli
        p = (mu1 * n1 + mu2 * n2) / (n1 + n2)
        stat = n1 * kl_divergence(mu1, p) + n2 * kl_divergence(mu2, p)
        glr = min(glr, stat)
    return glr


def track_and_stop(arm, delta, max_rounds=10000, seed=None):
    np.random.seed(seed)
    K = len(arm)
    counts = np.zeros(K, dtype=int)
    rewards = np.zeros(K)
    emp_means = np.zeros(K)
    t = 0
    # Initial pull of each arm
    for i in range(K):
        reward = arm[i]()
        counts[i] += 1
        rewards[i] += reward
        emp_means[i] = rewards[i] / counts[i]
        t += 1
    while t < max_rounds:
        # Compute optimal allocation
        allocation = compute_optimal_allocation(emp_means)
        # Find arm most under-sampled compared to allocation
        proportions = counts / counts.sum()
        under_sampled = np.argmin(proportions / (allocation + 1e-12))
        # Pull the selected arm
        reward = arm[under_sampled]()
        counts[under_sampled] += 1
        rewards[under_sampled] += reward
        emp_means[under_sampled] = rewards[under_sampled] / counts[under_sampled]
        t += 1
        # Stopping rule: GLR > threshold
        threshold = np.log((t + 1) * (K - 1) / delta)
        glr = generalized_likelihood_ratio(emp_means, counts)
        if glr > threshold:
            break
    best_arm = np.argmax(emp_means)
    return best_arm, emp_means, counts, t


if __name__ == "__main__":
    # Example: 3 arms with means 0.5, 0.6, 0.7
    arm_means = [0.5, 0.6, 0.7, 0.8, 0.9, 0.3, 0.2]
    delta = 0.05
    arms = [lambda mu=mu: np.random.normal(abs(mu), 0.01) for mu in arm_means]

    best_arm, emp_means, counts, t = track_and_stop(arms, delta, seed=42)
    print(f"Identified best arm: {best_arm}")
    print(f"Empirical means: {emp_means}")
    print(f"Number of pulls: {counts}")
    print(f"Total rounds: {t}")
