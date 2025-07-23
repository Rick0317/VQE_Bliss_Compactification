import numpy as np
from best_arm_identification import best_arm_successive_elimination

if __name__ == '__main__':
    # Define 3 arms with different means
    means = [0.1, 0.5, 0.3, 0.7, 0.1, 0.2, 0.91, 0.6]
    # means = [0.5, 0.6, 0.7]
    arms = [lambda mu=mu: np.random.normal(abs(mu), 0.01) for mu in means]
    np.random.seed(42)
    best_arm = best_arm_successive_elimination(arms, delta=0.05)
    print(f"Identified best arm: {best_arm}, true best arm: {np.argmax(abs(np.array(means)))}")

