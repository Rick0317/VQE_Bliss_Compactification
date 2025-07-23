import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from typing import List, Tuple, Optional
import argparse


class TopTwoThompsonSampling:
    """
    Top-Two Thompson Sampling algorithm for best-arm identification.

    This algorithm maintains Beta posterior distributions for each arm and selects
    the two arms with the highest Thompson samples at each round.
    """

    def __init__(self, num_arms: int, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        """
        Initialize the Top-Two Thompson Sampling algorithm.

        Args:
            num_arms: Number of arms in the bandit
            alpha_prior: Prior alpha parameter for Beta distribution
            beta_prior: Prior beta parameter for Beta distribution
        """
        self.num_arms = num_arms
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        # Initialize posterior parameters (Beta distribution parameters)
        self.alpha = np.full(num_arms, alpha_prior)
        self.beta = np.full(num_arms, beta_prior)

        # Keep track of statistics
        self.total_pulls = np.zeros(num_arms)
        self.total_rewards = np.zeros(num_arms)
        self.round_count = 0
        self.history = []

    def sample_from_posterior(self) -> np.ndarray:
        """
        Sample from the posterior distribution of each arm.

        Returns:
            Array of samples from each arm's posterior distribution
        """
        samples = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            samples[i] = np.random.beta(self.alpha[i], self.beta[i])
        return samples

    def select_arms(self) -> Tuple[int, int]:
        """
        Select the two arms with the highest Thompson samples.

        Returns:
            Tuple of (first_arm, second_arm) indices
        """
        samples = self.sample_from_posterior()

        # Find the two arms with highest samples
        sorted_indices = np.argsort(samples)
        first_arm = sorted_indices[-1]   # Highest sample
        second_arm = sorted_indices[-2]  # Second highest sample

        return first_arm, second_arm

    def pull_arm(self, arm: int) -> int:
        """
        Simulate pulling an arm and return the reward.
        This is a placeholder - in practice, you'd replace this with actual arm pulling.

        Args:
            arm: Index of the arm to pull

        Returns:
            Reward (0 or 1 for Bernoulli rewards)
        """
        # For demonstration, use some fixed true means
        true_means = np.array([0.3, 0.5, 0.7, 0.4, 0.6])[:self.num_arms]
        if len(true_means) < self.num_arms:
            # If we have more arms than predefined means, generate random means
            additional_means = np.random.uniform(0.2, 0.8, self.num_arms - len(true_means))
            true_means = np.concatenate([true_means, additional_means])

        return np.random.binomial(1, true_means[arm])

    def update_posterior(self, arm: int, reward: float):
        """
        Update the posterior distribution for the pulled arm.

        Args:
            arm: Index of the pulled arm
            reward: Observed reward (0 or 1)
        """
        self.total_pulls[arm] += 1
        self.total_rewards[arm] += reward

        # Update Beta distribution parameters
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)

        # Record history
        self.history.append({
            'round': self.round_count,
            'arm': arm,
            'reward': reward,
            'total_pulls': self.total_pulls.copy(),
            'total_rewards': self.total_rewards.copy()
        })

        self.round_count += 1

    def run_round(self) -> Tuple[int, float]:
        """
        Execute one round of the algorithm.

        Returns:
            Tuple of (selected_arm, reward)
        """
        # Select the two best arms
        first_arm, second_arm = self.select_arms()

        # Using Bernoulli distribution to select which one to sample
        selected_arm = first_arm

        # Pull the selected arm and observe reward
        reward = self.pull_arm(selected_arm)

        # Update posterior
        self.update_posterior(selected_arm, reward)

        return selected_arm, reward

    def get_best_arm_recommendation(self) -> int:
        """
        Get the current best arm recommendation based on posterior means.

        Returns:
            Index of the arm with highest posterior mean
        """
        posterior_means = self.alpha / (self.alpha + self.beta)
        return np.argmax(posterior_means)

    def get_posterior_stats(self) -> dict:
        """
        Get current posterior statistics for all arms.

        Returns:
            Dictionary containing posterior means, variances, and confidence intervals
        """
        posterior_means = self.alpha / (self.alpha + self.beta)
        posterior_vars = (self.alpha * self.beta) / ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))

        # 95% confidence intervals
        ci_lower = np.zeros(self.num_arms)
        ci_upper = np.zeros(self.num_arms)

        for i in range(self.num_arms):
            ci_lower[i] = beta.ppf(0.025, self.alpha[i], self.beta[i])
            ci_upper[i] = beta.ppf(0.975, self.alpha[i], self.beta[i])

        return {
            'means': posterior_means,
            'variances': posterior_vars,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'alpha': self.alpha.copy(),
            'beta': self.beta.copy()
        }

    def run_experiment(self, num_rounds: int, verbose: bool = False) -> dict:
        """
        Run the complete experiment for a specified number of rounds.

        Args:
            num_rounds: Number of rounds to run
            verbose: Whether to print progress

        Returns:
            Dictionary containing experiment results
        """
        results = {
            'rounds': [],
            'selected_arms': [],
            'rewards': [],
            'cumulative_rewards': []
        }

        cumulative_reward = 0

        for round_num in range(num_rounds):
            selected_arm, reward = self.run_round()
            cumulative_reward += reward

            results['rounds'].append(round_num)
            results['selected_arms'].append(selected_arm)
            results['rewards'].append(reward)
            results['cumulative_rewards'].append(cumulative_reward)

            if verbose and (round_num + 1) % 100 == 0:
                best_arm = self.get_best_arm_recommendation()
                stats = self.get_posterior_stats()
                means_str = np.array2string(stats['means'], precision=3, separator=', ')
                print(f"Round {round_num + 1}: Best arm = {best_arm}, "
                      f"Posterior means = {means_str}")

        # Add final statistics
        results['final_best_arm'] = self.get_best_arm_recommendation()
        results['final_stats'] = self.get_posterior_stats()
        results['total_pulls'] = self.total_pulls.copy()
        results['total_rewards'] = self.total_rewards.copy()

        return results

    def plot_results(self, results: dict):
        """
        Plot the results of the experiment.

        Args:
            results: Results dictionary from run_experiment
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot arm selection frequency
        arm_counts = np.bincount(results['selected_arms'], minlength=self.num_arms)
        ax1.bar(range(self.num_arms), arm_counts)
        ax1.set_xlabel('Arm')
        ax1.set_ylabel('Number of Pulls')
        ax1.set_title('Arm Selection Frequency')
        ax1.set_xticks(range(self.num_arms))

        # Plot cumulative rewards
        ax2.plot(results['rounds'], results['cumulative_rewards'])
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title('Cumulative Rewards Over Time')
        ax2.grid(True)

        # Plot posterior means with confidence intervals
        stats = results['final_stats']
        ax3.bar(range(self.num_arms), stats['means'],
                yerr=[stats['means'] - stats['ci_lower'],
                      stats['ci_upper'] - stats['means']],
                capsize=5)
        ax3.set_xlabel('Arm')
        ax3.set_ylabel('Posterior Mean')
        ax3.set_title('Final Posterior Means with 95% CI')
        ax3.set_xticks(range(self.num_arms))

        # Plot arm selection over time (moving average)
        window_size = min(50, len(results['rounds']) // 10)
        if window_size > 0:
            for arm in range(self.num_arms):
                arm_selections = np.array([1 if x == arm else 0 for x in results['selected_arms']])
                if len(arm_selections) >= window_size:
                    moving_avg = np.convolve(arm_selections, np.ones(window_size)/window_size, mode='valid')
                    ax4.plot(range(window_size-1, len(arm_selections)), moving_avg,
                            label=f'Arm {arm}', alpha=0.7)
            ax4.set_xlabel('Round')
            ax4.set_ylabel('Selection Probability')
            ax4.set_title(f'Arm Selection Probability (Moving Average, Window={window_size})')
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    """
    Example usage of the Top-Two Thompson Sampling algorithm.
    """
    parser = argparse.ArgumentParser(description='Top-Two Thompson Sampling for Best-Arm Identification')
    parser.add_argument('--num_arms', type=int, default=5, help='Number of arms')
    parser.add_argument('--num_rounds', type=int, default=1000, help='Number of rounds to run')
    parser.add_argument('--alpha_prior', type=float, default=1.0, help='Prior alpha parameter')
    parser.add_argument('--beta_prior', type=float, default=1.0, help='Prior beta parameter')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    parser.add_argument('--plot', action='store_true', help='Plot results')

    args = parser.parse_args()

    # Initialize the algorithm
    ttts = TopTwoThompsonSampling(
        num_arms=args.num_arms,
        alpha_prior=args.alpha_prior,
        beta_prior=args.beta_prior
    )

    print(f"Running Top-Two Thompson Sampling with {args.num_arms} arms for {args.num_rounds} rounds...")

    # Run the experiment
    results = ttts.run_experiment(args.num_rounds, verbose=args.verbose)

    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Best arm recommendation: {results['final_best_arm']}")
    print(f"Total pulls per arm: {results['total_pulls']}")
    print(f"Total rewards per arm: {results['total_rewards']}")
    print(f"Empirical means: {results['total_rewards'] / np.maximum(results['total_pulls'], 1)}")

    stats = results['final_stats']
    means_str = np.array2string(stats['means'], precision=3, separator=', ')
    print(f"Posterior means: {means_str}")
    print(f"95% Confidence intervals:")
    for i in range(args.num_arms):
        print(f"  Arm {i}: [{stats['ci_lower'][i]:.3f}, {stats['ci_upper'][i]:.3f}]")

    # Plot results if requested
    if args.plot:
        ttts.plot_results(results)


if __name__ == "__main__":
    main()
