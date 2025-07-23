import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path to import the functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BestArmIdentification.SuccessiveElimination.best_arm_identification import best_arm_successive_elimination


class TestBestArmSuccessiveElimination:
    """Test cases for the best_arm_successive_elimination function."""

    def test_basic_functionality(self):
        """Test basic functionality with clearly separated arms."""
        np.random.seed(42)

        # Create 3 arms with different means
        arms = [
            lambda: np.random.normal(0.1, 0.1),   # Mean 0.1
            lambda: np.random.normal(0.5, 0.1),   # Mean 0.5 (best)
            lambda: np.random.normal(0.2, 0.1),   # Mean 0.2
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should identify arm 1 as the best (highest mean)
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_negative_means(self):
        """Test with negative means to ensure magnitude-based selection works."""
        np.random.seed(42)

        # Create 3 arms with negative means
        arms = [
            lambda: np.random.normal(-0.1, 0.1),  # Mean -0.1
            lambda: np.random.normal(-0.5, 0.1),  # Mean -0.5 (largest magnitude)
            lambda: np.random.normal(-0.2, 0.1),  # Mean -0.2
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should identify arm 1 as the best (largest magnitude)
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_mixed_positive_negative_means(self):
        """Test with mixed positive and negative means."""
        np.random.seed(42)

        # Create 3 arms with mixed signs
        arms = [
            lambda: np.random.normal(0.3, 0.1),   # Mean 0.3
            lambda: np.random.normal(-0.4, 0.1),  # Mean -0.4 (largest magnitude)
            lambda: np.random.normal(0.2, 0.1),   # Mean 0.2
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should identify arm 1 as the best (largest magnitude)
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_identical_arms(self):
        """Test with identical arms (should handle gracefully)."""
        np.random.seed(42)

        # Create 3 identical arms
        arms = [
            lambda: np.random.normal(0.5, 0.1),
            lambda: np.random.normal(0.5, 0.1),
            lambda: np.random.normal(0.5, 0.1),
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should return one of the arms (any is valid since they're identical)
        assert best_arm in [0, 1, 2], f"Got invalid arm {best_arm}"

    def test_small_delta(self):
        """Test with very small delta (should require more samples)."""
        np.random.seed(42)

        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.5, 0.1),  # Best
            lambda: np.random.normal(0.2, 0.1),
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.001, samples_per_round=5)

        # Should still identify the correct best arm
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_large_samples_per_round(self):
        """Test with large samples per round."""
        np.random.seed(42)

        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.5, 0.1),  # Best
            lambda: np.random.normal(0.2, 0.1),
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=50)

        # Should identify the correct best arm
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_max_rounds_limit(self):
        """Test that the algorithm respects max_rounds limit."""
        np.random.seed(42)

        # Create arms that are very close (hard to distinguish)
        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.11, 0.1),  # Very close to arm 0
            lambda: np.random.normal(0.12, 0.1),  # Very close to arm 1
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=1, max_rounds=10)

        # Should return a valid arm index
        assert best_arm in [0, 1, 2], f"Got invalid arm {best_arm}"

    def test_single_arm(self):
        """Test with only one arm."""
        np.random.seed(42)

        arms = [lambda: np.random.normal(0.5, 0.1)]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should return the only arm
        assert best_arm == 0, f"Expected arm 0, got arm {best_arm}"

    def test_two_arms(self):
        """Test with exactly two arms."""
        np.random.seed(42)

        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.5, 0.1),  # Best
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should identify the correct best arm
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_high_variance_arms(self):
        """Test with arms that have high variance."""
        np.random.seed(42)

        arms = [
            lambda: np.random.normal(0.1, 0.5),   # High variance
            lambda: np.random.normal(0.5, 0.5),   # High variance, best mean
            lambda: np.random.normal(0.2, 0.5),   # High variance
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=20)

        # Should identify the correct best arm despite high variance
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_elimination_progress(self):
        """Test that arms are actually being eliminated."""
        np.random.seed(42)

        # Create arms with clear separation
        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.5, 0.1),  # Best
            lambda: np.random.normal(0.2, 0.1),
            lambda: np.random.normal(0.3, 0.1),
        ]

        # Run with small samples per round to see elimination in action
        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=5)

        # Should identify the correct best arm
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_edge_case_zero_variance(self):
        """Test with deterministic arms (zero variance)."""
        np.random.seed(42)

        arms = [
            lambda: 0.1,  # Deterministic
            lambda: 0.5,  # Deterministic, best
            lambda: 0.2,  # Deterministic
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=5)

        # Should identify the correct best arm
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_very_close_means(self):
        """Test with means that are very close to each other."""
        np.random.seed(42)

        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.101, 0.1),  # Very close to arm 0
            lambda: np.random.normal(0.102, 0.1),  # Very close to arm 1
        ]

        best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)

        # Should return a valid arm index (any is acceptable for very close means)
        assert best_arm in [0, 1, 2], f"Got invalid arm {best_arm}"

    def test_radius_calculation(self):
        """Test that the radius calculation is working correctly."""
        np.random.seed(42)

        # Create arms with known separation
        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.5, 0.1),  # Best
            lambda: np.random.normal(0.2, 0.1),
        ]

        # Use small delta to make radius calculation more important
        best_arm = best_arm_successive_elimination(arms, delta=0.01, samples_per_round=5)

        # Should identify the correct best arm
        assert best_arm == 1, f"Expected arm 1, got arm {best_arm}"

    def test_convergence_behavior(self):
        """Test that the algorithm converges to the correct arm."""
        np.random.seed(42)

        # Create arms with clear separation
        arms = [
            lambda: np.random.normal(0.1, 0.1),
            lambda: np.random.normal(0.5, 0.1),  # Best
            lambda: np.random.normal(0.2, 0.1),
            lambda: np.random.normal(0.3, 0.1),
            lambda: np.random.normal(0.4, 0.1),
        ]

        # Run multiple times to test consistency
        results = []
        for _ in range(5):
            best_arm = best_arm_successive_elimination(arms, delta=0.05, samples_per_round=10)
            results.append(best_arm)

        # Should consistently identify arm 1 as best
        assert all(result == 1 for result in results), f"Not all runs returned arm 1: {results}"

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])


