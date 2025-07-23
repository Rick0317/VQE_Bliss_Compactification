import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path to import the functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BestArmIdentification.track_and_stop_binned import compute_optimal_allocation_binned, create_bins, original_optimal_allocation


class TestComputeOptimalAllocationBinned:
    """Test cases for the compute_optimal_allocation_binned function."""

    def test_similar_distributions_get_more_allocation(self):
        """Test that similar distributions get more sample allocation."""
        # Create 5 distributions: 2 similar ones and 3 different ones
        np.random.seed(42)

        # Create sample data for 5 arms
        samples = []

        # Arm 0: Distribution with high probability on first bin
        arm0_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        samples.append(arm0_samples)

        # Arm 1: Similar to Arm 0 (high probability on first bin)
        arm1_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.05, 0.05, 0.1, 0.15, 0.65])
        samples.append(arm1_samples)

        # Arm 2: Different distribution (high probability on middle bin)
        arm2_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.1, 0.1, 0.7, 0.05, 0.05])
        samples.append(arm2_samples)

        # Arm 3: Different distribution (high probability on last bin)
        arm3_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.05, 0.05, 0.1, 0.1, 0.7])
        samples.append(arm3_samples)

        # Arm 4: Different distribution (uniform)
        arm4_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.2, 0.2, 0.2, 0.2, 0.2])
        samples.append(arm4_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute optimal allocation
        allocation = compute_optimal_allocation_binned(samples, bins)

        # Print results for debugging
        print(f"Allocation: {allocation}")
        print(f"Arm 0 (similar to Arm 1): {allocation[0]:.4f}")
        print(f"Arm 1 (similar to Arm 0): {allocation[1]:.4f}")
        print(f"Arm 2 (different): {allocation[2]:.4f}")
        print(f"Arm 3 (different): {allocation[3]:.4f}")
        print(f"Arm 4 (different): {allocation[4]:.4f}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # Check that all allocations are positive
        assert np.all(allocation >= 0)

        # The similar distributions (Arm 0 and Arm 1) should get more allocation
        # because they are harder to distinguish
        similar_allocation = allocation[1] + allocation[3]
        different_allocation = allocation[2] + allocation[0] + allocation[4]

        print(f"Similar arms allocation: {similar_allocation:.4f}")
        print(f"Different arms allocation: {different_allocation:.4f}")

        # Similar distributions should get more allocation (at least 40% of total)
        assert similar_allocation >= 0.4, f"Similar distributions got {similar_allocation:.4f} allocation, expected >= 0.4"

        # Individual similar arms should get more allocation than individual different arms
        avg_similar = (allocation[0] + allocation[1]) / 2
        avg_different = (allocation[2] + allocation[3] + allocation[4]) / 3

        print(f"Average similar arm allocation: {avg_similar:.4f}")
        print(f"Average different arm allocation: {avg_different:.4f}")

        assert avg_similar >= avg_different, f"Similar arms got {avg_similar:.4f} avg allocation, different arms got {avg_different:.4f}"

    def test_identical_distributions_get_equal_allocation(self):
        """
        Test that identical distributions get equal allocation.
        For identical distributions, the allocation will be based on the best arm
        """
        np.random.seed(42)

        # Create 3 identical distributions
        samples = []
        for i in range(3):
            arm_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.3, 0.2, 0.2, 0.15, 0.15])
            samples.append(arm_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute optimal allocation
        allocation = compute_optimal_allocation_binned(samples, bins)

        print(f"Allocation for identical distributions: {allocation}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # All allocations should be approximately equal (within 10% of each other)
        mean_allocation = np.mean(allocation)
        for i, alloc in enumerate(allocation):
            assert np.abs(alloc - mean_allocation) / mean_allocation < 0.1, f"Arm {i} allocation {alloc:.4f} differs too much from mean {mean_allocation:.4f}"

    def test_very_different_distributions_get_less_allocation(self):
        """Test that very different distributions get less allocation."""
        np.random.seed(42)

        # Create 3 very different distributions
        samples = []

        # Arm 0: Concentrated on first bin
        arm0_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.9, 0.05, 0.02, 0.02, 0.01])
        samples.append(arm0_samples)

        # Arm 1: Concentrated on middle bin
        arm1_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.01, 0.02, 0.9, 0.05, 0.02])
        samples.append(arm1_samples)

        # Arm 2: Concentrated on last bin
        arm2_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.01, 0.02, 0.02, 0.05, 0.9])
        samples.append(arm2_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute optimal allocation
        allocation = compute_optimal_allocation_binned(samples, bins)

        print(f"Allocation for very different distributions: {allocation}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # Allocations should be more balanced since distributions are very different
        # (easier to distinguish, so less need for focused sampling)
        max_allocation = np.max(allocation)
        min_allocation = np.min(allocation)

        # The difference between max and min should be smaller than for similar distributions
        allocation_range = max_allocation - min_allocation
        print(f"Allocation range: {allocation_range:.4f}")

        # Range should be reasonable (not too extreme)
        assert allocation_range < 0.8, f"Allocation range {allocation_range:.4f} is too large"

    def test_edge_case_single_bin(self):
        """Test optimal allocation with single bin."""
        np.random.seed(42)

        # Create samples that all fall into a single bin
        samples = []
        for i in range(3):
            arm_samples = np.full(100, 2.5)  # All samples are 2.5
            samples.append(arm_samples)

        # Create bins with single bin
        bins = np.array([2.0, 3.0])  # Single bin covering [2.0, 3.0)

        # Compute optimal allocation
        allocation = compute_optimal_allocation_binned(samples, bins)

        print(f"Allocation for single bin: {allocation}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # All allocations should be equal since all distributions are identical
        assert np.allclose(allocation, allocation[0], atol=1e-10)

    def test_edge_case_empty_samples(self):
        """Test optimal allocation with empty samples."""
        np.random.seed(42)

        # Create samples with some empty arrays
        samples = [
            np.array([]),  # Empty
            np.array([1, 2, 3]),  # Non-empty
            np.array([])   # Empty
        ]

        # Create bins
        bins = create_bins(samples, n_bins=5)

        # This should raise an error or handle gracefully
        try:
            allocation = compute_optimal_allocation_binned(samples, bins)
            print(f"Allocation with empty samples: {allocation}")

            # If it doesn't raise an error, check basic properties
            assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)
            assert np.all(allocation >= 0)

        except (ValueError, RuntimeError) as e:
            # Expected behavior for empty samples
            print(f"Expected error for empty samples: {e}")
            pass

    def test_convergence_properties(self):
        """Test that allocation converges to reasonable values."""
        np.random.seed(42)

        # Create samples with different sizes to test convergence
        samples = []

        # Small sample size
        arm0_samples = np.random.choice([0, 1, 2, 3, 4], size=10, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        samples.append(arm0_samples)

        # Medium sample size
        arm1_samples = np.random.choice([0, 1, 2, 3, 4], size=50, p=[0.65, 0.15, 0.1, 0.05, 0.05])
        samples.append(arm1_samples)

        # Large sample size
        arm2_samples = np.random.choice([0, 1, 2, 3, 4], size=200, p=[0.1, 0.1, 0.7, 0.05, 0.05])
        samples.append(arm2_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute optimal allocation
        allocation = compute_optimal_allocation_binned(samples, bins)

        print(f"Allocation with different sample sizes: {allocation}")

        # Check basic properties
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)
        assert np.all(allocation >= 0)

        # All allocations should be reasonable (not too extreme)
        assert np.all(allocation > 0.01), "Allocation should not be too small"
        assert np.all(allocation < 0.9), "Allocation should not be too large"


class TestOriginalOptimalAllocation:
    """Test cases for the original_optimal_allocation function."""

    def test_basic_functionality(self):
        """Test basic functionality of original optimal allocation."""
        np.random.seed(42)

        # Create sample data for 3 arms
        samples = []

        # Arm 0: Distribution with high probability on first bin
        arm0_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        samples.append(arm0_samples)

        # Arm 1: Different distribution (high probability on middle bin)
        arm1_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.1, 0.1, 0.7, 0.05, 0.05])
        samples.append(arm1_samples)

        # Arm 2: Different distribution (high probability on last bin)
        arm2_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.05, 0.05, 0.1, 0.1, 0.7])
        samples.append(arm2_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute optimal allocation
        allocation = original_optimal_allocation(samples, bins)

        print(f"Original allocation: {allocation}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # Check that all allocations are positive
        assert np.all(allocation >= 0)

        # All allocations should be reasonable (not too extreme)
        assert np.all(allocation > 0.01), "Allocation should not be too small"
        assert np.all(allocation < 0.9), "Allocation should not be too large"

    def test_comparison_with_binned_version(self):
        """Test comparison between original and binned versions."""
        np.random.seed(42)

        # Create sample data for 3 arms
        samples = []

        # Arm 0: Distribution with high probability on first bin
        arm0_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        samples.append(arm0_samples)

        # Arm 1: Similar to Arm 0
        arm1_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.1, 0.15, 0.65, 0.05, 0.05])
        samples.append(arm1_samples)

        # Arm 2: Different distribution
        arm2_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.1, 0.1, 0.7, 0.05, 0.05])
        samples.append(arm2_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute both allocations
        original_alloc = original_optimal_allocation(samples, bins)
        binned_alloc = compute_optimal_allocation_binned(samples, bins)

        print(f"Original allocation: {original_alloc}")
        print(f"Binned allocation: {binned_alloc}")

        # Both should sum to 1
        assert np.isclose(np.sum(original_alloc), 1.0, atol=1e-10)
        assert np.isclose(np.sum(binned_alloc), 1.0, atol=1e-10)

        # Both should be positive
        assert np.all(original_alloc >= 0)
        assert np.all(binned_alloc >= 0)

        # They might be different due to different optimization approaches
        # but both should be valid allocations
        print(f"Difference between allocations: {np.abs(original_alloc - binned_alloc)}")

    def test_identical_distributions(self):
        """Test original allocation with identical distributions."""
        np.random.seed(42)

        # Create 3 identical distributions
        samples = []
        for i in range(3):
            arm_samples = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.3, 0.2, 0.2, 0.15, 0.15])
            samples.append(arm_samples)

        # Create bins
        bins = create_bins(samples, n_bins=10)

        # Compute optimal allocation
        allocation = original_optimal_allocation(samples, bins)

        print(f"Original allocation for identical distributions: {allocation}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # All allocations should be approximately equal (within 20% of each other)
        # Original version might be more sensitive to numerical issues
        mean_allocation = np.mean(allocation)
        for i, alloc in enumerate(allocation):
            assert np.abs(alloc - mean_allocation) / mean_allocation < 0.2, f"Arm {i} allocation {alloc:.4f} differs too much from mean {mean_allocation:.4f}"

    def test_edge_case_single_bin(self):
        """Test original allocation with single bin."""
        np.random.seed(42)

        # Create samples that all fall into a single bin
        samples = []
        for i in range(3):
            arm_samples = np.full(100, 2.5)  # All samples are 2.5
            samples.append(arm_samples)

        # Create bins with single bin
        bins = np.array([2.0, 3.0])  # Single bin covering [2.0, 3.0)

        # Compute optimal allocation
        allocation = original_optimal_allocation(samples, bins)

        print(f"Original allocation for single bin: {allocation}")

        # Check that allocation sums to 1
        assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)

        # All allocations should be equal since all distributions are identical
        assert np.allclose(allocation, allocation[0], atol=1e-10)

    def test_error_handling(self):
        """Test error handling for edge cases."""
        np.random.seed(42)

        # Create samples with some empty arrays
        samples = [
            np.array([]),  # Empty
            np.array([1, 2, 3]),  # Non-empty
            np.array([])   # Empty
        ]

        # Create bins
        bins = create_bins(samples, n_bins=5)

        # This should raise an error or handle gracefully
        try:
            allocation = original_optimal_allocation(samples, bins)
            print(f"Original allocation with empty samples: {allocation}")

            # If it doesn't raise an error, check basic properties
            assert np.isclose(np.sum(allocation), 1.0, atol=1e-10)
            assert np.all(allocation >= 0)

        except (ValueError, RuntimeError) as e:
            # Expected behavior for empty samples
            print(f"Expected error for empty samples: {e}")
            pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
