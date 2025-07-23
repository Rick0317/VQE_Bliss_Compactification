import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path to import the functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BestArmIdentification.track_and_stop_binned import glr_binned, empirical_distribution_binned, create_bins

class TestGLRBinned:
    def test_basic_glr(self):
        """Test GLR with three clearly separated distributions."""
        np.random.seed(0)
        samples = [
            np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
            np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.1, 0.7, 0.1, 0.05, 0.05]),
            np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.05, 0.05, 0.1, 0.1, 0.7]),
        ]
        bins = create_bins(samples, n_bins=10)
        emp_dists = [empirical_distribution_binned(s, bins) for s in samples]
        counts = np.array([len(s) for s in samples])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        best_arm = np.argmax(
            np.abs([np.sum(d * bin_centers) for d in emp_dists]))
        print(f"Best arm is {best_arm}")
        glr = glr_binned(emp_dists, counts, best_arm, bin_centers)
        print(glr)
        assert glr > 0
        assert np.isfinite(glr)

    def test_identical_distributions(self):
        """GLR should be very small for identical distributions."""
        np.random.seed(0)
        samples = [np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.2]*5) for _ in range(3)]
        bins = create_bins(samples, n_bins=10)
        emp_dists = [empirical_distribution_binned(s, bins) for s in samples]
        counts = np.array([len(s) for s in samples])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        best_arm = 0

        glr = glr_binned(emp_dists, counts, best_arm, bin_centers)
        assert glr >= 0
        assert glr < 1e-2  # Should be very small

    def test_single_bin(self):
        """
        Test GLR with all samples in a single bin.

        """
        samples = [np.full(100, 2.5) for _ in range(3)]
        bins = np.array([2.0, 3.0])
        emp_dists = [empirical_distribution_binned(s, bins) for s in samples]
        counts = np.array([len(s) for s in samples])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        best_arm = 0

        glr = glr_binned(emp_dists, counts, best_arm, bin_centers)
        assert np.isclose(glr, 0.0, atol=1e-10)

    def test_edge_case_zero_counts(self):
        """Test GLR with one arm having very few samples."""
        np.random.seed(0)
        samples = [
            np.random.choice([0, 1, 2, 3, 4], size=1, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
            np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.1, 0.7, 0.1, 0.05, 0.05]),
            np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.05, 0.05, 0.1, 0.1, 0.7]),
        ]
        bins = create_bins(samples, n_bins=10)
        emp_dists = [empirical_distribution_binned(s, bins) for s in samples]
        counts = np.array([len(s) for s in samples])
        bin_centers = (bins[:-1] + bins[1:]) / 2
        best_arm = np.argmax([np.sum(d * bin_centers) for d in emp_dists])

        glr = glr_binned(emp_dists, counts, best_arm, bin_centers)
        assert glr >= 0
        assert np.isfinite(glr)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
