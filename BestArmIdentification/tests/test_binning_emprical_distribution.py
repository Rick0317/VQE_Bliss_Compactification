import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path to import the functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BestArmIdentification.track_and_stop_binned import create_bins, empirical_distribution_binned


class TestCreateBins:
    """
    Testing create_bins function
    """

    def test_basic_binning(self):
        """Test basic binning with simple data."""
        samples = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        n_bins = 5
        bins = create_bins(samples, n_bins)

        # Check that we get the expected number of bins
        assert len(bins) == n_bins + 1

        # Check that bins cover the range with padding
        expected_min = 1 - (6 - 1) * 0.01
        expected_max = 6 + (6 - 1) * 0.01
        assert np.isclose(bins[0], expected_min, atol=1e-10)
        assert np.isclose(bins[-1], expected_max, atol=1e-10)

        # Check that bins are evenly spaced
        bin_widths = np.diff(bins)
        assert np.allclose(bin_widths, bin_widths[0], atol=1e-10)

    def test_single_value_samples(self):
        """Test binning when all samples have the same value."""
        samples = [np.array([5.0]), np.array([5.0]), np.array([5.0])]
        n_bins = 10
        bins = create_bins(samples, n_bins)

        # Should still create n_bins + 1 bin edges
        assert len(bins) == n_bins + 1

        # All bins should be the same value since range is 0
        assert np.allclose(bins, 5.0, atol=1e-10)

    def test_empty_samples(self):
        """Test binning with empty sample arrays."""
        samples = [np.array([]), np.array([1, 2, 3])]
        n_bins = 5

        bins = create_bins(samples, n_bins)
        assert len(bins) == n_bins + 1


    def test_all_empty_samples(self):
        """Test binning when all sample arrays are empty."""
        samples = [np.array([]), np.array([])]
        n_bins = 5

        with pytest.raises(ValueError):
            create_bins(samples, n_bins)

    def test_different_n_bins(self):
        """Test binning with different numbers of bins."""
        samples = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        for n_bins in [1, 10, 50, 100]:
            bins = create_bins(samples, n_bins)
            assert len(bins) == n_bins + 1

    def test_negative_values(self):
        """Test binning with negative values."""
        samples = [np.array([-5, -3, -1]), np.array([1, 3, 5])]
        n_bins = 5
        bins = create_bins(samples, n_bins)

        # Check that bins cover the range with padding
        expected_min = -5 - (5 - (-5)) * 0.01
        expected_max = 5 + (5 - (-5)) * 0.01
        print(bins)
        assert np.isclose(bins[0], expected_min, atol=1e-10)
        assert np.isclose(bins[-1], expected_max, atol=1e-10)

    def test_floating_point_values(self):
        """Test binning with floating point values."""
        samples = [np.array([1.5, 2.7, 3.1]), np.array([4.2, 5.8, 6.3])]
        n_bins = 5
        bins = create_bins(samples, n_bins)

        # Check that bins are properly spaced
        bin_widths = np.diff(bins)
        assert np.allclose(bin_widths, bin_widths[0], atol=1e-10)

    def test_padding_calculation(self):
        """Test that padding is correctly calculated."""
        samples = [np.array([0, 10])]
        n_bins = 5
        bins = create_bins(samples, n_bins)

        # Padding should be 1% of the range (10 * 0.01 = 0.1)
        expected_min = 0 - 0.1
        expected_max = 10 + 0.1
        assert np.isclose(bins[0], expected_min, atol=1e-10)
        assert np.isclose(bins[-1], expected_max, atol=1e-10)


class TestEmpiricalDistributionBinned:
    """Test cases for the empirical_distribution_binned function."""

    def test_basic_distribution(self):
        """Test basic empirical distribution calculation."""
        samples = [np.array([1.5, 2.7, 3.1]), np.array([4.2, 5.8, 6.3])]
        n_bins = 100
        bins = create_bins(samples, n_bins)
        sample_1 = np.array([1.5, 2.7, 3.1])
        dist = empirical_distribution_binned(sample_1, bins)
        print(bins)
        print(dist)
        # Should have 3 bins (len(bins) - 1)
        assert len(dist) == 100

        # Should sum to 1
        assert np.isclose(np.sum(dist), 1.0, atol=1e-10)

        # All probabilities should be positive
        assert np.all(dist > 0)

    def test_alpha_parameter(self):
        """Test that alpha parameter adds small constant to avoid zero probabilities."""
        samples = np.array([1, 1, 1])  # All samples in first bin
        bins = np.array([0.5, 1.5, 2.5])

        # With default alpha
        dist_default = empirical_distribution_binned(samples, bins)

        # With larger alpha
        dist_large_alpha = empirical_distribution_binned(samples, bins, alpha=1.0)

        # The second bin should have higher probability with larger alpha
        assert dist_large_alpha[1] > dist_default[1]

    def test_empty_samples(self):
        """Test empirical distribution with empty samples."""
        samples = np.array([])
        bins = np.array([0, 1, 2])

        dist = empirical_distribution_binned(samples, bins)

        # Should still return a valid distribution
        assert len(dist) == 2
        assert np.isclose(np.sum(dist), 1.0, atol=1e-10)
        # Should be uniform due to alpha
        assert np.allclose(dist, dist[0], atol=1e-10)

    def test_samples_outside_bins(self):
        """Test behavior when samples fall outside the bin range."""
        samples = np.array([-1, 0, 1, 2, 3, 4])  # Some values outside bins
        bins = np.array([0.5, 1.5, 2.5])  # Only covers [0.5, 2.5)

        dist = empirical_distribution_binned(samples, bins)

        # Should still return valid distribution
        assert len(dist) == 2
        assert np.isclose(np.sum(dist), 1.0, atol=1e-10)
        assert np.all(dist > 0)

    def test_single_bin(self):
        """Test empirical distribution with single bin."""
        samples = np.array([1, 2, 3])
        bins = np.array([0, 4])  # Single bin covering [0, 4)

        dist = empirical_distribution_binned(samples, bins)

        assert len(dist) == 1
        assert np.isclose(dist[0], 1.0, atol=1e-10)

    def test_floating_point_samples(self):
        """Test empirical distribution with floating point samples."""
        samples = np.array([1.1, 1.9, 2.1, 2.9])
        bins = np.array([1.0, 2.0, 3.0])

        dist = empirical_distribution_binned(samples, bins)

        assert len(dist) == 2
        assert np.isclose(np.sum(dist), 1.0, atol=1e-10)
        assert np.all(dist > 0)

    def test_different_alpha_values(self):
        """Test empirical distribution with different alpha values."""
        samples = np.array([1, 1, 1])
        bins = np.array([0.5, 1.5, 2.5])

        dist_alpha_0 = empirical_distribution_binned(samples, bins, alpha=0.0)
        dist_alpha_1 = empirical_distribution_binned(samples, bins, alpha=1.0)

        # With alpha=0, first bin should have probability 1 (before normalization)
        # With alpha=1, all bins should have equal probability
        assert dist_alpha_0[0] > dist_alpha_1[0]
        assert dist_alpha_0[1] < dist_alpha_1[1]

    def test_normalization(self):
        """Test that the distribution is properly normalized."""
        samples = np.array([1, 2, 3, 4, 5])
        bins = np.array([0, 2, 4, 6])

        dist = empirical_distribution_binned(samples, bins)

        # Should sum to 1
        assert np.isclose(np.sum(dist), 1.0, atol=1e-10)

        # All probabilities should be between 0 and 1
        assert np.all(dist >= 0)
        assert np.all(dist <= 1)


class TestIntegration:
    """Integration tests combining both functions."""

    def test_create_bins_and_empirical_distribution(self):
        """Test that create_bins and empirical_distribution_binned work together."""
        samples = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        n_bins = 5

        # Create bins
        bins = create_bins(samples, n_bins)

        # Compute empirical distribution for each sample set
        dist1 = empirical_distribution_binned(samples[0], bins)
        dist2 = empirical_distribution_binned(samples[1], bins)

        # Both distributions should be valid
        assert len(dist1) == n_bins
        assert len(dist2) == n_bins
        assert np.isclose(np.sum(dist1), 1.0, atol=1e-10)
        assert np.isclose(np.sum(dist2), 1.0, atol=1e-10)

    def test_edge_cases_integration(self):
        """Test integration with edge cases."""
        # Test with overlapping samples
        samples = [np.array([1, 2, 3]), np.array([2, 3, 4])]
        n_bins = 3

        bins = create_bins(samples, n_bins)
        dist1 = empirical_distribution_binned(samples[0], bins)
        dist2 = empirical_distribution_binned(samples[1], bins)

        # Both should be valid distributions
        assert np.isclose(np.sum(dist1), 1.0, atol=1e-10)
        assert np.isclose(np.sum(dist2), 1.0, atol=1e-10)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
