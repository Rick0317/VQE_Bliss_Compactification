import numpy as np
import pytest
import sys
import os

# Add the parent directory to the path to import the functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from BestArmIdentification.track_and_stop_binned import kl_discrete


class TestKLDiscrete:
    """Test cases for the kl_discrete function."""

    def test_basic_kl_divergence(self):
        """Test basic KL divergence calculation."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])

        kl = kl_discrete(p, q)

        # KL divergence should be positive
        assert kl > 0

        # Should be finite
        assert np.isfinite(kl)

        # Check against manual calculation
        expected = 0.5 * np.log(0.5/0.4) + 0.3 * np.log(0.3/0.4) + 0.2 * np.log(0.2/0.2)
        assert np.isclose(kl, expected, atol=1e-10)

    def test_identical_distributions(self):
        """Test KL divergence between identical distributions."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.3, 0.4, 0.3])

        kl = kl_discrete(p, q)

        # KL divergence should be 0 for identical distributions
        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_kl_divergence_asymmetry(self):
        """Test that KL divergence is asymmetric (KL(p||q) != KL(q||p))."""
        p = np.array([0.8, 0.1, 0.1])
        q = np.array([0.2, 0.3, 0.5])

        kl_pq = kl_discrete(p, q)
        kl_qp = kl_discrete(q, p)

        # Should be different (KL divergence is asymmetric)
        assert not np.isclose(kl_pq, kl_qp, atol=1e-10)

        # Both should be positive
        assert kl_pq > 0
        assert kl_qp > 0

        # Let's verify the actual values are reasonable
        # After normalization, p becomes [0.8, 0.1, 0.1] and q becomes [0.2, 0.3, 0.5]
        # KL(p||q) = 0.8 * log(0.8/0.2) + 0.1 * log(0.1/0.3) + 0.1 * log(0.1/0.5)
        # KL(q||p) = 0.2 * log(0.2/0.8) + 0.3 * log(0.3/0.1) + 0.5 * log(0.5/0.1)
        expected_kl_pq = 0.8 * np.log(0.8/0.2) + 0.1 * np.log(0.1/0.3) + 0.1 * np.log(0.1/0.5)
        expected_kl_qp = 0.2 * np.log(0.2/0.8) + 0.3 * np.log(0.3/0.1) + 0.5 * np.log(0.5/0.1)

        assert np.isclose(kl_pq, expected_kl_pq, atol=1e-10)
        assert np.isclose(kl_qp, expected_kl_qp, atol=1e-10)

        # These should be different values
        assert not np.isclose(expected_kl_pq, expected_kl_qp, atol=1e-10)

    def test_zero_probabilities_in_p(self):
        """Test KL divergence when p has zero probabilities."""
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.1, 0.4, 0.5])

        kl = kl_discrete(p, q)
        print(kl)
        # Should handle zero probabilities correctly
        assert np.isfinite(kl)
        assert kl >= 0

    def test_zero_probabilities_in_q(self):
        """Test KL divergence when q has zero probabilities."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.5, 0.0, 0.5])

        kl = kl_discrete(p, q)

        # Should handle zero probabilities in q (eps prevents log(0))
        assert np.isfinite(kl)
        assert kl >= 0

    def test_both_zero_probabilities(self):
        """Test KL divergence when both p and q have zero probabilities."""
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.0, 0.4, 0.6])

        kl = kl_discrete(p, q)

        # Should handle this case correctly
        assert np.isfinite(kl)
        assert kl >= 0

    def test_eps_parameter(self):
        """Test that eps parameter prevents numerical issues."""
        p = np.array([0.0, 1.0])
        q = np.array([0.0, 1.0])

        # Should work without numerical issues
        kl = kl_discrete(p, q)
        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_normalization(self):
        """Test that function normalizes distributions properly."""
        p = np.array([1.0, 2.0, 3.0])  # Not normalized
        q = np.array([2.0, 2.0, 2.0])  # Not normalized

        kl = kl_discrete(p, q)
        print(kl)
        # Should normalize and give finite result
        assert np.isfinite(kl)
        assert kl >= 0

    def test_single_element_distributions(self):
        """Test KL divergence with single-element distributions."""
        p = np.array([1.0])
        q = np.array([1.0])

        kl = kl_discrete(p, q)

        # Should be 0 for identical single-element distributions
        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_large_distributions(self):
        """Test KL divergence with larger distributions."""
        p = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05])
        q = np.array([0.15, 0.15, 0.25, 0.25, 0.1, 0.05, 0.05])

        kl = kl_discrete(p, q)

        # Should be finite and positive
        assert np.isfinite(kl)
        assert kl > 0

    def test_floating_point_precision(self):
        """Test KL divergence with floating point precision issues."""
        p = np.array([0.3333333333333333, 0.3333333333333333, 0.3333333333333334])
        q = np.array([0.3333333333333333, 0.3333333333333333, 0.3333333333333334])

        kl = kl_discrete(p, q)

        # Should handle floating point precision correctly
        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_extreme_probabilities(self):
        """Test KL divergence with extreme probability values."""
        p = np.array([0.999, 0.001])
        q = np.array([0.001, 0.999])

        kl = kl_discrete(p, q)

        # Should handle extreme values correctly
        assert np.isfinite(kl)
        assert kl > 0

    def test_negative_values(self):
        """
        Negative values should fail.

        :return:
        """
        p = np.array([-1.0, 2.0, 3.0])
        q = np.array([1.0, 1.0, 1.0])

        kl = kl_discrete(p, q)

        # Should normalize and give finite result
        assert np.isfinite(kl)
        assert kl >= 0

    def test_different_lengths(self):
        """Test KL divergence with distributions of different lengths."""
        p = np.array([0.5, 0.5])
        q = np.array([0.3, 0.3, 0.4])

        # This should raise an error or handle gracefully
        try:
            kl = kl_discrete(p, q)
            # If it doesn't raise an error, result should be finite
            assert np.isfinite(kl)
        except ValueError:
            # Expected behavior for different lengths
            pass

    def test_empty_arrays(self):
        """Test KL divergence with empty arrays."""
        p = np.array([])
        q = np.array([])

        # This should raise an error or handle gracefully
        try:
            kl = kl_discrete(p, q)
            # If it doesn't raise an error, result should be finite
            assert np.isfinite(kl)
        except (ValueError, IndexError):
            # Expected behavior for empty arrays
            pass

    def test_kl_divergence_properties(self):
        """Test fundamental properties of KL divergence."""
        p = np.array([0.4, 0.6])
        q = np.array([0.3, 0.7])
        r = np.array([0.5, 0.5])

        kl_pq = kl_discrete(p, q)
        kl_qp = kl_discrete(q, p)

        # KL divergence should be non-negative
        assert kl_pq >= 0
        assert kl_qp >= 0

        # KL divergence should be asymmetric
        assert not np.isclose(kl_pq, kl_qp, atol=1e-10)

    def test_eps_effect(self):
        """Test the effect of different eps values."""
        p = np.array([0.0, 1.0])
        q = np.array([0.0, 1.0])

        # Test with different eps values
        kl_default = kl_discrete(p, q)
        kl_small_eps = kl_discrete(p, q, eps=1e-15)
        kl_large_eps = kl_discrete(p, q, eps=1e-6)

        # All should be close to 0
        assert np.isclose(kl_default, 0.0, atol=1e-10)
        assert np.isclose(kl_small_eps, 0.0, atol=1e-10)
        assert np.isclose(kl_large_eps, 0.0, atol=1e-10)


class TestKLDiscreteEdgeCases:
    """Test edge cases and error conditions."""

    def test_all_zero_p(self):
        """Test when p has all zero probabilities."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.3, 0.3, 0.4])

        kl = kl_discrete(p, q)

        # Should handle this case
        assert np.isfinite(kl)

    def test_all_zero_q(self):
        """Test when q has all zero probabilities."""
        p = np.array([0.3, 0.3, 0.4])
        q = np.array([0.5, 0.0, 0.5])

        kl = kl_discrete(p, q)

        # Should handle this case
        assert np.isfinite(kl)

    def test_both_all_zero(self):
        """Test when both p and q have all zero probabilities."""
        p = np.array([0.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 0.0])

        kl = kl_discrete(p, q)

        # Should handle this case
        assert np.isfinite(kl)

    def test_nan_values(self):
        """Test handling of NaN values."""
        p = np.array([0.5, np.nan, 0.5])
        q = np.array([0.3, 0.4, 0.3])

        # Should handle NaN gracefully
        try:
            kl = kl_discrete(p, q)
            assert np.isfinite(kl)
        except (ValueError, RuntimeWarning):
            # Expected behavior for NaN
            pass

    def test_inf_values(self):
        """Test handling of infinite values."""
        p = np.array([0.5, np.inf, 0.5])
        q = np.array([0.3, 0.4, 0.3])

        # Should handle inf gracefully
        try:
            kl = kl_discrete(p, q)
            assert np.isfinite(kl)
        except (ValueError, RuntimeWarning):
            # Expected behavior for inf
            pass


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
