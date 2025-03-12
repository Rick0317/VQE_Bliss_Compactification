import unittest
from Screenings.flip_based_screening import *


class TestFlipBasedScreening(unittest.TestCase):
    def test_extract_flips(self):
        set_1, set_y, set_x = extract_flips(((0, "Z"), (1, "Y"), (2, "X")))
        assert set_1 == {1, 2}, "Set doesn't match"
        assert set_y == {1}, "Set doesn't match"
        assert set_x == {2}, "Set doesn't match"

    def test_get_flip_indices_list_hamiltonian(self):
        """
        Test get_flip_indices_list_hamiltonian
        :return:
        """
        # TODO
        pass

    def test_get_y_parity(self):
        # TODO
        pass

    def test_is_zero_gradients(self):
        # TODO
        pass
