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
        qubit_op = QubitOperator('X0 Y3', 0.5)
        flip_set, flip_y_set, flip_x_set \
            = get_flip_indices_pauli_string(qubit_op)
        assert flip_set == {0, 3}, "Flip set doesn't match"
        assert flip_y_set == {3}, "Flip set doesn't match"
        assert flip_x_set == {0}, "Flip set doesn't match"

    def test_get_y_parity(self):
        qubit_op = QubitOperator('X0 Y3', 0.5)
        is_parity_even = get_y_parity(qubit_op)
        assert not is_parity_even, "Y parity doesn't match"

        qubit_op = QubitOperator('Y0 Y3', 0.5)
        is_parity_even = get_y_parity(qubit_op)
        assert is_parity_even, "Y parity doesn't match"


    def test_is_zero_gradients(self):
        hamiltonian = QubitOperator('X1 Y2', 0.5) + QubitOperator('X3 Z5', 0.5)
        qubit_op = QubitOperator('X0 Y3', 0.5)
        is_zero = is_zero_gradients(hamiltonian, qubit_op)
        assert is_zero, "Gradient doesn't match"

        hamiltonian = QubitOperator('X0 Y3', 0.5) + QubitOperator('X3 Z5', 0.5)
        qubit_op = QubitOperator('X0 Y3', 0.5)
        is_zero = is_zero_gradients(hamiltonian, qubit_op)
        assert not is_zero, "Gradient doesn't match"
