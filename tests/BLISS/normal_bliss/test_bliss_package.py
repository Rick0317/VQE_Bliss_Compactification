import unittest
from BLISS.normal_bliss.bliss_package import *

class TestBlissPackage(unittest.TestCase):
    def test_params_to_matrix_op(self):
        n = 2
        params = [1, 2, 3]
        ferm_op = params_to_matrix_op(params, n)
        print(ferm_op)

        n = 4
        params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ferm_op = params_to_matrix_op(params, n)
        print(ferm_op)

    def test_params_to_tensor_op(self):
        n = 2
        params = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        ferm_op = params_to_tensor_op(params, n)
        print(ferm_op)

    def test_params_to_tensor_specific_op(self):
        n = 2
        params = [i for i in range(n ** 4)]
        ferm_op = params_to_tensor_specific_op(params, n, [0, 1])
        print(ferm_op)

    def test_construct_H_bliss_mu3_o2(self):
        pass

    def test_construct_H_bliss_mu3_cheapo(self):
        pass

    def test_construct_H_bliss_mu123_o12(self):
        pass

    def test_construct_H_bliss_m12_o1(self):
        pass

    def test_optimize_bliss_mu3_o2(self):
        pass

    def test_optimize_bliss_mu3_cheapo(self):
        pass

    def test_optimize_bliss_mu123_o12(self):
        pass

    def test_optimize_bliss_m12_o1(self):
        pass

    def test_check_correctness(self):
        pass
