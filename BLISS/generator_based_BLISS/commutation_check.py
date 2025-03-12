"""
This file checks the commutation relation between a Killer that annihilates
the reference state (HF) and the product of unitary operators that were chosen
prior to the current step.
"""
from openfermion import FermionOperator as FO, get_sparse_operator as gso
from scipy.linalg import expm


if __name__ == "__main__":
    generator = FO("2^ 0") - FO("0^ 2")
    polynomial_gen = 2 * generator

    matrix_rep = gso(polynomial_gen, 4).toarray()

    unitary_op = expm(matrix_rep)

    killer = FO("2^ 0")
    matrix_rep_killer = gso(killer, 4).toarray()

    commutator = unitary_op @ matrix_rep_killer - matrix_rep_killer @ unitary_op
    print(commutator)

