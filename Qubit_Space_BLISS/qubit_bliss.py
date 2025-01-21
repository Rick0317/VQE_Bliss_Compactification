""" We apply the principle of BLISS to Qubit Hamiltonian """
import numpy as np
from cirq import q
from openfermion import FermionOperator, QubitOperator, jordan_wigner
import pickle
from SolvableQubitHamiltonians.main_utils_partitioning import copy_hamiltonian
from SolvableQubitHamiltonians.qwc_decomposition import qwc_decomposition


def qubit_bliss(hamiltonian: QubitOperator):
    """
    Apply Bliss to the given Qubit Hamiltonian.
    Here, we use the product of Z's as the Symmetry operator.
    <Z1 Z2 ... Zn> = 1 if the number of electrons in the eigenstate is even
    :param hamiltonian:
    :return:
    """
    pass


if __name__ == "__main__":
    N = 8
    Ne = 4

    filename = f'../SolvableQubitHamiltonians/ham_lib/h4_sto-3g.pkl'
    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)

    total_particle = FermionOperator()
    for i in range(N):
        total_particle += FermionOperator(f'{i}^ {i}')

    o_example = QubitOperator('X0 X1 Y2 Y7')

    a_example = 1j * QubitOperator('Y0 X1 X3 X5')

    z_string = QubitOperator('Z0 Z1 Z2 Z3 Z4 Z5 Z6 Z7')
    # z_string = QubitOperator()
    # for i in range(8):
    #     z_string += 0.5 * QubitOperator(f'Z{i}')

    commut_check = a_example * z_string - z_string * a_example
    print(f"Check 1: {commut_check}")

    qubit_op = jordan_wigner(Hamil)
    H_copy = copy_hamiltonian(qubit_op)
    # decomp = sorted_insertion_decomposition(H_copy, methodtag)
    decomp = qwc_decomposition(H_copy)
    commut_check2 = qubit_op * z_string - z_string * qubit_op
    print(f"Check 2: {commut_check2}")

    commutator = qubit_op * a_example - a_example * qubit_op

    commut_check3 = commutator * z_string - z_string * commutator
    print(f"Check 3: {commut_check3}")

    print(len(decomp))
    print(type(decomp[0]))
    decomposed = decomp[0]
    commut_check4 = decomposed * z_string - z_string * decomposed
    print(f"Check 4: {commut_check4}")

    commutator2 = decomposed * a_example - a_example * decomposed

    commut_check5 = commutator2 * z_string - z_string * commutator2
    print(f"Check 5: {commut_check5}")

    term_items = commutator2.terms
    print(term_items.keys())
    for term, coeff in commutator2.terms.items():
        multiplied = z_string * QubitOperator(term)
        first_term, _ = next(iter(multiplied.terms.items()))
        print(first_term)
        if first_term in term_items.keys():
            print("Exists")

