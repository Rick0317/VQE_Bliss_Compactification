"""
In this file, we achieve the followings:
1. Find a pool of Pauli Products from multiple operators
2.
3.
4.
"""
from openfermion import QubitOperator
from typing import List
from entities.paulis import PauliString
from utils.ferm_utils import ferm_to_qubit
from Decompositions.qwc_decomposition import qwc_decomposition
from Decompositions.fc_decomposition import fc_decomposition
import pickle


def get_pauli_product_pool(operators: List[QubitOperator]):
    """
    Given a list of operators, find a pool of Pauli Products
    :param operators:
    :return: set of Pauli Products (unique)
    """
    pauli_product_pool = []
    for operator in operators:
        for term in operator.terms:
            pauli_product_pool.append(PauliString(term))

    return list({str(p): p for p in pauli_product_pool}.values())


def find_qwc_groups(operators: List[PauliString]):
    """
    Given a list of PauliString, we find qwc groups
    :param operators: List of PauliStrings
    :return: Groups of PauliStrings based on QWC decomposition
    """
    qubit_op = QubitOperator()
    for operator in operators:
        qubit_op += operator.to_qubit_operator()
    decomposed = qwc_decomposition(qubit_op)
    return decomposed


def find_fc_groups(operators: List[PauliString]):
    """
    Given a list of PauliStrings, we find fc groups
    :param operators: List of PauliStrings
    :return: Groups of PauliStrings based on FC decomposition
    """
    qubit_op = QubitOperator()
    for operator in operators:
        qubit_op += operator.to_qubit_operator()

    decomposed = fc_decomposition(qubit_op)
    return decomposed


if __name__ == "__main__":

    ### Test the unique terms finding ###

    moltag = 'Beh2'

    N = 4
    Ne = 2

    filename = f'../ham_lib/h2_fer.bin'
    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)

    filename2 = f'../ham_lib/lih_fer.bin'
    with open(filename2, 'rb') as f:
        Hamil2 = pickle.load(f)

    H_q = ferm_to_qubit(Hamil)
    print(f"Qubit Hamiltonian {H_q}")
    H_q_2 = ferm_to_qubit(Hamil2)
    print(f"Qubit Hamiltonian {H_q_2}")

    pauli_product_pool = get_pauli_product_pool([H_q, H_q_2])
    # print(f"Pauli Product pool {pauli_product_pool}")

    print(f"Number of Pauli Products {len(pauli_product_pool)}")

    # qwc_groups = find_qwc_groups(pauli_product_pool)
    # print(f"QWC groups {len(qwc_groups)}")
    fc_groups = find_fc_groups(pauli_product_pool)
    print(f"FC groups {len(fc_groups)}")
