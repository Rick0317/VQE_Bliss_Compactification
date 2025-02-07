"""
Implementation of the shared Pauli technique
"""
import numpy as np
from openfermion import (
    variance,
    get_sparse_operator as gso
)
from paulis import PauliString, pauli_ops_to_qop
from SymplecticVectorSpace.space_F_definition import SpaceFVector, vector_2_pauli
from itertools import product


def update_decomp_w_shared_paulis(psi, N, original_decomp):
    """
    Update the commuting decomposition by introducing ghost paulis into some commuting sets.
    :param original_decomp: The original Pauli decomposition of the input Hamiltonian
    :return: The updated Pauli decomposition with ghost paulis in each set
    """
    new_decomp = original_decomp.copy()

    frag_combs = select_combs(psi, N, original_decomp)

    pauli_added_combs = select_sharable_paulis(frag_combs, original_decomp, N, psi)

    for combination in pauli_added_combs[:1]:
        (c, qubit_op, index_a, index_b) = combination
        new_decomp[index_a] -= c.real * qubit_op
        new_decomp[index_b] += c.real * qubit_op

    return new_decomp

def select_sharable_paulis(frag_combs, original_decomp, N, psi):
    """
    Given a set of fragments, we find the shareable Pauli operatoars between
    pairs of fragments so that we can transfer some coefficient to the smaller
    fragment.
    :param frag_combs: The pairs of fragments we consider
    :param original_decomp: The decomposition of the input Hamiltonian
    :param N: The number of sites in the Hamiltonian
    :param psi: The quantum state found by CISD/ For experiments, we use the exact groundstate.
    :return: [(c, pauli, frag_a, frag_b), ...]
    """

    variance_sum = 0
    for fragement in original_decomp:
        variance_sum += np.sqrt(variance(gso(fragement, N), psi))

    pauli_added_combs = []
    for combination in frag_combs:
        (index_a, index_b) = combination
        frag_a = original_decomp[index_a]
        frag_b = original_decomp[index_b]
        size_a = len(frag_a)
        size_b = len(frag_b)

        if size_b > size_a:
            frag_a, frag_b = frag_b, frag_a

        sharable_pauli_a2b = []

        # From the two fragments, find the Pauli operator that could be shared.
        for term_a, _ in frag_a.terms.items():
            for term_b, _ in frag_b.terms.items():
                if term_a == term_b:
                    sharable_pauli_a2b.append((frag_a, frag_b, term_a))

        for sharable_pauli in sharable_pauli_a2b:
            # TODO: Find the coefficeint c for each of the shareable Pauli string.
            c = ...

            pauli_added_combs.append((c, sharable_pauli, frag_a, frag_b))

    return pauli_added_combs


def select_combs(psi, N, original_decomp):
    """
    Given a decomposition of the input Hamiltonian, select the set of fragment combinations
    :param original_decomp:
    :return: A list of fragment combinations as indices in the decomposition: [(a, b)]
    """

    n_frag = len(original_decomp)
    vars = np.zeros(n_frag, dtype=np.complex128)
    for i, frag in enumerate(original_decomp):
        vars[i] = variance(gso(frag, N), psi)

    p = 4 * sum(np.sqrt(vars))

    score_board = {}
    for a in range(n_frag):
        for b in range(a+1, n_frag):
            var_a = vars[a]
            var_b = vars[b]

            # The score is the upper-bound of the variance metric reduction
            # Eq (15) of the ghost Pauli paper.
            score = p * (np.sqrt(var_a * var_b)) / (np.sqrt(var_a) + np.sqrt(var_b))
            score_board[(a, b)] = score

    sorted_items = sorted(score_board.items(), key=lambda item: item[1],
                          reverse=True)

    top_50_count = len(score_board) // 2 + (
        1 if len(score_board) % 2 != 0 else 0)

    top_keys = [key for key, value in sorted_items[:top_50_count]]
    print(top_keys)

    return top_keys


if __name__ == '__main__':
