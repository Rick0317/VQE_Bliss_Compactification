"""
This file implements the screening method used in the iQCC paper.
It will be used in ADAPT-VQE to screen off some of the gradient terms
before the measurements.
"""
from openfermion import QubitOperator, FermionOperator, bravyi_kitaev, normal_ordered
import pickle
from OperatorPools.generalized_fermionic_pool import get_anti_hermitian_two_body


def extract_flips(terms: tuple):
    """
    Given a tuple of Pauli operators, return the sets of flips
    :param terms:
    :return:
    """
    flip_set = set()
    flip_y_set = set()
    flip_x_set = set()

    for term in terms:
        if term[1] == 'Y':
            flip_set.add(term[0])
            flip_y_set.add(term[0])
        elif term[1] == 'X':
            flip_set.add(term[0])
            flip_x_set.add(term[0])

    return flip_set, flip_y_set, flip_x_set


def get_flip_indices_pauli_string(pauli_string: QubitOperator):
    """
    Return the indices set for the flips (Xj, Yj)
    :param pauli_string: This has to be a Pauli string.
    :return:
    """
    for term, coeff in pauli_string.terms.items():
        return extract_flips(term)


def get_flip_indices_list_hamiltonian(hamiltonian: QubitOperator):
    """
    Return the partitioning of hamiltonian based on the Flips.
    Each fragment shares the same flip indices.
    :param hamiltonian:
    :return:
    """
    partitioning = {}
    flip_indices_set = set()
    for term, coeff in hamiltonian.terms.items():
        flip_set, flip_y_set, flip_x_set = extract_flips(term)
        frozen_flip_set = frozenset(flip_set)
        if frozen_flip_set not in partitioning:
            flip_indices_set.add(frozen_flip_set)
            partitioning[frozen_flip_set] = [term]
        else:
            partitioning[frozen_flip_set].append(term)

    return partitioning, flip_indices_set


def get_y_parity(generator: QubitOperator):
    """
    Return the Y parity set for the generator.
    :param generator:
    :return:
    """
    is_parity_even = True
    for term, coeff in generator.terms.items():
        _, flip_y_set, __ = extract_flips(term)
        if len(flip_y_set) % 2 != 0:
            is_parity_even = False
            return is_parity_even
    return is_parity_even


def is_zero_gradients(hamiltonian: QubitOperator, generator: QubitOperator):
    """
    Given hamiltonian and generator, determine if the gradient [H, A] = 0
    or not based on the flips.
    :return:
    """
    _, hamil_flips \
        = get_flip_indices_list_hamiltonian(hamiltonian)
    _, generator_flips \
        = get_flip_indices_list_hamiltonian(generator)
    # is_y_even = get_y_parity(generator)
    #
    # if is_y_even:
    #     return True
    # return False

    if hamil_flips & generator_flips:
        return False
    else:
        return True


def get_non_zero_anti_hermitian_two_body(hamiltonian, N):
    H_q = bravyi_kitaev(hamiltonian)
    non_zero_anti_hermitian_two_body = []
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    anti_herm = get_anti_hermitian_two_body((p, r, s, q))
                    operator_q = bravyi_kitaev(anti_herm)

                    is_zero = is_zero_gradients(H_q, operator_q)
                    if not is_zero:
                        non_zero_anti_hermitian_two_body.append(anti_herm)
    return non_zero_anti_hermitian_two_body


if __name__ == '__main__':
    N = 8
    Ne = 4
    mol_name = "H4"

    filename = f'../ham_lib/h4_sto-3g.pkl'

    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)

    ordered_hamil = normal_ordered(Hamil)

    H_q = bravyi_kitaev(ordered_hamil)

    anti_herm = get_anti_hermitian_two_body((4, 2, 0, 5))
    operator_q = bravyi_kitaev(anti_herm)
    print(operator_q)

    is_zero = is_zero_gradients(H_q, operator_q)



    zero_gradient_sets = set()
    for p in range(N):
        for q in range(N):
            for r in range(N):
                for s in range(N):
                    anti_herm = get_anti_hermitian_two_body((p, r, s, q))
                    operator_q = bravyi_kitaev(anti_herm)

                    is_zero = is_zero_gradients(H_q, operator_q)
                    if is_zero:
                        zero_gradient_sets.add((p, r, s, q))
    print(zero_gradient_sets)
    total_terms = N ** 4
    print(f"Screened terms: {len(zero_gradient_sets)} / { total_terms }")
