import pickle

from openfermion import (
normal_ordered,
get_majorana_operator,
bravyi_kitaev,
variance,
get_sparse_operator as gso,
expectation
)
from Decompositions.qwc_decomposition import qwc_decomposition
from BLISS.normal_bliss.customized_bliss_package import *
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
from SolvableQubitHamiltonians.main_utils_partitioning import copy_hamiltonian
from StatePreparation.reference_state_utils import get_cisd_gs
from itertools import combinations
from SharedPauli.shared_pauli_package import apply_shared_pauli
from BLISS.BLISS import bliss_three_body_indices_filtered
from OperatorPools.generalized_fermionic_pool import *
import os
import csv


def abs_of_dict_value(x):
    return np.abs(x[1])


def ferm_to_qubit(H: FermionOperator):
    Hqub = bravyi_kitaev(H)
    # Hqub -= Hqub.constant
    # Hqub.compress()
    # Hqub.terms = dict(
    # sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub


def commutator_variance(H: FermionOperator, decomp, N, psi):
    """
    Computes the variance of the [H, G] - K.
    :param H: The Hamiltonian to compute the variance of.
    :param decomp: The decomposition of the Hamiltonian.
    :param G: The fermion operator to define the gradient
    :param K: The Killer operator applied to the whole [H, G]
    :param N: The number of sites
    :return: The variance metric
    """
    # psi = ggs(gso(H, N))[1]

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, N), psi)
    return np.sum((vars) ** (1 / 2)) ** 2


def sz_operator(num_spin_orbitals):
    """Creates the total Sz operator in OpenFermion"""
    sz_op = FermionOperator()
    for i in range(num_spin_orbitals // 2):  # Iterate over spatial orbitals
        spin_up = 2 * i     # Even indices are spin-up
        spin_down = 2 * i + 1  # Odd indices are spin-down
        sz_op += 0.5 * (FermionOperator(((spin_up, 1), (spin_up, 0))) -
                        FermionOperator(((spin_down, 1), (spin_down, 0))))
    return sz_op


def get_hartree_fock(num_spin_orbitals, num_particles):
    state = np.zeros(2 ** num_spin_orbitals, dtype=np.complex128)
    binary_index = sum(2 ** q for q in
                          range(num_particles))
    state[binary_index] = 1

    return state


def generate_fixed_particle_basis(n_spin_orbitals, n_particles):
    """ Generate all basis states for a given particle number. """
    basis = []
    for state in combinations(range(n_spin_orbitals), n_particles):
        basis.append(sum(1 << i for i in state))  # Binary encoding
    return basis


def project_hamiltonian(H, basis):
    """ Project the full Hamiltonian onto the fixed-particle-number subspace. """
    n = len(basis)
    subspace_H = np.zeros((n, n))

    for i, b_i in enumerate(basis):
        for j, b_j in enumerate(basis):
            if i <= j:  # Only compute upper triangle (symmetric matrix)
                subspace_H[i, j] = H[b_i, b_j]
                subspace_H[j, i] = subspace_H[i, j]

    return subspace_H


if __name__ == '__main__':
    N = 8
    Ne = 4
    mol_name = "h4"

    filename = f'../ham_lib/h4_sto-3g.pkl'

    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)

    ordered_hamil = normal_ordered(Hamil)

    com_name_list = [
        (3, 2, 0, 1),
        (6, 3, 7, 0),
        (4, 3, 5, 2),
        (2, 5, 1, 6),
        (0, 7, 5, 2),
        (1, 6, 0, 7),
        (3, 6, 0, 5),
        (2, 1, 3, 4),
        (1, 6, 6, 5),
        (0, 1, 3, 2),
        (5, 4, 2, 1)
    ]
    anti_com_list = [
        get_anti_hermitian_two_body(indices) for indices in com_name_list
    ]

    for i in range(len(anti_com_list)):
        G = anti_com_list[i]
        hamil_copy1 = copy_ferm_hamiltonian(ordered_hamil)
        hamil_copy2 = copy_ferm_hamiltonian(ordered_hamil)
        g_copy1 = copy_ferm_hamiltonian(G)
        g_copy2 = copy_ferm_hamiltonian(G)

        commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2
        H = normal_ordered(commutator)
        H_in_q = ferm_to_qubit(H)

        energy, cisd_state = get_cisd_gs(mol_name, H_in_q, N, 'wfs', )

        # Define the total number operator N = sum_i a_i† a_i
        number_operator = FermionOperator()
        for mode in range(N):
            number_operator += FermionOperator(f"{mode}^ {mode}")  # a_i† a_i

        # Convert to qubit operator using Jordan-Wigner transformation
        number_operator_qubit = bravyi_kitaev(number_operator)

        print(
            f"Particle number Qubit: {expectation(gso(number_operator_qubit), cisd_state)}")


        print(
            f"Sz Qubit: {expectation(gso(ferm_to_qubit(sz_operator(N)), N), cisd_state)}")


        original_exp = expectation(gso(H_in_q, N), cisd_state)
        print(f"Original Expectation value Qubit: {original_exp}")

        copied_H = copy_ferm_hamiltonian(H)
        H_q = ferm_to_qubit(copied_H)
        print("Commutator Obtained in Fermion and Qubit space")
        H_copy = copy_hamiltonian(H_q)
        # decomp = sorted_insertion_decomposition(H_copy, methodtag)
        decomp = qwc_decomposition(H_copy)
        print("Original Decomposition Complete")
        original_var = commutator_variance(H_q, decomp, N, cisd_state).real
        print(f"Original variance: {original_var}")

        majo = get_majorana_operator(H)

        one_norm = 0
        for term, coeff in majo.terms.items():
            if term != ():
                one_norm += abs(coeff)

        print("Original 1-Norm", one_norm)
        copied_H2 = copy_ferm_hamiltonian(H)
        bliss_output = bliss_three_body_indices_filtered(copied_H2, N, Ne)

        bliss_exp = expectation(gso(ferm_to_qubit(bliss_output), N),
                                cisd_state)
        print(f"BLISS Expectation value Qubit : {bliss_exp}")

        H_before_bliss_test = copy_ferm_hamiltonian(H)
        H_bliss_output_test = copy_ferm_hamiltonian(bliss_output)
        # check_correctness(H_before_bliss_test, H_bliss_output_test, Ne)
        print("Bliss correctness check complete")

        H_bliss_output = copy_ferm_hamiltonian(bliss_output)
        H_bliss_q = ferm_to_qubit(bliss_output)
        print("BLISS Complete. Obtained in Fermion and Qubit space")

        majo_blissed = get_majorana_operator(H_bliss_output)
        blissed_one_norm = 0
        for term, coeff in majo_blissed.terms.items():
            if term != ():
                blissed_one_norm += abs(coeff)

        print("Blissed 1-Norm", blissed_one_norm)

        H_bliss_copy = copy_hamiltonian(H_bliss_q)

        blissed_decomp = qwc_decomposition(H_bliss_copy)
        print("Blissed Decomposition Complete")

        H_decompose_copy = copy_hamiltonian(H_bliss_q)

        blissed_vars = commutator_variance(H_decompose_copy,
                                           blissed_decomp.copy(), N,
                                           cisd_state).real

        print(f"Blissed variance: {blissed_vars}")

        H_q = copy_hamiltonian(H_bliss_q)

        var, last_var, measured_groups, expectation_v = apply_shared_pauli(H_q, blissed_decomp, N, Ne, cisd_state)

        print(f"Expectation value after Shared Pauli: {expectation_v}")

        print(f"Updated variance {last_var}")

        file_name = f"bliss_commutator_ghost_result_{mol_name}.csv"

        file_exists = os.path.isfile(file_name)
        # Open the file in append mode or write mode
        with open(file_name, mode='a' if file_exists else 'w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write the header only if the file doesn't exist
            if not file_exists:
                writer.writerow(
                    ['As', 'Original Variance', 'BLISS variance',
                     'Shared Pauli'])

            # Write the data
            writer.writerow(
                [com_name_list[i], original_var, blissed_vars, last_var])



