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
from scipy.optimize import minimize
from BLISS.normal_bliss.customized_bliss_package import *
from BLISS.normal_bliss.indices_filter import filter_indices
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
from SolvableQubitHamiltonians.main_utils_partitioning import copy_hamiltonian
from SharedPauli.shared_paulis import (
    get_sharable_paulis,
    get_share_pauli_only_decomp,
    get_pw_grp_idxes_no_fix_len,
    get_overlapping_decomp,
    get_sharable_only_decomp,
    get_coefficient_orderings,
    get_all_pw_indices,
    get_pauli_coeff_map
)
from StatePreparation.reference_state_utils import get_cisd_gs
import time
from itertools import combinations

from SharedPauli.coefficient_optimizer import (
get_split_measurement_variance_unconstrained,
get_meas_alloc,
optimize_coeffs_parallel

)

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
        '3^ 2^ 0 1',
        '6^ 3^ 7 0',
        '4^ 3^ 5 2',
        '2^ 5^ 1 6',
        '0^ 7^ 5 2',
        '1^ 6^ 0 7',
        '3^ 6^ 0 5',
        '2^ 1^ 3 4',
        '1^ 6^ 6 5',
        '0^ 1^ 3 2',
        '5^ 4^ 2 1'
    ]
    # idx_list = [0, 1, 2, 3]
    anti_com_list = [
        FermionOperator('3^ 2^ 0 1') - FermionOperator('1^ 0^ 2 3'),
        FermionOperator('6^ 3^ 7 0') - FermionOperator('0^ 7^ 3 6'),
        FermionOperator('4^ 3^ 5 2') - FermionOperator('2^ 5^ 3 4'),
        FermionOperator('2^ 5^ 1 6') - FermionOperator('6^ 1^ 5 2'),
        FermionOperator('0^ 7^ 5 2') - FermionOperator('2^ 5^ 7 0'),
        FermionOperator('1^ 6^ 0 7') - FermionOperator('7^ 0^ 6 1'),
        FermionOperator('3^ 6^ 0 5') - FermionOperator('5^ 0^ 6 3'),
        FermionOperator('2^ 1^ 3 4') - FermionOperator('4^ 3^ 1 2'),
        FermionOperator('1^ 6^ 6 5') - FermionOperator('5^ 6^ 6 1'),
        FermionOperator('0^ 1^ 3 2') - FermionOperator('2^ 3^ 1 0'),
        FermionOperator('5^ 4^ 2 1') - FermionOperator('1^ 2^ 4 5'),

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

        # ground_state_commute = ggs(gso(H_in_q, N))[1]

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

        two_body_list = filter_indices(H, N, Ne)
        if len(two_body_list) != 0:

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
            optimization_wrapper, initial_guess = optimize_bliss_mu3_customizable(
                copied_H2, N, Ne, two_body_list)

            res = minimize(optimization_wrapper, initial_guess, method='Powell',
                           options={'disp': True, 'maxiter': 100000})

            H_before_modification = copy_ferm_hamiltonian(H)
            bliss_output, killer = construct_H_bliss_mu3_customizable(
                H_before_modification, res.x, N, Ne, two_body_list)

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

            start = time.time()

            coeff_map = get_pauli_coeff_map(blissed_decomp)

            # 1
            sharable_paulis_dict, sharable_paulis_list, sharable_pauli_indices_list = get_sharable_paulis(
                blissed_decomp)
            sharable_paulis_fixed_list = [indices[-1] for indices in
                                          sharable_pauli_indices_list]

            # 2
            fragment_idx_to_sharable_paulis, pw_grp_idxes_no_fix, pw_grp_idxes_fix = get_share_pauli_only_decomp(
                sharable_paulis_dict)

            pw_grp_idxes_no_fix_len = get_pw_grp_idxes_no_fix_len(
                pw_grp_idxes_no_fix)

            # 3
            all_sharable_contained_decomp, all_sharable_no_fixed_decomp = get_overlapping_decomp(
                sharable_paulis_dict, blissed_decomp, pw_grp_idxes_fix)

            # 4
            sharable_only_decomp, sharable_only_no_fixed_decomp = get_sharable_only_decomp(
                sharable_paulis_dict, blissed_decomp, pw_grp_idxes_fix,
                coeff_map)

            # 5
            fixed_grp, all_sharable_contained_no_fixed_decomp, new_sharable_pauli_indices_list, new_grp_len_list, new_grp_idx_start \
                = get_coefficient_orderings(sharable_only_decomp,
                                            sharable_paulis_list,
                                            sharable_pauli_indices_list,
                                            coeff_map)

            # 6
            pw_indices = get_all_pw_indices(sharable_paulis_list,
                                            sharable_only_no_fixed_decomp,
                                            pw_grp_idxes_no_fix,
                                            new_grp_idx_start)

            meas_alloc = get_meas_alloc(blissed_decomp, N, cisd_state)

            # Get the linear equation for finding the gradient descent direction
            matrix, b = optimize_coeffs_parallel(
                pw_grp_idxes_fix,
                pw_grp_idxes_no_fix_len,
                meas_alloc,
                sharable_only_decomp,
                sharable_only_no_fixed_decomp,
                pw_indices, cisd_state, N, blissed_decomp, alpha=0.001)
            print("M, b obtained")
            sol = np.linalg.lstsq(matrix, b.T, rcond=None)
            x0 = sol[0]
            coeff = x0.T[0]

            end = time.time()
            print(f"Updated coefficient obtained in {end - start}")

            # Update the fragment by modifying the coefficients of shared Pauli operators.
            last_var, meas_alloc, var, fixed_grp_coefficients, measured_groups = (
                get_split_measurement_variance_unconstrained(
                    coeff,
                    blissed_decomp,
                    sharable_paulis_list,
                    sharable_paulis_fixed_list,
                    sharable_only_no_fixed_decomp,
                    pw_grp_idxes_no_fix,
                    new_grp_idx_start,
                    H_decompose_copy,
                    N,
                    cisd_state,
                    Ne
                ))

            expectation_v = 0
            for fragment in measured_groups:
                expectation_v += expectation(gso(fragment, N), cisd_state)
            print(f"Expectation value after Shared Pauli: {expectation_v}")
            # assert np.isclose(expectation_v.real, original_exp.real,
            #                   atol=1E-3), "Expectation value shouldn't change"

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
