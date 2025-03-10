import pickle
from openfermion import (
normal_ordered,
get_majorana_operator,
bravyi_kitaev,
variance,
get_ground_state as ggs,
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
import os
import csv
from SharedPauli.coefficient_optimizer import (
get_split_measurement_variance_unconstrained,
get_meas_alloc,
optimize_coeffs_parallel

)
import time
from itertools import combinations

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


def prepare_superposition_state(n_qubits, n_particles, coefficients=None):
    """
    Prepare a superposition of quantum states with a fixed number of particles.

    Args:
        n_qubits (int): Number of qubits (fermionic modes).
        n_particles (int): Number of occupied particles.
        coefficients (list, optional): Coefficients for superposition. Defaults to equal superposition.

    Returns:
        np.ndarray: Normalized statevector representing the superposition.
    """
    # Generate all possible combinations of occupied orbitals
    occupied_combinations = list(combinations(range(n_qubits), n_particles))

    # Initialize the statevector
    statevector = np.zeros(2 ** n_qubits, dtype=complex)

    # If no coefficients are provided, use equal superposition
    if coefficients is None:
        coefficients = np.ones(len(occupied_combinations),
                               dtype=complex) / np.sqrt(
            len(occupied_combinations))

    # Construct the superposition state
    for i, occupied_orbitals in enumerate(occupied_combinations):
        basis_index = sum(2 ** q for q in
                          occupied_orbitals)  # Convert binary occupation to decimal index
        statevector[basis_index] += coefficients[i]  # Add weighted contribution

    # Normalize the statevector
    statevector /= np.linalg.norm(statevector)

    return statevector


def jordan_wigner_transform(state_fermion, num_qubits):
    """
    Transforms a fermionic state vector to the qubit state vector
    using the Jordan-Wigner transformation.

    :param state_fermion: The state vector in fermionic basis (numpy array)
    :param num_qubits: The number of qubits in the system
    :return: Transformed state vector in qubit basis
    """
    state_qubit = np.copy(state_fermion)

    # Perform the Jordan-Wigner transformation by applying the appropriate phase factors
    for i in range(num_qubits):
        # Apply phase factor (-1)^k to the state to simulate the fermionic anticommutation relations
        for j in range(i):
            state_qubit = np.roll(state_qubit,
                                  2 ** j)  # shifting the state vector
            state_qubit = np.multiply(state_qubit,
                                      (-1) ** j)  # applying the (-1)^j factor

    return state_qubit


if __name__ == '__main__':
    N = 8
    Ne = 4
    mol_name = "H2O"

    filename = f'../SolvableQubitHamiltonians/ham_lib/h4_sto-3g.pkl'

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

    # System parameters
    num_spin_orbitals = N  # Must be even
    num_particles = Ne  # Total number of particles (even to allow S_z = 0)
    num_up = num_particles // 2  # Half goes to spin-up
    num_down = num_particles // 2  # Half goes to spin-down

    # Generate basis states separately for spin-up (even indices) and spin-down (odd indices)
    spin_up_basis = list(
        combinations(range(0, num_spin_orbitals, 2), num_up))  # Even indices
    spin_down_basis = list(
        combinations(range(1, num_spin_orbitals, 2), num_down))  # Odd indices

    # Generate all valid Fock basis states with equal up and down electrons
    basis_states = [up + down for up in spin_up_basis for down in
                    spin_down_basis]

    # Initialize a zero state
    state_vector = np.zeros(2 ** num_spin_orbitals, dtype=complex)

    # Assign random amplitudes to valid states
    for basis in basis_states:
        index = sum(
            1 << i for i in basis)  # Convert bit positions to decimal index
        state_vector[index] = np.random.rand() + 1j * np.random.rand()

    # Normalize the state
    state_vector /= np.linalg.norm(state_vector)

    ground_state = state_vector

    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    sparse_matrix = gso(total_number_operator, N)

    # Step 3: Convert to a dense matrix (optional)
    dense_matrix = sparse_matrix.toarray()

    particle_number = np.real(ground_state.conj().T @ dense_matrix @ ground_state)
    print(f"Particle Number: {particle_number}")

    sz_sparse = gso(sz_operator(num_spin_orbitals), N)

    sz_in_spin_orbitals = sz_sparse.toarray()

    sz_value = np.real(
        ground_state.conj().T @ sz_in_spin_orbitals @ ground_state)
    print(f"Sz: {sz_value}")



    for i in range(len(anti_com_list)):
        G = anti_com_list[i]
        hamil_copy1 = copy_ferm_hamiltonian(ordered_hamil)
        hamil_copy2 = copy_ferm_hamiltonian(ordered_hamil)
        g_copy1 = copy_ferm_hamiltonian(G)
        g_copy2 = copy_ferm_hamiltonian(G)

        commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2
        H = normal_ordered(commutator)
        H_in_q = ferm_to_qubit(H)

        # <[H, A]>

        ground_state = ggs(gso(H_in_q, N))[1]
        ground_state_ferm = ggs(gso(H, N))[1]

        # Define the total number operator N = sum_i a_i† a_i
        number_operator = FermionOperator()
        for i in range(N):
            number_operator += FermionOperator(f"{i}^ {i}")  # a_i† a_i

        # Convert to qubit operator using Jordan-Wigner transformation
        number_operator_qubit = bravyi_kitaev(number_operator)

        print(f"Particle number: {expectation(gso(number_operator_qubit), ground_state)}")
        print(
            f"Particle number: {expectation(gso(number_operator), ground_state_ferm)}")

        print(f"Sz Qubit: {expectation(gso(ferm_to_qubit(sz_operator(num_spin_orbitals)), N), ground_state)}")
        print(f"Sz Ferm: {expectation(gso(sz_operator(num_spin_orbitals), N), ground_state_ferm)}")

        original_exp = expectation(gso(H, N), ground_state_ferm)
        print(f"Original Expectation value: {original_exp}")

        original_exp = expectation(gso(H_in_q, N), ground_state)
        print(f"Original Expectation value2: {original_exp}")

        two_body_h = FermionOperator()
        three_body_h = FermionOperator()

        two_body_list = filter_indices(H, N, Ne)
        if len(two_body_list) != 0:
            majo = get_majorana_operator(two_body_h)

            one_norm = 0
            for term, coeff in majo.terms.items():
                if term != ():
                    one_norm += abs(coeff)

            print("Original 1-Norm 2-body", one_norm)

            majo = get_majorana_operator(three_body_h)

            one_norm = 0
            for term, coeff in majo.terms.items():
                if term != ():
                    one_norm += abs(coeff)

            print("Original 1-Norm 3-body", one_norm)


            copied_H = copy_ferm_hamiltonian(H)
            H_q = ferm_to_qubit(copied_H)
            print("Commutator Obtained in Fermion and Qubit space")
            H_copy = copy_hamiltonian(H_q)
            # decomp = sorted_insertion_decomposition(H_copy, methodtag)
            decomp = qwc_decomposition(H_copy)
            print("Original Decomposition Complete")
            original_var = commutator_variance(H_q, decomp, N, ground_state)
            print(f"Original variance: {original_var}")

            majo = get_majorana_operator(H)

            one_norm = 0
            for term, coeff in majo.terms.items():
                if term != ():
                    one_norm += abs(coeff)

            print("Original 1-Norm", one_norm)

            copied_H2 = copy_ferm_hamiltonian(H)
            optimization_wrapper, initial_guess = optimize_bliss_mu3_customizable(copied_H2, N, Ne, two_body_list)

            # res = minimize(optimization_wrapper, initial_guess, method='BFGS',
            #                options={'gtol': 1e-300, 'disp': True, 'maxiter': 600, 'eps': 1e-2})
            # Try a different method that doesn't require gradients
            res = minimize(optimization_wrapper, initial_guess, method='Powell',
                           options={'disp': True, 'maxiter': 100000})

            H_before_modification = copy_ferm_hamiltonian(H)
            bliss_output, killer = construct_H_bliss_mu3_customizable(H_before_modification, res.x, N, Ne, two_body_list)

            bliss_exp = expectation(gso(bliss_output, N), ground_state)
            print(f"BLISS Expectation value: {bliss_exp}")

            bliss_exp = expectation(gso(ferm_to_qubit(bliss_output), N), ground_state)
            print(f"BLISS Expectation value: {bliss_exp}")

            print(f"First 5 parameters: {res.x[:5]}")

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

            # blissed_decomp = sorted_insertion_decomposition(H_bliss_copy, methodtag)
            blissed_decomp = qwc_decomposition(H_bliss_copy)
            print("Blissed Decomposition Complete")

            H_decompose_copy = copy_hamiltonian(H_bliss_q)

            blissed_vars = commutator_variance(H_decompose_copy,
                                               blissed_decomp.copy(), N, ground_state)

            print(f"Blissed variance: {blissed_vars}")

            # # Apply Ghost Pauli method
            psi = ggs(gso(H_bliss_copy, N))[1]
            # psi = ground_state
            # bliss_ghost_decomp = update_decomp_w_ghost_paulis(psi, N, blissed_decomp)
            # new_H_q = QubitOperator()
            # for fragment in bliss_ghost_decomp:
            #     new_H_q += fragment
            #
            # blissed_expectation = expectation(gso(new_H_q), psi)
            # print(f"Blissed ghost expectation: {blissed_expectation}")
            #
            # blissed_ghost_vars = commutator_variance(new_H_q, bliss_ghost_decomp, N, Ne)
            # print(f"Blissed Ghost variance: {blissed_ghost_vars}")

            # Shared Pauli

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

            pw_grp_idxes_no_fix_len = get_pw_grp_idxes_no_fix_len(pw_grp_idxes_no_fix)

            # 3
            all_sharable_contained_decomp, all_sharable_no_fixed_decomp = get_overlapping_decomp(
                sharable_paulis_dict, blissed_decomp, pw_grp_idxes_fix)

            # 4
            sharable_only_decomp, sharable_only_no_fixed_decomp = get_sharable_only_decomp(
                sharable_paulis_dict, blissed_decomp, pw_grp_idxes_fix, coeff_map)

            # 5
            fixed_grp, all_sharable_contained_no_fixed_decomp, new_sharable_pauli_indices_list, new_grp_len_list, new_grp_idx_start \
                = get_coefficient_orderings(sharable_only_decomp, sharable_paulis_list,
                                            sharable_pauli_indices_list, coeff_map)

            # 6
            pw_indices = get_all_pw_indices(sharable_paulis_list,
                                            sharable_only_no_fixed_decomp,
                                            pw_grp_idxes_no_fix, new_grp_idx_start)

            meas_alloc = get_meas_alloc(blissed_decomp, N, psi)

            # Get the linear equation for finding the gradient descent direction
            matrix, b = optimize_coeffs_parallel(
                pw_grp_idxes_fix,
                pw_grp_idxes_no_fix_len,
                meas_alloc,
                sharable_only_decomp,
                sharable_only_no_fixed_decomp,
                pw_indices, psi, N, blissed_decomp, alpha=0.001)
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
                    ground_state,
                    Ne
                ))

            expectation_v = 0
            for fragment in measured_groups:
                expectation_v += expectation(gso(fragment, N), ground_state)
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
                        ['As', 'Original Variance', 'BLISS variance', 'Shared Pauli'])

                # Write the data
                writer.writerow(
                    [com_name_list[i], original_var, blissed_vars, last_var])
