from openfermion import QubitOperator
from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator, \
    expectation
import pickle
import sys
import os
import numpy as np
import json
import multiprocessing as mp
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StatePreparation.reference_state_utils import get_cisd_gs, get_occ_no, \
    get_reference_state
from utils.ferm_utils import ferm_to_qubit
from scipy.sparse.linalg import eigsh
from best_arm_identification import best_arm_successive_elimination, \
    successive_elimination_var_considered
from Decompositions.qwc_decomposition import qwc_decomposition
from sample_test import sample_test


def compute_gradient_expectation(frag_sparse, wavefunction):
    """
    Compute <ψ|A|ψ> where A is the operator (commutator fragment)
    Returns the real part of the expectation value
    """
    # Compute <ψ|A|ψ> = ψ† A ψ
    expectation = np.vdot(wavefunction, frag_sparse.dot(wavefunction))

    # Return the real part
    return expectation.real


def create_hf_state(ref_occ, n_qubits):
    """
    Create Hartree-Fock reference state directly from occupation string
    """
    # Convert occupation string to binary index
    binary_index = 0
    for i, occ in enumerate(ref_occ):
        if occ == '1':
            binary_index += 2 ** (n_qubits - 1 - i)

    # Create wavefunction vector
    hf_state = np.zeros(2 ** n_qubits)
    hf_state[binary_index] = 1.0

    return hf_state


def create_specific_ucc_generator(a, b, j, i):
    double_excitation = FermionOperator(f'{a}^ {b}^ {j} {i}', 1.0)
    # T2† term: a†_i a†_j a_b a_a
    double_excitation_dagger = FermionOperator(f'{i}^ {j}^ {b} {a}', 1.0)
    return double_excitation - double_excitation_dagger


def variance_based_allocation(hf_state, fragments, n_qubits):
    """
    Compute the variance based allocation. Allocation adds up to 1.
    :param hf_state: Hartree-Fock state
    :param fragments: QubitOperator that represents a list of fragments
    :return:
    """
    variance_list = []
    sum = 0
    for fragment in fragments:
        commutator_original_sparse = get_sparse_operator(fragment, n_qubits)

        dense_matrix = commutator_original_sparse.toarray()

        square = dense_matrix @ dense_matrix

        square_expectation = np.vdot(hf_state, square @ hf_state).real
        expectation = np.vdot(hf_state, dense_matrix @ hf_state).real

        variance = square_expectation - expectation ** 2
        sum += variance
        variance_list.append(variance)

    return np.array(variance_list) / sum


def create_perturbed_state_ucc(hf_wavefunction, ucc_generator_sparse,
                               theta=0.01):
    """
    Create a perturbed state using UCC rotation: |ψ⟩ = e^(θG)|ψ_CISD⟩
    """
    from scipy.sparse.linalg import expm_multiply

    print(f"Applying UCC rotation with θ = {theta} to HF")
    # Apply e^(θG) to the CISD state
    perturbed_state = expm_multiply(theta * ucc_generator_sparse,
                                    hf_wavefunction)

    # Normalize
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    return perturbed_state


def process_single_rotation(args):
    """
    Process a single rotation and save data directly to JSON files
    """
    rot_idx, (i_rot, j_rot, a_rot,
              b_rot), H_qubit, cisd_wavefunction, n_qubits, mol = args

    print(
        f"=== Processing rotation {rot_idx}: ({i_rot}, {j_rot}, {a_rot}, {b_rot}) ===")

    eigen_info_rot = []

    rotation_gen_ferm = create_specific_ucc_generator(a_rot, b_rot, j_rot,
                                                      i_rot)
    rotation_gen_qubit = ferm_to_qubit(rotation_gen_ferm)

    commutator_qubit = H_qubit * rotation_gen_qubit - rotation_gen_qubit * H_qubit

    # Decompose the Hamiltonian into fragments based on QWC
    decomposition = qwc_decomposition(commutator_qubit)
    total_fragments = len(decomposition)

    # Calculate the allocation ratio based on the variance
    allocation = variance_based_allocation(cisd_wavefunction, decomposition,
                                           n_qubits)
    print(f'Total fragments: {total_fragments}')
    print(f'Allocation: {allocation}')

    total_expectation = 0

    for frag_index, fragment_q in enumerate(decomposition):
        commutator_original_sparse = get_sparse_operator(fragment_q, n_qubits)

        # Use the same matrix for both calculations
        dense_matrix = commutator_original_sparse.toarray()

        eigenvalues, eigenvectors = np.linalg.eigh(dense_matrix)

        psi_in_eigenbasis = eigenvectors.conj().T @ cisd_wavefunction

        probabilities = np.abs(psi_in_eigenbasis) ** 2

        # Use the same dense matrix for direct calculation to ensure consistency
        grad = np.vdot(cisd_wavefunction, dense_matrix @ cisd_wavefunction).real
        mean = np.sum(eigenvalues * probabilities)
        total_expectation += mean

        print(
            f"Gradient (real part), Mean: {grad}, {mean} and they are close: {np.isclose(grad, mean, rtol=1e-6)}")
        print("-" * 50)

        eigen_info_rot.append({
            'generator_idx': rot_idx,
            'fragment_idx': frag_index,
            'eigenvalues': eigenvalues.tolist(),
            # Convert to list for JSON serialization
            'probabilities': probabilities.tolist()
            # Convert to list for JSON serialization
        })

    # Save eigen_info for this rotation
    eigen_filename = f"eigen_info_{mol}_rot_{rot_idx}.json"
    with open(eigen_filename, 'w') as f:
        json.dump(eigen_info_rot, f, indent=2)

    # Save allocation_info for this rotation
    allocation_filename = f"allocation_info_{mol}_rot_{rot_idx}.json"
    allocation_data = {
        'generator_idx': rot_idx,
        'allocation': allocation.tolist()
        # Convert to list for JSON serialization
    }
    with open(allocation_filename, 'w') as f:
        json.dump(allocation_data, f, indent=2)

    return rot_idx, eigen_filename, allocation_filename


def combine_json_files(results, mol):
    """
    Combine individual JSON files into final combined files
    """
    all_eigen_info = []
    all_allocation_info = []

    # Sort results by rotation index to maintain order
    sorted_results = sorted(results, key=lambda x: x[0])

    for rot_idx, eigen_filename, allocation_filename in sorted_results:
        with open(eigen_filename, 'r') as f:
            eigen_info_rot = json.load(f)
        with open(allocation_filename, 'r') as f:
            allocation_data = json.load(f)

        all_eigen_info.extend(eigen_info_rot)
        all_allocation_info.append(allocation_data)

    # Save combined data to JSON files
    with open(f"eigen_info_{mol}.json", 'w') as f:
        json.dump(all_eigen_info, f, indent=2)

    with open(f"allocation_info_{mol}.json", 'w') as f:
        json.dump(all_allocation_info, f, indent=2)

    print(f"Data saved to eigen_info_{mol}.json and allocation_info_{mol}.json")

    # Clean up individual rotation files
    for rot_idx, eigen_filename, allocation_filename in results:
        os.remove(eigen_filename)
        os.remove(allocation_filename)

    print("Individual rotation files cleaned up")


if __name__ == "__main__":
    # Load the molecular Hamiltonian

    # filename2 = f'../ham_lib/lih_fer.bin'
    # with open(filename2, 'rb') as f:
    #     lih_hamiltonian = pickle.load(f)
    mol = 'h4st2'
    with open('../../ham_lib/h4st2_fer.bin', 'rb') as f:
        lih_hamiltonians = pickle.load(f)

    # Convert to qubit Hamiltonian
    H_qubit = ferm_to_qubit(lih_hamiltonians)

    # Determine the actual number of qubits by finding the maximum qubit index
    max_qubit_index = 0
    for term in H_qubit.terms:
        for qubit_index, pauli_op in term:
            max_qubit_index = max(max_qubit_index, qubit_index)

    # Add 1 because indices are 0-based
    n_qubits = max_qubit_index + 1

    print('n_qubits', n_qubits)
    n_electrons = 4

    ref_occ = get_occ_no('h4', n_qubits)
    print(f"Reference occupation for {mol}: {ref_occ}")

    # Get CISD ground state
    cisd_energy, cisd_wavefunction = get_cisd_gs('h4', H_qubit, n_qubits,
                                                 gs_format='wfs', tf='bk')

    # Get the HF state
    hf_wavefunction = create_hf_state(ref_occ, n_qubits)
    theta = 0.2

    occupied = list(range(n_electrons))
    virtual = list(
        range(n_electrons, n_qubits))

    # Generate all valid combinations of indices for the generators
    # Stop generating more than MAX_COUNTS
    MAX_COUNTS = 36
    valid_combinations = []
    for i in occupied:
        for j in occupied:
            if i < j:
                for a in virtual:
                    for b in virtual:
                        if a < b:  # Avoid double counting for virtual pairs
                            valid_combinations.append((i, j, a, b))
                            if len(valid_combinations) >= MAX_COUNTS:
                                break
                    if len(valid_combinations) >= MAX_COUNTS:
                        break
                if len(valid_combinations) >= MAX_COUNTS:
                    break
        if len(valid_combinations) >= MAX_COUNTS:
            break

    # Main loop to generate reward functions for different rotation combinations

    # Use multiprocessing for parallel processing of rotations
    try:
        # Use a conservative number of processes to avoid memory issues
        n_processes = min(mp.cpu_count(), 4)  # Limit to 4 processes max
        print(f"Using {n_processes} processes for parallel computation")

        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(process_single_rotation, [
                (rot_idx, (i_rot, j_rot, a_rot, b_rot), H_qubit,
                 cisd_wavefunction, n_qubits, mol)
                for rot_idx, (i_rot, j_rot, a_rot, b_rot) in
                enumerate(valid_combinations)
            ])

        # Combine results and save them
        combine_json_files(results, mol)

    except Exception as e:
        print(f"Multiprocessing failed: {e}")
        print("Falling back to sequential processing...")

        # Fallback to sequential processing
        results = []
        for rot_idx, (i_rot, j_rot, a_rot, b_rot) in enumerate(
                valid_combinations):
            result = process_single_rotation((rot_idx,
                                              (i_rot, j_rot, a_rot, b_rot),
                                              H_qubit, cisd_wavefunction,
                                              n_qubits, mol))
            results.append(result)

        combine_json_files(results, mol)





