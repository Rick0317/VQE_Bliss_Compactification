from openfermion import QubitOperator
from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator, expectation
import pickle
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StatePreparation.reference_state_utils import get_cisd_gs, get_occ_no, get_reference_state
from utils.ferm_utils import ferm_to_qubit
from scipy.sparse.linalg import eigsh
from best_arm_identification import best_arm_successive_elimination, successive_elimination_var_considered
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
            binary_index += 2**(n_qubits - 1 - i)

    # Create wavefunction vector
    hf_state = np.zeros(2**n_qubits)
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


def create_perturbed_state_ucc(hf_wavefunction, ucc_generator_sparse, theta=0.01):
    """
    Create a perturbed state using UCC rotation: |ψ⟩ = e^(θG)|ψ_CISD⟩
    """
    from scipy.sparse.linalg import expm_multiply

    print(f"Applying UCC rotation with θ = {theta} to HF")
    # Apply e^(θG) to the CISD state
    perturbed_state = expm_multiply(theta * ucc_generator_sparse, hf_wavefunction)

    # Normalize
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    return perturbed_state


if __name__ == "__main__":
    # Load the molecular Hamiltonian

    # filename2 = f'../ham_lib/lih_fer.bin'
    # with open(filename2, 'rb') as f:
    #     lih_hamiltonian = pickle.load(f)
    mol = 'h3st'
    with open('../../ham_lib/h3st_fer.bin', 'rb') as f:
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

    ref_occ = get_occ_no(mol, n_qubits)
    print(f"Reference occupation for {mol}: {ref_occ}")

    # Get CISD ground state
    cisd_energy, cisd_wavefunction = get_cisd_gs(mol, H_qubit, n_qubits,
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
    reward_fns = []
    allocations = []
    gradients = []
    expectation_values = []
    eigen_info = []
    allocation_info = []

    for rot_idx, (i_rot, j_rot, a_rot, b_rot) in enumerate(valid_combinations):
        print(f"=== Processing rotation {rot_idx}: ({i_rot}, {j_rot}, {a_rot}, {b_rot}) ===")

        fragments = []
        gradients_rot = []
        eigen_info_rot = []

        rotation_gen_ferm = create_specific_ucc_generator(a_rot, b_rot, j_rot, i_rot)
        rotation_gen_qubit = ferm_to_qubit(rotation_gen_ferm)

        commutator_qubit = H_qubit * rotation_gen_qubit - rotation_gen_qubit * H_qubit

        # # Decompose the Hamiltonian into fragments based on QWC
        decomposition = qwc_decomposition(commutator_qubit)
        total_fragments = len(decomposition)

        # with open(f"commutator_qubit{mol}.txt", "a") as f:
        #     f.write(mol)
        #     f.write("\n")
        #     f.write(f"UCCSD Generator indices i, j, a, b: {i_rot, j_rot, a_rot, b_rot}\n")
        #     f.write("\n")
        #     f.write(f"Total QWC fragments: {total_fragments} \n")
        #     f.write(f"Commutator (Arm): \n")
        #     f.write(str(decomposition))
        #     f.write("\n")
        #     f.write("-" * 50)
        #     f.write("\n")


        #
        # Calculate the allocation ratio based on the variance
        allocation = variance_based_allocation(cisd_wavefunction, decomposition, n_qubits)
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
            gradients_rot.append(grad)
            mean = np.sum(eigenvalues * probabilities)
            total_expectation += mean

            print(f"Gradient (real part), Mean: {grad}, {mean} and they are close: {np.isclose(grad, mean, rtol=1e-6)}")
            print("-" * 50)

            eigen_info_rot.append({
                'generator_idx': rot_idx,
                'fragment_idx': frag_index,
                'eigenvalues': eigenvalues,
                'probabilities': probabilities
            })

            # Each fragment has a probability distribution defined based on the
            # eigenvalues and the probabilities.
            fragments.append(lambda eigv=eigenvalues, prob=probabilities: np.random.choice(eigv, p=prob))
        reward_fns.append(fragments)
        allocations.append(allocation)
        expectation_values.append(total_expectation)
        gradients.extend(gradients_rot)
        eigen_info.extend(eigen_info_rot)
        allocation_info.append({
            'generator_idx': rot_idx,
            'allocation': allocation
        })
    # Write gradients to file
    with open(f"gradients_{mol}.txt", "w") as f:
        f.write("Gradients:\n")
        for g in gradients:
            f.write(f"{g}\n")

    np.savez(f"eigen_info_{mol}.npz", *eigen_info)
    np.savez(f"allocation_info_{mol}.npz", *allocation_info)

    magnitudes = np.abs(expectation_values)
    max_magnitude = np.max(magnitudes)
    max_indices = np.where(magnitudes == max_magnitude)[0]
