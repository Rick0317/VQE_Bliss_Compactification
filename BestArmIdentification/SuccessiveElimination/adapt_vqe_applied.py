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
from best_arm_identification import best_arm_successive_elimination

def compute_gradient_expectation(H_sparse, pauli_sparse, wavefunction):
    """
    Compute <ψ|[H, P]|ψ> = <ψ|HP - PH|ψ>
    """
    # Compute HP|ψ> and PH|ψ>
    HP_psi = H_sparse.dot(pauli_sparse.dot(wavefunction))
    PH_psi = pauli_sparse.dot(H_sparse.dot(wavefunction))

    # Compute <ψ|[H,P]|ψ> = <ψ|HP|ψ> - <ψ|PH|ψ>
    gradient = np.vdot(wavefunction, HP_psi) - np.vdot(wavefunction, PH_psi)

    return gradient


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
    filename2 = f'../ham_lib/lih_fer.bin'
    with open(filename2, 'rb') as f:
        lih_hamiltonian = pickle.load(f)
    # with open('../ham_lib/h4_sto-3g.pkl', 'rb') as f:
    #     h4_hamiltonians = pickle.load(f)

    # Convert to qubit Hamiltonian
    H_qubit = ferm_to_qubit(lih_hamiltonian)


    # Determine the actual number of qubits by finding the maximum qubit index
    max_qubit_index = 0
    for term in H_qubit.terms:
        for qubit_index, pauli_op in term:
            max_qubit_index = max(max_qubit_index, qubit_index)

    n_qubits = max_qubit_index + 1  # Add 1 because indices are 0-based

    H_sparse = get_sparse_operator(H_qubit, n_qubits)

    print('n_qubits', n_qubits)
    n_electrons = 4

    ref_occ = get_occ_no('lih', n_qubits)
    print(f"Reference occupation for H4: {ref_occ}")

    # Get CISD ground state
    cisd_energy, cisd_wavefunction = get_cisd_gs('lih', H_qubit, n_qubits,
                                                 gs_format='wfs', tf='bk')

    hf_wavefunction = create_hf_state(ref_occ, n_qubits)
    theta = 0.2



    occupied = list(range(n_electrons))
    virtual = list(
        range(n_electrons, n_qubits))

    # Generate all valid combinations
    valid_combinations = []
    for i in occupied:
        for j in occupied:
            if i < j:  # Avoid double counting for occupied pairs
                for a in virtual:
                    for b in virtual:
                        if a < b:  # Avoid double counting for virtual pairs
                            valid_combinations.append((i, j, a, b))
                            if len(valid_combinations) >= 20:
                                break
                    if len(valid_combinations) >= 20:
                        break
                if len(valid_combinations) >= 20:
                    break
        if len(valid_combinations) >= 20:
            break
    for state in valid_combinations:
        i_rot_1, j_rot_1, a_rot_1, b_rot_1 = state
        print(f'{i_rot_1, j_rot_1, a_rot_1, b_rot_1}')
        # Create rotation generator and perturbed state
        rotation_gen_ferm = create_specific_ucc_generator(a_rot_1, b_rot_1, j_rot_1, i_rot_1)
        rotation_gen_qubit = ferm_to_qubit(rotation_gen_ferm)
        rotation_gen_sparse = get_sparse_operator(rotation_gen_qubit, n_qubits)
        perturbed_state = create_perturbed_state_ucc(hf_wavefunction,
                                                     rotation_gen_sparse, theta)

        reward_fns = []
        gradients = []


        noise_level = 0.0001


        for rot_idx, (i_rot, j_rot, a_rot, b_rot) in enumerate(valid_combinations):
            print("=== Started creating reward functions ===")

            rotation_gen_ferm = create_specific_ucc_generator(a_rot, b_rot, j_rot,
                                                              i_rot)
            rotation_gen_qubit = ferm_to_qubit(rotation_gen_ferm)

            rotation_gen_sparse = get_sparse_operator(rotation_gen_qubit, n_qubits)
            H_sparse = H_sparse.tocsr()
            G_sparse = rotation_gen_sparse.tocsr()

            commutator_sparse = H_sparse @ G_sparse - G_sparse @ H_sparse

            dense_matrix = commutator_sparse.toarray()

            eigenvalues, eigenvectors = np.linalg.eigh(dense_matrix)

            # Quantum State in the transformed basis.
            psi_in_eigenbasis = eigenvectors.conj().T @ perturbed_state

            probabilities = np.abs(psi_in_eigenbasis) ** 2

            grad = compute_gradient_expectation(H_sparse, rotation_gen_sparse, perturbed_state).real
            gradients.append(grad)
            mean = np.sum(eigenvalues * probabilities)
            print(f"Gradient: {grad}")
            print(f"Mean Eigenvalues for H4: {mean}")

            reward_fns.append(lambda eigv=eigenvalues, prob=probabilities: np.random.choice(eigv, p=prob))
            # samples = [lambda eigv=eigenvalues, prob=probabilities: np.random.choice(abs(eigv), p=prob) for _ in range(1000)]
            # empirical_mean = np.mean(samples)
            # print(f"Empirical mean: {empirical_mean}")

        with open("gradients.txt", "w") as f:
            f.write("Gradients:\n")
            for g in gradients:
                f.write(f"{g}\n")

        print(f"Gradients: {gradients}")
        magnitudes = np.abs(gradients)
        max_magnitude = np.max(magnitudes)
        max_indices = np.where(magnitudes == max_magnitude)[0]
        np.random.seed(10)
        best_arm = best_arm_successive_elimination(reward_fns, max_rounds=10000000, delta=0.05)
        print(f"Identified best arm: {best_arm}")
        print(f"Expected best: {max_indices}")

        with open("gradient_analysis_results_lih.txt", "a") as f:
            f.write(f"Identified best arm: {best_arm}\n")
            f.write(f"Expected best (max magnitude indices): {max_indices}\n")
            f.write(
                f"i_rot: {i_rot_1}, j_rot: {j_rot_1}, a_rot: {a_rot_1}, b_rot: {b_rot_1}\n")
            f.write("-" * 40 + "\n")  # Separator line










