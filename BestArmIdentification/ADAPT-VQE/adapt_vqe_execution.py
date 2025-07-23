import numpy as np
import pickle
from openfermion import FermionOperator, QubitOperator, get_sparse_operator, expectation
from scipy.optimize import minimize
import sys
import os
from scipy.sparse.linalg import eigsh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StatePreparation.reference_state_utils import get_occ_no, get_reference_state
from utils.ferm_utils import ferm_to_qubit
from OperatorPools.generalized_fermionic_pool import get_all_anti_hermitian, get_all_uccsd_anti_hermitian
from BestArmIdentification.SuccessiveElimination.best_arm_identification import (
    successive_elimination_var_considered)


# --- Utility Functions ---
def commutator(A, B):
    return A * B - B * A

def compute_gradient(H_sparse, op_sparse, state):
    HP_psi = H_sparse.dot(op_sparse.dot(state))
    PH_psi = op_sparse.dot(H_sparse.dot(state))
    return np.vdot(state, HP_psi) - np.vdot(state, PH_psi)

def apply_unitary(state, op_sparse, theta):
    from scipy.sparse.linalg import expm_multiply
    return expm_multiply(theta * op_sparse, state) / np.linalg.norm(expm_multiply(theta * op_sparse, state))

def vqe_objective(params, ops_sparse, state, H_sparse):
    psi = state.copy()
    for theta, op in zip(params, ops_sparse):
        psi = apply_unitary(psi, op, theta)
    return np.real(np.vdot(psi, H_sparse.dot(psi)))

def direct_gradients_calc(H_sparse, state, pool_sparse):
    grads = [np.abs(compute_gradient(H_sparse, op, state)) for op in
             pool_sparse]
    max_grad = np.max(grads)
    best_idx = np.argmax(grads)
    return max_grad, best_idx


def adapt_vqe(H_ferm, n_electrons, max_iter=30, grad_tol=1e-6, verbose=True):
    # Convert to qubit Hamiltonian H_q
    H_qubit = ferm_to_qubit(H_ferm)
    n_qubits = max(idx for term in H_qubit.terms for idx, _ in term) + 1
    H_sparse = get_sparse_operator(H_qubit, n_qubits)

    # Prepare reference state (Hartree-Fock)
    ref_occ = get_occ_no('h4', n_qubits)
    state = get_reference_state(ref_occ, gs_format='wfs')

    energy = np.real(np.vdot(state, H_sparse.dot(state)))

    print("HF energy:", energy)

    # Build operator pool (Generalized anti-Hermitian)
    pool = list(get_all_uccsd_anti_hermitian(n_qubits, n_electrons))
    print(f"Pool size: {len(pool)}")
    pool_sparse = [get_sparse_operator(ferm_to_qubit(op), n_qubits) for op in pool]

    ansatz_ops = []
    params = []
    energies = []
    for iteration in range(max_iter):
        # Compute gradients for all pool operators
        # Use the Best arm strategy here.
        # best_idx, total_samples = successive_elimination_var_considered(arms, allocations)
        # max_grad = np.abs(compute_gradient(H_sparse, pool_sparse[best_idx], state))

        max_grad, best_idx = direct_gradients_calc(H_sparse, state, pool_sparse)

        if verbose:
            print(f"Iteration {iteration}: max gradient = {max_grad:.6e}, best idx = {best_idx}")
        # Add best operator to ansatz
        ansatz_ops.append(pool_sparse[best_idx])
        params.append(0.0)
        # VQE optimization
        def obj(x):
            return vqe_objective(x, ansatz_ops, state, H_sparse)
        res = minimize(obj, params, method='BFGS')
        params = list(res.x)
        # Update state
        psi = state.copy()
        for theta, op in zip(params, ansatz_ops):
            psi = apply_unitary(psi, op, theta)
        state = psi
        energy = np.real(np.vdot(state, H_sparse.dot(state)))
        energies.append(energy)
        if verbose:
            print(f"  Energy after iteration {iteration}: {energy:.8f}")
    return energies, params, ansatz_ops, state


def exact_ground_state_energy_and_vector(H_sparse):
    vals, vecs = eigsh(H_sparse, k=1, which='SA')
    return vals[0], vecs[:, 0]


if __name__ == "__main__":
    # Example: Load H4 Hamiltonian
    with open(os.path.join(os.path.dirname(__file__), '../../ham_lib/h4_sto-3g.pkl'), 'rb') as f:
        H_ferm = pickle.load(f)
    n_electrons = 4  # For H4
    energies, params, ansatz_ops, adapt_state = adapt_vqe(H_ferm, n_electrons)
    print("Final energy:", energies[-1])
    print("Parameters:", params)
    print(f"Ansatz depth: {len(ansatz_ops)}")

    # Compute and print exact ground state energy and vector
    H_qubit = ferm_to_qubit(H_ferm)
    n_qubits = max(idx for term in H_qubit.terms for idx, _ in term) + 1
    H_sparse = get_sparse_operator(H_qubit, n_qubits)
    exact_energy, exact_gs = exact_ground_state_energy_and_vector(H_sparse)
    print(f"Exact ground state energy (diagonalization): {exact_energy:.8f}")

    # Compute overlap (fidelity)
    overlap = np.abs(np.vdot(adapt_state, exact_gs)) ** 2
    print(f"Fidelity (|<ADAPT-VQE|Exact>|^2): {overlap:.8f}")
