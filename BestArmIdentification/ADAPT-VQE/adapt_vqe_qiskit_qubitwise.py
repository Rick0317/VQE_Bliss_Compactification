import numpy as np
from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import Aer, AerSimulator
from qiskit.quantum_info import Statevector, Operator, SparsePauliOp
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize
from scipy.linalg import expm
import pickle
import os
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import csv
from datetime import datetime
import json

# --- OpenFermion imports for chemistry mapping ---
from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator

# Add path to import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OperatorPools.generalized_fermionic_pool import get_all_uccsd_anti_hermitian, get_all_anti_hermitian
from utils.ferm_utils import ferm_to_qubit
from SolvableQubitHamiltonians.fermi_util import get_commutator
from Decompositions.qwc_decomposition import qwc_decomposition
from StatePreparation.reference_state_utils import get_reference_state, get_occ_no
from adapt_vqe_qiskit import (create_ansatz_circuit, measure_expectation,
                              get_statevector, openfermion_qubitop_to_sparsepauliop,
                              fermion_operator_to_qiskit_operator, exact_ground_state_energy, save_results_to_csv)

def pauli_commute(p1, p2):
    # Returns True if p1 and p2 commute on a single qubit
    if p1 == 'I' or p2 == 'I' or p1 == p2:
        return True
    # Check anti-commuting pairs
    anti_commuting_pairs = {('X', 'Y'), ('Y', 'X'), ('X', 'Z'), ('Z', 'X'), ('Y', 'Z'), ('Z', 'Y')}
    return (p1, p2) not in anti_commuting_pairs


def does_commute(pauli_string1, pauli_string2):
    if len(pauli_string1) != len(pauli_string2):
        raise ValueError("Pauli strings must have the same length")
    return all(pauli_commute(p1, p2) for p1, p2 in zip(pauli_string1, pauli_string2))


def does_diagonalize(group_op, key_unitary):
    """
    We check if key_unitary diagonalizes all the Pauli Strings in group_op
    :param group_op: The QWC group operator
    :param key_unitary: The diagonalizing unitary in the form of 'IXYI' where X and Y are rotations to diagonalize X and Y respectively
    :return:
    """
    for pauli_string, coeff in group_op.to_list():
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X' and key_unitary[i] != 'X':
                return False
            elif pauli == 'Y' and key_unitary[i] != 'Y':
                return False
            elif pauli == 'Z' and key_unitary[i] != 'I':
                return False
    return True


def get_diagonalizing_pauli(group_op):
    """
    Given a QWC group operator, find the diagonalizing Pauli String that diagonalizes all the Pauli Strings in group_op
    :param group_op: SparsePauliOp representing a QWC group
    :return: Pauli string indicating diagonalizing rotations ('X' for Ry, 'Y' for Rx, 'I' for no rotation)
    """
    # Get all Pauli strings from the group
    pauli_list = group_op.to_list()
    if not pauli_list:
        return 'I'  # Empty group

    # Get the length of Pauli strings (number of qubits)
    n_qubits = len(pauli_list[0][0])

    # Initialize diagonalizing string with all identities
    diagonalizing_pauli = ['I'] * n_qubits

    # For each qubit position, determine what rotation is needed
    for pauli_string, coeff in pauli_list:
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                # X basis requires Ry(-π/2) rotation, represented as 'X'
                diagonalizing_pauli[i] = 'X'
            elif pauli == 'Y':
                # Y basis requires Rx(π/2) rotation, represented as 'Y'
                diagonalizing_pauli[i] = 'Y'
            # 'Z' and 'I' don't require rotations, so keep as 'I'

    return ''.join(diagonalizing_pauli)


def get_qubit_w_commutation_group(H_fermion, pool_fermion, n_qubits):
    """
    Categorizes each QWC group and express each fragment with the categories
    :param H_fermion: Hamiltonian in Fermion Operator
    :param pool_fermion: The pool of fermion generators
    :param n_qubits: The number of qubits
    :return:
    """
    # {Diagonalizing Unitary of a QWC group: i}
    fragment_group_indices_map = {}

    # {ferm_op: [1, 2, 3, ... ]}
    commutator_indices_map = {}

    for i, ferm_op in enumerate(pool_fermion):
        commutator_indices_map[i] = []
        # Compute fermion commutator [H, G]
        commutator_fermion = get_commutator(H_fermion, ferm_op)

        # Convert to qubit operator
        commutator_qubit = ferm_to_qubit(commutator_fermion)

        # Decompose into Pauli groups
        pauli_groups = qwc_decomposition(commutator_qubit)

        for group in pauli_groups:
            # Check which unitary diagonalizes the group.
            group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

            # Check if pauli_string commutes with any existing unitary
            found_commuting_group = False
            for key_unitary, group_index in fragment_group_indices_map.items():
                # We use does_diagonalize to check if the diagonalizing unitary
                # diagonalizes all the pauli_string in the group
                if does_diagonalize(group_op, key_unitary):
                    # If the group already exists, add the index of the existing QWC group
                    commutator_indices_map[i].append(group_index)
                    found_commuting_group = True
                    break

            # If no commuting group found, create a new group
            if not found_commuting_group:
                diagonalizing_pauli = get_diagonalizing_pauli(group_op)
                new_group_index = len(fragment_group_indices_map)
                fragment_group_indices_map[diagonalizing_pauli] = new_group_index
                commutator_indices_map[i].append(new_group_index)

    return fragment_group_indices_map, commutator_indices_map


def save_commutator_maps(fragment_group_indices_map, commutator_indices_map, filename):
    """Save commutator maps to a JSON file"""
    try:
        # Convert all keys to strings for JSON serialization
        data = {
            'fragment_group_indices_map': {str(k): v for k, v in fragment_group_indices_map.items()},
            'commutator_indices_map': {str(k): v for k, v in commutator_indices_map.items()}
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Commutator maps saved to {filename}")

    except Exception as e:
        print(f"Error saving commutator maps: {e}")


def load_commutator_maps(filename):
    """Load commutator maps from a JSON file"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        # Convert string keys back to appropriate types
        fragment_group_indices_map = {k: v for k, v in data['fragment_group_indices_map'].items()}
        commutator_indices_map = {int(k): v for k, v in data['commutator_indices_map'].items()}

        print(f"Commutator maps loaded from {filename}")
        return fragment_group_indices_map, commutator_indices_map

    except FileNotFoundError:
        print(f"Commutator maps file {filename} not found")
        return None, None
    except Exception as e:
        print(f"Error loading commutator maps: {e}")
        return None, None


def generate_commutator_cache_filename(molecule_name, n_qubits, n_electrons, pool_size):
    """Generate a cache filename based on molecular system parameters"""
    return f"commutator_maps_{molecule_name}_{n_qubits}q_{n_electrons}e_{pool_size}pool.json"


def get_commutator_maps_cached(H_fermion, pool_fermion, n_qubits, molecule_name='h4', n_electrons=4, cache_file=None):
    """
    Get commutator maps with caching support

    Args:
        H_fermion: Hamiltonian fermion operator
        pool_fermion: List of fermion operators in the pool
        n_qubits: Number of qubits
        molecule_name: Name of molecule (for caching)
        n_electrons: Number of electrons (for caching)
        cache_file: Optional filename to cache/load commutator maps

    Returns:
        tuple: (fragment_group_indices_map, commutator_indices_map)
    """
    if cache_file is not None:
        # Try to load from cache first
        fragment_map, commutator_map = load_commutator_maps(cache_file)
        if fragment_map is not None and commutator_map is not None:
            # Verify the cache matches current pool size
            if len(commutator_map) == len(pool_fermion):
                return fragment_map, commutator_map
            else:
                print(f"Cache size mismatch: expected {len(pool_fermion)}, got {len(commutator_map)}")

    # Compute commutator maps if not cached or cache invalid
    print("Computing commutator maps...")
    fragment_map, commutator_map = get_qubit_w_commutation_group(H_fermion, pool_fermion, n_qubits)

    # Save to cache if filename provided
    if cache_file is not None:
        save_commutator_maps(fragment_map, commutator_map, cache_file)

    return fragment_map, commutator_map


def _measure_single_fragment(args):
    """Worker function for parallel measurement of a single fragment"""
    current_circuit, pauli_string, fragment_index, shots = args

    try:
        # Create measurement circuit
        meas_circuit = current_circuit.copy()
        meas_circuit.add_register(ClassicalRegister(len(pauli_string), 'c'))

        # Apply basis rotations for measurement
        # Note: Qiskit Pauli strings use reverse indexing (leftmost = highest qubit)
        n_qubits = len(pauli_string)
        for i, pauli in enumerate(pauli_string):
            qubit_idx = n_qubits - 1 - i  # Reverse the qubit indexing
            if pauli == 'X':
                meas_circuit.ry(-np.pi / 2, qubit_idx)
            elif pauli == 'Y':
                meas_circuit.rx(np.pi / 2, qubit_idx)
            # Z measurement requires no rotation

        # Add measurements
        for i in range(len(pauli_string)):
            meas_circuit.measure(i, i)

        # Run simulation
        simulator = AerSimulator()
        compiled_circuit = transpile(meas_circuit, simulator)
        result = simulator.run(compiled_circuit, shots=shots).result()
        counts = result.get_counts()

        return fragment_index, counts

    except Exception as e:
        print(f"Error measuring fragment {fragment_index} with Pauli string {pauli_string}: {e}")
        return fragment_index, {}


def get_counts_for_each_fragment(current_circuit, fragment_group_indices_map, shots=8192):
    """
    Compute the distribution of the wavefunction from the current circuit (parallelized version)
    :param current_circuit: The present form of the parametrized circuit
    :param fragment_group_indices_map: The map of each QWC to indices
    :param shots: The number of measurements
    :return: {1: {counts data}, 2: {counts data}, ... }
    """
    # Prepare arguments for parallel execution
    args_list = [(current_circuit, pauli_string, fragment_index, shots)
                 for pauli_string, fragment_index in fragment_group_indices_map.items()]

    # Use multiprocessing to parallelize measurements
    num_processes = min(cpu_count(), len(args_list))  # Don't use more processes than fragments

    counts_for_each_fragment = {}

    if len(args_list) == 1:
        # If only one fragment, don't use multiprocessing overhead
        fragment_index, counts = _measure_single_fragment(args_list[0])
        counts_for_each_fragment[fragment_index] = counts
    else:
        # Use multiprocessing for multiple fragments
        try:
            with Pool(num_processes) as p:
                results = p.map(_measure_single_fragment, args_list)

            # Collect results
            for fragment_index, counts in results:
                counts_for_each_fragment[fragment_index] = counts

        except Exception as e:
            print(f"Multiprocessing failed ({e}), falling back to sequential processing")
            # Fallback to sequential processing
            for args in args_list:
                fragment_index, counts = _measure_single_fragment(args)
                counts_for_each_fragment[fragment_index] = counts

    return counts_for_each_fragment


def compute_commutator_gradient(H_fermion, generator_ferm, fragment_indices, counts_for_each_fragment, n_qubits, shots=8192):
    """
    Compute the energy gradient of the generator
    :param H_fermion: Fermion Operator of the Hamiltonian
    :param generator_ferm: Fermion Operator of the generator
    :param fragment_indices: The QWC fragments indices of the decomposed [H, G]
    :param counts_for_each_fragment: The measurement information of the state for each fragment
    :return:
    """
    commutator_fermion = get_commutator(H_fermion, generator_ferm)

    # Convert to qubit operator
    commutator_qubit = ferm_to_qubit(commutator_fermion)

    # Decompose into QWC groups
    pauli_groups = qwc_decomposition(commutator_qubit)
    total_expectation = 0.0
    for i, group in enumerate(pauli_groups):
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
        fragment_group_index = fragment_indices[i]
        counts = counts_for_each_fragment[fragment_group_index]
        sub_total_expectation = 0.0
        for i, (pauli_string, coeff) in enumerate(group_op.to_list()):
            expectation = 0.0
            for bitstring, count in counts.items():
                # Calculate parity
                parity = 1
                for i, pauli in enumerate(pauli_string):
                    if pauli != 'I' and bitstring[i] == '1':
                        parity *= -1
                expectation += parity * count / shots

            sub_total_expectation += coeff * expectation

        total_expectation += sub_total_expectation

    return total_expectation



def adapt_vqe_qiskit(H_qubit_op, n_qubits, n_electrons, pool, H_fermion, pool_fermion, fragment_group_indices_map, commutator_indices_map, max_iter=30, grad_tol=5e-2, verbose=True, use_multiprocessing=True):
    """
    ADAPT-VQE algorithm with multiprocessing support for gradient computation.

    Returns:
        tuple: (energies, params, ansatz_ops, final_state, total_measurements)
    """

    # Prepare reference state
    ansatz_ops = []
    params = []
    energies = []
    total_measurements = 0  # Track total measurements throughout the process

    final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops,
                                          params)
    energy = measure_expectation(final_circuit, H_qubit_op)
    print(f"HF energy (Qiskit): {energy}")

    for iteration in range(max_iter):
        # Create current ansatz circuit
        print(f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')
        current_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)

        # Compute gradients for all pool operators using commutator measurement
        grads = []
        shots = 1024
        if verbose:
            print(f"Computing gradients for {len(pool)} operators...")

        counts_for_each_fragment = get_counts_for_each_fragment(current_circuit, fragment_group_indices_map, shots=shots)

        # Sequential processing fallback
        grads = []
        iteration_measurements = shots * len(counts_for_each_fragment)
        total_measurements += iteration_measurements

        if verbose:
            print(f"  Measurements this iteration: {iteration_measurements}")

        # Parallelize gradient calculation
        def _compute_single_gradient(args):
            """Worker function for parallel gradient computation"""
            i, H_fermion, ferm_op, fragment_indices, counts_for_each_fragment, n_qubits, shots = args
            try:
                grad = compute_commutator_gradient(H_fermion, ferm_op, fragment_indices, counts_for_each_fragment, n_qubits, shots)
                return i, np.abs(grad)
            except Exception as e:
                print(f"Error computing gradient for operator {i}: {e}")
                return i, 0.0

        # Prepare arguments for parallel execution
        args_list = [(i, H_fermion, ferm_op, commutator_indices_map[i], counts_for_each_fragment, n_qubits, shots)
                     for i, ferm_op in enumerate(pool_fermion)]

        num_processes = min(cpu_count(), len(args_list))
        grads = [0.0] * len(pool_fermion)  # Pre-allocate with correct size

        if len(args_list) == 1:
            # If only one gradient, don't use multiprocessing overhead
            i, grad = _compute_single_gradient(args_list[0])
            grads[i] = grad
            if verbose:
                print(f"  Gradient {i+1}/{len(pool)}: {grad:.6e}")
        else:
            # Use multiprocessing for multiple gradients
            try:
                with Pool(num_processes) as p:
                    results = p.map(_compute_single_gradient, args_list)

                # Collect results and maintain order
                for i, grad in results:
                    grads[i] = grad

                if verbose:
                    # Print some gradient values for debugging
                    for i, grad in enumerate(grads):
                        if i % 10 == 0:
                            print(f"  Gradient {i+1}/{len(pool)}: {grad:.6e}")

            except Exception as e:
                print(f"Gradient multiprocessing failed ({e}), falling back to sequential processing")
                # Fallback to sequential processing
                for args in args_list:
                    i, grad = _compute_single_gradient(args)
                    grads[i] = grad
                    if verbose and i % 10 == 0:
                        print(f"  Gradient {i+1}/{len(pool)}: {grad:.6e}")

        max_grad = np.max(grads)
        best_idx = np.argmax(grads)

        if verbose:
            print(f"Iteration {iteration}: max gradient = {max_grad:.6e}, best = {best_idx}")
            # Print some gradient values for debugging
            for i, grad in enumerate(grads):
                if i % 10 == 0:
                    print(f"  Gradient {i+1}/{len(pool)}: {grad:.6e}")

        if max_grad < grad_tol:
            if verbose:
                print("Converged: gradient below threshold.")
            break

        # Add best operator to ansatz
        ansatz_ops.append(pool[best_idx])
        params.append(0.0)

                # Optimize parameters using scipy.optimize.minimize
        def vqe_obj(x):
            circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, x)
            energy = measure_expectation(circuit, H_qubit_op)
            return energy

        # Debug: Test objective function at a few points
        if verbose and iteration == 0:
            print(f"  Testing objective function:")
            test_params = [0.0, -0.05, -0.08, -0.1, 0.05, 0.1]
            for test_val in test_params:
                test_energy = vqe_obj([test_val])
                print(f"    θ = {test_val:6.3f}: E = {test_energy:.8f}")

        # Try optimization with different methods and starting points
        initial_guess = params.copy()
        print(f"  Starting optimization from: {initial_guess}")

        # Use a more robust optimization strategy
        res = minimize(vqe_obj, initial_guess, method='COBYLA',
                      options={'maxiter': 200, 'disp': False})

        params = list(res.x)

        # Update energy
        final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)
        energy = measure_expectation(final_circuit, H_qubit_op)
        energies.append(energy)

        if verbose:
            print(f"  Energy after iteration {iteration}: {energy:.8f}")

    # Return final state
    final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops, params)
    final_state = get_statevector(final_circuit)

    if verbose:
        print(f"Total measurements used: {total_measurements}")

    return energies, params, ansatz_ops, final_state, total_measurements


if __name__ == "__main__":
    # Load H4 Hamiltonian from file
    with open('../../ham_lib/h4_sto-3g.pkl', 'rb') as f:
        fermion_op = pickle.load(f)

    # Determine n_qubits from Hamiltonian
    qubit_op = ferm_to_qubit(fermion_op)
    n_qubits = max(idx for term in qubit_op.terms for idx, _ in term) + 1
    n_electrons = 4  # For H4

    # Convert to Qiskit SparsePauliOp
    H_qubit_op = openfermion_qubitop_to_sparsepauliop(qubit_op, n_qubits)

    # Compute exact ground state energy
    H_sparse = H_qubit_op.to_matrix(sparse=True)
    exact_energy, exact_gs = exact_ground_state_energy(H_sparse)
    print(f"Exact ground state energy (diagonalization): {exact_energy:.8f}")

    ref_occ = get_occ_no('h4', n_qubits)
    hf_state = get_reference_state(ref_occ, gs_format='wfs')


    # Build UCCSD operator pool and convert to Qiskit Operators
    print(f"Building UCCSD pool for {n_qubits} qubits, {n_electrons} electrons...")
    uccsd_pool_fermion = list(get_all_uccsd_anti_hermitian(n_qubits, n_electrons))
    pool = [fermion_operator_to_qiskit_operator(op, n_qubits) for op in uccsd_pool_fermion]
    print(f"UCCSD pool size: {len(pool)}")

    # Generate cache filename for commutator maps
    cache_filename = generate_commutator_cache_filename('h4', n_qubits, n_electrons, len(uccsd_pool_fermion))

    # Get commutator maps with caching
    fragment_group_indices_map, commutator_indices_map = get_commutator_maps_cached(
        fermion_op, uccsd_pool_fermion, n_qubits,
        molecule_name='h4', n_electrons=n_electrons, cache_file=cache_filename)

    print(f"QWC groups found: {len(fragment_group_indices_map)}")
    print(f"Commutator mappings for {len(commutator_indices_map)} operators")


    # Run ADAPT-VQE with commutator gradients (using multiprocessing for gradient calculation)

    # Configuration parameters
    use_parallel = True
    max_workers = None
    executor_type = 'multiprocessing'  # For this version
    molecule_name = 'h4'

    energies, params, ansatz, final_state, total_measurements = adapt_vqe_qiskit(
        H_qubit_op, n_qubits, n_electrons, pool, fermion_op, uccsd_pool_fermion,
        fragment_group_indices_map, commutator_indices_map)

    # Calculate results
    final_energy = energies[-1]
    overlap = np.abs(np.vdot(final_state.data, exact_gs)) ** 2
    ansatz_depth = len(ansatz)

    # Print results (as before)
    print("Final energy:", final_energy)
    print("Parameters:", params)
    print(f"Ansatz depth: {ansatz_depth}")
    print(f"Total measurements: {total_measurements}")
    print(f"Fidelity (|<ADAPT-VQE|Exact>|^2): {overlap:.8f}")

    # Save results to CSV
    save_results_to_csv(
        final_energy=final_energy,
        total_measurements=total_measurements,
        exact_energy=exact_energy,
        fidelity=overlap,
        molecule_name=molecule_name,
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        pool_size=len(pool),
        use_parallel=use_parallel,
        executor_type=executor_type,
        max_workers=max_workers,
        ansatz_depth=ansatz_depth,
        filename='adapt_vqe_qubitwise_results.csv'  # Different filename to distinguish this qubit-wise approach
    )
