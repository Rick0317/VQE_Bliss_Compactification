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
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import gc
import psutil
import csv
from datetime import datetime

# --- OpenFermion imports for chemistry mapping ---
from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator

# Add path to import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OperatorPools.generalized_fermionic_pool import \
    get_all_uccsd_anti_hermitian, get_all_anti_hermitian
from utils.ferm_utils import ferm_to_qubit
from SolvableQubitHamiltonians.fermi_util import get_commutator
from Decompositions.qwc_decomposition import qwc_decomposition
from StatePreparation.reference_state_utils import get_reference_state, \
    get_occ_no
from BestArmIdentification.SuccessiveElimination.best_arm_identification import integer_allocations_one_guaranteed


def compute_gradient(H_sparse, op_sparse, state):
    """Compute gradient ⟨ψ|[H,G]|ψ⟩ = ⟨ψ|HG - GH|ψ⟩"""
    HP_psi = H_sparse.dot(op_sparse.dot(state))
    PH_psi = op_sparse.dot(H_sparse.dot(state))
    return np.vdot(state, HP_psi) - np.vdot(state, PH_psi)


def get_hf_bitstring(n_qubits, n_electrons):
    # Returns a bitstring with n_electrons ones at the left
    return '1' * n_electrons + '0' * (n_qubits - n_electrons)


def hf_circuit(n_qubits, n_electrons):
    # Prepare Hartree-Fock state using reference state utilities
    ref_occ = get_occ_no('h4', n_qubits)
    hf_state = get_reference_state(ref_occ, gs_format='wfs')

    # Create circuit and initialize with the HF state
    qc = QuantumCircuit(n_qubits)
    # Normalize the state vector to avoid precision issues
    hf_state_normalized = hf_state / np.linalg.norm(hf_state)
    qc.initialize(hf_state_normalized, range(n_qubits))
    return qc


def get_statevector(qc):
    backend = Aer.get_backend('statevector_simulator')
    t_qc = transpile(qc, backend)
    result = backend.run(t_qc).result()
    return Statevector(result.get_statevector(t_qc))


def exact_ground_state_energy(H_sparse):
    vals, vecs = eigsh(H_sparse, k=1, which='SA')
    return vals[0], vecs[:, 0]


def openfermion_qubitop_to_sparsepauliop(q_op, n_qubits):
    """
    Convert OpenFermion Qubit Operators to SparsePauliOp
    :param q_op:
    :param n_qubits:
    :return:
    """
    paulis = []
    coeffs = []
    for term, coeff in q_op.terms.items():
        pauli_str = ['I'] * n_qubits
        for idx, p in term:
            pauli_str[idx] = p
        paulis.append(''.join(pauli_str))
        coeffs.append(coeff)
    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def fermion_operator_to_qiskit_operator(ferm_op, n_qubits):
    """Convert a FermionOperator to a Qiskit Operator via bravyi_kitaev mapping"""
    qubit_op = ferm_to_qubit(ferm_op)
    sparse_matrix = get_sparse_operator(qubit_op, n_qubits)
    return Operator(sparse_matrix.toarray())


def measure_pauli_expectation(circuit, pauli_op, shots=8192):
    """Measure expectation value of a Pauli operator using shot-based simulation"""
    # Create measurement circuit for each Pauli term
    total_expectation = 0.0
    counts = {}

    for i, (pauli_string, coeff) in enumerate(pauli_op.to_list()):
        if pauli_string == 'I' * len(pauli_string):
            # Identity term contributes coefficient directly
            total_expectation += coeff
            continue

        if i == 0:

            # Create measurement circuit for this Pauli string
            meas_circuit = circuit.copy()
            meas_circuit.add_register(ClassicalRegister(len(pauli_string), 'c'))

            # Apply basis rotations for measurement
            # Note: Qiskit Pauli strings use reverse indexing (leftmost = highest qubit)
            n_qubits = len(pauli_string)
            for i, pauli in enumerate(pauli_string):
                qubit_idx = n_qubits - 1 - i  # Reverse the qubit indexing
                if pauli == 'X':
                    meas_circuit.ry(-np.pi/2, qubit_idx)
                elif pauli == 'Y':
                    meas_circuit.rx(np.pi/2, qubit_idx)
                # Z measurement requires no rotation

            # Add measurements
            for i in range(len(pauli_string)):
                meas_circuit.measure(i, i)

            # Run simulation
            simulator = AerSimulator()
            compiled_circuit = transpile(meas_circuit, simulator)
            result = simulator.run(compiled_circuit, shots=shots).result()
            counts = result.get_counts()

        # Calculate expectation value for this Pauli term
        expectation = 0.0
        for bitstring, count in counts.items():
            # Calculate parity (-1)^(number of 1s in positions where Pauli is not I)
            parity = 1
            for i, pauli in enumerate(pauli_string):
                if pauli != 'I' and bitstring[i] == '1':  # bitstring[i] corresponds to pauli_string[i]
                    parity *= -1
            expectation += parity * count / shots

        total_expectation += coeff * expectation

    return np.real(total_expectation)


def measure_expectation(circuit, observable, backend=None, shots=8192):
    """Measure expectation value of an observable given a quantum circuit"""
    if backend is None or isinstance(backend, type(Aer.get_backend(
            'statevector_simulator'))):
        # Use statevector simulator for exact results
        state = get_statevector(circuit)
        return np.real(state.expectation_value(observable))
    else:
        # Use shot-based simulation
        return measure_pauli_expectation(circuit, observable, shots)


def create_ansatz_circuit(n_qubits, n_electrons, operators, parameters):
    """Create a parameterized ansatz circuit"""
    circuit = hf_circuit(n_qubits, n_electrons)
    for op, theta in zip(operators, parameters):
        # Apply exp(i * theta * op) to the circuit
        # Compute matrix exponential: exp(i * theta * G)
        unitary_matrix = expm(theta * op.data)
        unitary_gate = Operator(unitary_matrix)
        circuit.append(unitary_gate, range(n_qubits))
    return circuit


def compute_commutator_gradient(current_state_circuit, H_fermion,
                                generator_fermion, n_qubits):
    """Compute gradient using commutator [H,G] decomposed into Pauli operators"""
    # Compute fermion commutator [H, G]
    commutator_fermion = get_commutator(H_fermion, generator_fermion)

    # Convert to qubit operator
    commutator_qubit = ferm_to_qubit(commutator_fermion)

    # Decompose into Pauli groups
    pauli_groups = qwc_decomposition(commutator_qubit)

    # Measure expectation value of each Pauli group
    total_expectation = 0.0
    for group in pauli_groups:
        # Convert to Qiskit SparsePauliOp
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
        expectation_val = measure_expectation(current_state_circuit, group_op,
                                              backend=True)
        total_expectation += expectation_val

    return np.real(total_expectation)


def prepare_commutator_decomps(pool, pool_fermion,
                               H_fermion, n_qubits):
    commutator_dict = {}
    for i, (op, ferm_op) in enumerate(zip(pool, pool_fermion)):
        commutator_fermion = get_commutator(H_fermion, ferm_op)

        commutator_qubit = ferm_to_qubit(commutator_fermion)

        pauli_groups = qwc_decomposition(commutator_qubit)

        commutator_dict[i] = []

        for group in pauli_groups:
            group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
            commutator_dict[i].append(group_op)

    return commutator_dict



def evaluate_arm_worker(args):
    """
    Worker function to evaluate a single arm in parallel (for both threading and multiprocessing).
    Includes memory management to prevent OOM crashes.

    Args:
        args: Tuple containing (arm_id, current_state_circuit, fragments,
               actual_allocation, rounds, total_allocation, total_allocation_history)

    Returns:
        Tuple of (arm_id, rewards, total_allocation, updated_total_allocation_history)
    """
    arm_id, current_state_circuit, fragments, actual_allocation, rounds, total_allocation, prev_total_allocation = args

    try:
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        rewards = 0
        for j, group in enumerate(fragments):
            expectation_val = measure_expectation(current_state_circuit,
                                                  group,
                                                  backend=True,
                                                  shots=actual_allocation[j])
            rewards += expectation_val

            # Force garbage collection after each group to free memory
            if j % 5 == 0:  # Every 5 groups
                gc.collect()

            # Check memory usage periodically
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            if current_memory > initial_memory + 500:  # If memory increased by >500MB
                print(f"Warning: Arm {arm_id} memory usage high: {current_memory:.1f} MB")
                gc.collect()

        # Final cleanup
        del current_state_circuit, fragments, actual_allocation
        gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Arm {arm_id}: Memory {initial_memory:.1f} -> {final_memory:.1f} MB")

        return arm_id, rewards, total_allocation, prev_total_allocation + total_allocation

    except Exception as e:
        print(f"Error in arm {arm_id}: {str(e)}")
        # Force cleanup on error
        gc.collect()
        raise


def get_optimal_worker_count(n_qubits, max_workers=None, executor_type='process'):
    """
    Determine optimal worker count based on available memory and system resources.

    Args:
        n_qubits: Number of qubits (affects memory usage)
        max_workers: User-specified maximum workers
        executor_type: 'process' or 'thread'

    Returns:
        Optimal number of workers
    """
    # Get system info
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Estimate memory per worker (rough approximation)
    # Quantum simulation memory scales exponentially with qubits
    estimated_memory_per_worker = min(2 ** (n_qubits - 8), 8.0)  # GB, capped at 8GB

    # Calculate memory-based limit
    memory_based_limit = max(1, int(memory_gb * 0.8 / estimated_memory_per_worker))

    # For processes, be more conservative due to overhead
    if executor_type == 'process':
        recommended_workers = min(cpu_count - 1, memory_based_limit)
    else:  # threading
        recommended_workers = min(cpu_count, memory_based_limit)

    # Apply user limit if specified
    if max_workers is not None:
        recommended_workers = min(recommended_workers, max_workers)

    recommended_workers = max(1, recommended_workers)  # At least 1 worker

    print(f"System info: {cpu_count} CPUs, {memory_gb:.1f} GB available memory")
    print(f"Estimated memory per worker: {estimated_memory_per_worker:.1f} GB")
    print(f"Recommended workers: {recommended_workers}")

    return recommended_workers


def best_arm_qiskit_parallel(current_state_circuit, commutator_dict, allocations, delta=0.05, max_rounds=10, max_workers=None, executor_type='process'):
    """
    Parallel version of best_arm_qiskit using ThreadPoolExecutor or ProcessPoolExecutor.

    Args:
        current_state_circuit: Current quantum circuit state
        commutator_dict: Dictionary mapping arm indices to Pauli group fragments
        allocations: Dictionary mapping arm indices to allocation probabilities
        delta: Confidence parameter for successive elimination
        max_rounds: Maximum number of rounds
        max_workers: Maximum number of parallel workers (None for default)
        executor_type: 'thread' for ThreadPoolExecutor, 'process' for ProcessPoolExecutor

    Returns:
        Tuple of (max_gradient, best_arm_index, total_samples)
    """
    K = len(allocations.items())
    active_arms = list(range(K))
    estimates = np.zeros(K)
    pulls = np.zeros(K)
    total_allocation_history = np.zeros(K)
    rounds = 0

    while len(active_arms) > 1 and rounds < max_rounds:
        rounds += 1
        print(f"Round {rounds}")

        # Prepare arguments for parallel execution
        worker_args = []
        for i in active_arms:
            if rounds == 1:
                total_allocation = 10000
                fragments = commutator_dict[i]
                actual_allocation = integer_allocations_one_guaranteed(
                    allocations[i], total_allocation)
            else:
                total_allocation = 10000
                fragments = commutator_dict[i]
                actual_allocation = integer_allocations_one_guaranteed(
                    allocations[i], total_allocation)

            worker_args.append((i, current_state_circuit, fragments,
                               actual_allocation, rounds, total_allocation,
                               total_allocation_history[i]))

                # Execute arm evaluations in parallel
        start_time = time.time()

        # Choose executor type
        if executor_type.lower() == 'thread':
            ExecutorClass = ThreadPoolExecutor
            executor_name = "ThreadPoolExecutor"
        elif executor_type.lower() == 'process':
            ExecutorClass = ProcessPoolExecutor
            executor_name = "ProcessPoolExecutor"
        else:
            raise ValueError(f"executor_type must be 'thread' or 'process', got {executor_type}")

        # Determine optimal worker count - estimate n_qubits from circuit
        n_qubits = len(current_state_circuit.qubits) if hasattr(current_state_circuit, 'qubits') else 8
        optimal_workers = get_optimal_worker_count(n_qubits, max_workers, executor_type)

        print(f"Using {executor_name} with {optimal_workers} workers")

        # Monitor system memory before starting
        memory_before = psutil.virtual_memory()
        print(f"Memory before: {memory_before.available / (1024**3):.1f} GB available ({memory_before.percent:.1f}% used)")

        with ExecutorClass(max_workers=optimal_workers) as executor:
            # Submit all tasks
            future_to_arm = {executor.submit(evaluate_arm_worker, args): args[0]
                            for args in worker_args}

            # Collect results
            for future in as_completed(future_to_arm):
                arm_id = future_to_arm[future]
                try:
                    arm_id, rewards, total_allocation, updated_total_allocation = future.result()

                    # Update estimates and allocation history
                    if rounds == 1:
                        pulls[arm_id] += 1000
                        estimates[arm_id] += rewards
                    else:
                        pulls[arm_id] += 1000
                        estimates[arm_id] = (estimates[arm_id] * total_allocation_history[arm_id] +
                                           rewards * total_allocation) / updated_total_allocation

                    total_allocation_history[arm_id] = updated_total_allocation
                    print(f"Completed arm {arm_id}: reward = {rewards:.6f}")

                except Exception as exc:
                    print(f"Arm {arm_id} generated an exception: {exc}")
                    if "terminated abruptly" in str(exc) or "process pool" in str(exc).lower():
                        print(f"⚠️  Process killed (likely OOM). Try:")
                        print(f"   - Reduce max_workers: max_workers={max(1, optimal_workers//2)}")
                        print(f"   - Use threading: executor_type='thread'")
                        print(f"   - Reduce allocation sizes")

        parallel_time = time.time() - start_time

        # Monitor memory after execution
        memory_after = psutil.virtual_memory()
        print(f"Memory after: {memory_after.available / (1024**3):.1f} GB available ({memory_after.percent:.1f}% used)")
        print(f"Parallel evaluation completed in {parallel_time:.2f} seconds")

        # Force garbage collection
        gc.collect()

        # Elimination phase (same as original)
        means = estimates
        print(f"Means estimates: {means}")
        print(f"Active arms: {active_arms}")
        max_mean = max(abs(means[active_arms]))

        radius = np.sqrt(
            np.log(0.01 * len(active_arms) * (pulls ** 2) / delta) / pulls)

        print(f"{rounds} / {max_rounds} / {len(active_arms)}, Radius: {radius}")

        new_active_arms = []
        for i in active_arms:
            if abs(means[i]) + radius[i] >= max_mean - radius[i]:
                new_active_arms.append(i)
        active_arms = new_active_arms

    # Final results
    means = estimates / rounds
    print(f"Active arms: {active_arms}")
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    print(f"Round: {rounds}, Mean Estimates: {means}")
    print(f"Pulls from each arm: {pulls}")

    return abs(means[best_arm]), best_arm, sum(total_allocation_history)


def best_arm_qiskit(current_state_circuit, commutator_dict, allocations, delta=0.05, max_rounds=10):
    """

    :param commutator_dict:
    :param allocations: dict
    :return:
    """
    K = len(allocations.items())
    active_arms = list(range(K))
    estimates = np.zeros(K)
    pulls = np.zeros(K)
    total_allocation_history = np.zeros(K)
    rounds = 0

    while len(active_arms) > 1 and rounds < max_rounds:
        rounds += 1
        print(f"Round {rounds}")
        for i in active_arms:
            print(f"Which Arm: {i}")
            if rounds == 1:
                total_allocation = 100000
                total_allocation_history[i] += total_allocation
                fragments = commutator_dict[i]

                # uniform_distribution = [1 / len(fragments) for _ in range(len(fragments))]
                actual_allocation = integer_allocations_one_guaranteed(
                    allocations[i], total_allocation)
                # actual_allocation = integer_allocations_one_guaranteed(
                #     uniform_distribution, total_allocation)
                rewards = 0
                for j, group in enumerate(fragments):
                    expectation_val = measure_expectation(current_state_circuit,
                                                          group,
                                                          backend=True,
                                                          shots=actual_allocation[j])
                    rewards += expectation_val
                pulls[i] += 50000
                estimates[i] += rewards
            else:
                total_allocation = 10000
                total_allocation_history[i] += total_allocation
                fragments = commutator_dict[i]
                # uniform_distribution = [1 / len(fragments) for _ in
                #                         range(5000len(fragments))]
                actual_allocation = integer_allocations_one_guaranteed(allocations[i], total_allocation)
                # actual_allocation = integer_allocations_one_guaranteed(
                #     uniform_distribution, total_allocation)
                # actual_allocation = [166, 166, 166, 166, 166, 166]
                rewards = 0
                for j, group in enumerate(fragments):
                    expectation_val = measure_expectation(current_state_circuit,
                                                          group,
                                                          backend=True,
                                                          shots=
                                                          actual_allocation[j])
                    rewards += expectation_val
                pulls[i] += 5000
                estimates[i] = (estimates[i] * total_allocation_history[i] + rewards * total_allocation) / (total_allocation_history[i] + total_allocation)

        # means = estimates / np.maximum(pulls, 1)
        means = estimates
        print(f"Means estimates: {means}")
        print(f"Active arms: {active_arms}")
        max_mean = max(abs(means[active_arms]))

        radius = np.sqrt(
            np.log(0.000001 * len(active_arms) * (pulls ** 2) / delta) / pulls)

        print(f"{rounds} / {max_rounds} / {len(active_arms)}, Radius: {radius}")

        new_active_arms = []
        for i in active_arms:
            if abs(means[i]) + radius[i] >= max_mean - radius[i]:
                new_active_arms.append(i)
        active_arms = new_active_arms


    # means = estimates / np.maximum(pulls, 1)
    means = estimates / rounds
    print(f"Active arms: {active_arms}")
    best_arm = max(np.array(active_arms), key=lambda i: abs(means[i]))
    print(f"Round: {rounds}, Mean Estimates: {means}")
    print(f"Pulls from each arm: {pulls}")


    return abs(means[best_arm]), best_arm, sum(total_allocation_history)



def get_allocation_estimate(generators, H_fermion, n_qubits, n_electrons):
    allocation_dict = {}

    # Create HF circuit once for reuse
    hf_circuit_state = hf_circuit(n_qubits, n_electrons)

    for i, generator in enumerate(generators):
        commutator_fermion = get_commutator(H_fermion, generator)

        # Convert to qubit operator
        commutator_qubit = ferm_to_qubit(commutator_fermion)

        # Decompose into Pauli groups
        pauli_groups = qwc_decomposition(commutator_qubit)
        variances = []

        for group in pauli_groups:
            # Convert to Qiskit SparsePauliOp
            group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)

            # Calculate expectation value <ψ|A|ψ>
            expectation_val = measure_expectation(hf_circuit_state, group_op, backend=None)

            # For general operator A = Σᵢ cᵢ Pᵢ, we need to calculate <ψ|A²|ψ>
            # A² = (Σᵢ cᵢ Pᵢ)(Σⱼ cⱼ Pⱼ) = Σᵢⱼ cᵢcⱼ PᵢPⱼ
            # We can compute this by squaring the SparsePauliOp
            group_op_squared = group_op @ group_op  # Matrix multiplication gives A²
            expectation_val_squared = measure_expectation(hf_circuit_state, group_op_squared, backend=None)

            # Var(A) = E[A²] - E[A]²
            variance = expectation_val_squared - expectation_val**2
            variances.append(variance)

        # Convert variances to probabilities that sum to 1
        total_variance = sum(variances)
        if total_variance > 0:
            probabilities = [var / total_variance for var in variances]
        else:
            # If all variances are zero, use uniform distribution
            probabilities = [1.0 / len(variances) for _ in variances]

        allocation_dict[i] = probabilities

    return allocation_dict


def save_allocations(allocations, filename):
    """Save allocation dictionary to a JSON file"""
    # Convert numpy arrays to lists for JSON serialization
    allocations_serializable = {}
    for key, value in allocations.items():
        allocations_serializable[str(key)] = [float(x) for x in value]

    with open(filename, 'w') as f:
        json.dump(allocations_serializable, f, indent=2)
    print(f"Allocations saved to {filename}")


def load_allocations(filename):
    """Load allocation dictionary from a JSON file"""
    try:
        with open(filename, 'r') as f:
            allocations_serializable = json.load(f)

        # Convert back to integer keys
        allocations = {}
        for key, value in allocations_serializable.items():
            allocations[int(key)] = value

        print(f"Allocations loaded from {filename}")
        return allocations
    except FileNotFoundError:
        print(f"Allocation file {filename} not found")
        return None
    except Exception as e:
        print(f"Error loading allocations: {e}")
        return None


def get_allocation_estimate_cached(generators, H_fermion, n_qubits, n_electrons, cache_file=None):
    """
    Get allocation estimates with caching support

    Args:
        generators: List of fermion operators
        H_fermion: Hamiltonian fermion operator
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        cache_file: Optional filename to cache/load allocations

    Returns:
        allocation_dict: Dictionary mapping operator indices to probability distributions
    """
    if cache_file is not None:
        # Try to load from cache first
        allocations = load_allocations(cache_file)
        if allocations is not None:
            # Verify the cache matches current pool size
            if len(allocations) == len(generators):
                return allocations
            else:
                print(f"Cache size mismatch: expected {len(generators)}, got {len(allocations)}")

    # Compute allocations if not cached or cache invalid
    print("Computing allocation estimates...")
    allocations = get_allocation_estimate(generators, H_fermion, n_qubits, n_electrons)

    # Save to cache if filename provided
    if cache_file is not None:
        save_allocations(allocations, cache_file)

    return allocations


def generate_cache_filename(molecule_name, n_qubits, n_electrons, pool_size):
    """Generate a cache filename based on molecular system parameters"""
    return f"allocations_{molecule_name}_{n_qubits}q_{n_electrons}e_{pool_size}pool.json"


def save_results_to_csv(final_energy, total_measurements, exact_energy, fidelity,
                       molecule_name, n_qubits, n_electrons, pool_size,
                       use_parallel, executor_type, max_workers,
                       ansatz_depth, filename='adapt_vqe_results.csv'):
    """
    Save ADAPT-VQE results to a CSV file.

    Args:
        final_energy: Final energy from ADAPT-VQE
        total_measurements: Total number of measurements used
        exact_energy: Exact ground state energy from diagonalization
        fidelity: Fidelity with exact ground state
        molecule_name: Name of the molecule
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        pool_size: Size of the operator pool
        use_parallel: Whether parallel evaluation was used
        executor_type: Type of executor used ('process' or 'thread')
        max_workers: Maximum number of workers used
        ansatz_depth: Final ansatz depth (number of operators)
        filename: CSV filename to save results
    """

    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)

    # Prepare the row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    energy_error = abs(final_energy - exact_energy)

    row_data = {
        'timestamp': timestamp,
        'molecule': molecule_name,
        'n_qubits': n_qubits,
        'n_electrons': n_electrons,
        'pool_size': pool_size,
        'final_energy': final_energy,
        'exact_energy': exact_energy,
        'energy_error': energy_error,
        'fidelity': fidelity,
        'total_measurements': total_measurements,
        'ansatz_depth': ansatz_depth,
        'use_parallel': use_parallel,
        'executor_type': executor_type if use_parallel else 'serial',
        'max_workers': max_workers if use_parallel else 1
    }

    # Write to CSV
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(row_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        # Write the data row
        writer.writerow(row_data)

    print(f"Results saved to {filename}")


def adapt_vqe_qiskit(H_qubit_op, n_qubits, n_electrons, pool, H_fermion,
                     pool_fermion, max_iter=30, grad_tol=1e-2, verbose=True,
                     molecule_name='h4', use_parallel=True, max_workers=None, executor_type='process'):
    """
    ADAPT-VQE algorithm with best-arm identification and parallel evaluation.

    Args:
        H_qubit_op: Hamiltonian as Qiskit SparsePauliOp
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        pool: Pool of operators as Qiskit Operators
        H_fermion: Hamiltonian as FermionOperator
        pool_fermion: Pool of operators as FermionOperators
        max_iter: Maximum number of ADAPT-VQE iterations
        grad_tol: Gradient tolerance for convergence
        verbose: Whether to print progress
        molecule_name: Name of molecule (for caching)
        use_parallel: Whether to use parallel arm evaluation (default: True)
        max_workers: Maximum number of parallel workers (default: None = use all cores)
        executor_type: 'thread' for ThreadPoolExecutor, 'process' for ProcessPoolExecutor (default: 'process')

    Returns:
        Tuple of (energies, parameters, ansatz_operators, final_state, total_measurements)
    """

    # Prepare reference state
    ansatz_ops = []
    params = []
    energies = []

    total_measurements = 0

    final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops,
                                          params)
    energy = measure_expectation(final_circuit, H_qubit_op)
    print(f"HF energy (Qiskit): {energy}")

    # Generate cache filename for allocations
    cache_filename = generate_cache_filename(molecule_name, n_qubits, n_electrons, len(pool_fermion))

    for iteration in range(max_iter):
        # Create current ansatz circuit
        print(
            f'Iteration {iteration}, Length of Ansatz: {len(ansatz_ops)}, Parameters: {params}')
        current_circuit = create_ansatz_circuit(n_qubits, n_electrons,
                                                ansatz_ops, params)

        # Compute gradients for all pool operators using commutator measurement
        if verbose:
            print(f"Computing gradients for {len(pool)} operators...")

        commutator_dict = prepare_commutator_decomps(pool, pool_fermion, H_fermion, n_qubits)
        print(f"Commutator dict computed")
        allocations = get_allocation_estimate_cached(pool_fermion, H_fermion, n_qubits, n_electrons, cache_file=cache_filename)
        if use_parallel:
            max_grad, best_idx, total_samples = best_arm_qiskit_parallel(current_circuit, commutator_dict, allocations, max_workers=max_workers, executor_type=executor_type)
        else:
            max_grad, best_idx, total_samples = best_arm_qiskit(current_circuit, commutator_dict, allocations)
        total_measurements += total_samples

        if verbose:
            print(
                f"Iteration {iteration}: max gradient = {max_grad:.6e}, best = {best_idx}")
        if max_grad < grad_tol:
            if verbose:
                print("Converged: gradient below threshold.")
            break

        # Add best operator to ansatz
        ansatz_ops.append(pool[best_idx])
        params.append(0.0)

        # Optimize parameters using scipy.optimize.minimize
        def vqe_obj(x):
            circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops,
                                            x)
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
        final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops,
                                              params)
        energy = measure_expectation(final_circuit, H_qubit_op)
        energies.append(energy)

        if verbose:
            print(f"  Energy after iteration {iteration}: {energy:.8f}")

    # Return final state
    final_circuit = create_ansatz_circuit(n_qubits, n_electrons, ansatz_ops,
                                          params)
    final_state = get_statevector(final_circuit)
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
    ref_occ = get_occ_no('h4', n_qubits)
    hf_state = get_reference_state(ref_occ, gs_format='wfs')

    # Build UCCSD operator pool and convert to Qiskit Operators
    print(
        f"Building UCCSD pool for {n_qubits} qubits, {n_electrons} electrons...")
    uccsd_pool_fermion = list(
        get_all_uccsd_anti_hermitian(n_qubits, n_electrons))
    pool = [fermion_operator_to_qiskit_operator(op, n_qubits) for op in
            uccsd_pool_fermion]
    print(f"UCCSD pool size: {len(pool)}")

    # Debug: Compare with OpenFermion gradient calculation
    if len(pool) > 0:
        print("Comparing first few gradients with OpenFermion calculation:")
        from openfermion import get_sparse_operator

        H_sparse_of = get_sparse_operator(qubit_op, n_qubits)
        ref_occ = get_occ_no('h4', n_qubits)
        hf_state_of = get_reference_state(ref_occ, gs_format='wfs')

        for i in range(min(3, len(uccsd_pool_fermion))):
            # OpenFermion gradient
            G_qubit = ferm_to_qubit(uccsd_pool_fermion[i])
            G_sparse = get_sparse_operator(G_qubit, n_qubits)
            of_grad = compute_gradient(H_sparse_of, G_sparse, hf_state_of)
            print(
                f"  Operator {i}: OpenFermion gradient = {np.abs(of_grad):.6e}")
        print()

    # Run ADAPT-VQE with commutator gradients
    # use_parallel=True (default) enables parallel arm evaluation
    # max_workers=None (default) uses all available CPU cores
    # executor_type='process' (default) uses ProcessPoolExecutor for better CPU-bound performance
    # To use serial evaluation: use_parallel=False
    # To limit parallel workers: max_workers=4 (for example)
    # To use threading: executor_type='thread'

    # Configuration parameters
    use_parallel = True
    max_workers = None
    executor_type = 'process'
    molecule_name = 'h4'

    energies, params, ansatz, final_state, total_measurements = adapt_vqe_qiskit(H_qubit_op,
                                                             n_qubits,
                                                             n_electrons, pool,
                                                             fermion_op,
                                                             uccsd_pool_fermion,
                                                             molecule_name=molecule_name,
                                                             use_parallel=use_parallel,
                                                             max_workers=max_workers,
                                                             executor_type=executor_type)

    # Calculate results
    final_energy = energies[-1]
    overlap = np.abs(np.vdot(final_state.data, exact_gs)) ** 2
    ansatz_depth = len(ansatz)

    # Print results (as before)
    print("Final energy:", final_energy)
    print(f"Total measurements: {total_measurements}")
    print(f"Exact ground state energy (diagonalization): {exact_energy:.8f}")
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
        filename='adapt_vqe_results.csv'
    )
