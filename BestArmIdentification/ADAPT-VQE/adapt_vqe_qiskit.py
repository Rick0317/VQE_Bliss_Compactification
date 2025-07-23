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

# --- OpenFermion imports for chemistry mapping ---
from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator

# Add path to import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OperatorPools.generalized_fermionic_pool import get_all_uccsd_anti_hermitian, get_all_anti_hermitian
from utils.ferm_utils import ferm_to_qubit
from SolvableQubitHamiltonians.fermi_util import get_commutator
from Decompositions.qwc_decomposition import qwc_decomposition
from StatePreparation.reference_state_utils import get_reference_state, get_occ_no

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
    if backend is None or isinstance(backend, type(Aer.get_backend('statevector_simulator'))):
        # Use statevector simulator for exact results
        state = get_statevector(circuit)
        return np.real(state.expectation_value(observable))
    else:
        # Use shot-based simulation
        return measure_pauli_expectation(circuit, observable, 8192)

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

def compute_commutator_gradient(current_state_circuit, H_fermion, generator_fermion, n_qubits):
    """Compute gradient using commutator [H,G] decomposed into Pauli operators"""
    # Compute fermion commutator [H, G]
    commutator_fermion = get_commutator(H_fermion, generator_fermion)

    # Convert to qubit operator
    commutator_qubit = ferm_to_qubit(commutator_fermion)

    # Decompose into Pauli groups
    pauli_groups = qwc_decomposition(commutator_qubit)

    # Measure expectation value of each Pauli group
    total_expectation = 0.0

    total_measurements = 0
    for group in pauli_groups:
        # Convert to Qiskit SparsePauliOp
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
        expectation_val = measure_expectation(current_state_circuit, group_op, backend=True, shots=8192)
        total_expectation += expectation_val
        total_measurements += 8192

    return np.real(total_expectation), total_measurements

def compute_gradient_wrapper(args):
    """Wrapper function for multiprocessing gradient computation"""
    current_circuit, H_fermion, generator_fermion, n_qubits, idx = args
    try:
        print(f"Computing gradient for {idx}")
        grad, total_measurements = compute_commutator_gradient(current_circuit, H_fermion, generator_fermion, n_qubits)
        return idx, np.abs(grad), total_measurements
    except Exception as e:
        print(f"Error computing gradient for operator {idx}: {e}")
        return idx, 0.0, 0  # Return 0 measurements for failed computation


def adapt_vqe_qiskit(H_qubit_op, n_qubits, n_electrons, pool, H_fermion, pool_fermion, max_iter=30, grad_tol=1e-2, verbose=True, use_multiprocessing=True):
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
        if verbose:
            print(f"Computing gradients for {len(pool)} operators...")

        # Compute gradients - use multiprocessing if enabled and available
        if use_multiprocessing:
            try:
                num_processes = cpu_count()
                with Pool(num_processes) as p:
                    # Prepare arguments as tuples for the worker function
                    args_list = [(current_circuit, H_fermion, ferm_op, n_qubits, i)
                                for i, ferm_op in enumerate(pool_fermion)]
                    # Map the function over the arguments list
                    results = p.map(compute_gradient_wrapper, args_list)
                    # Sort results by index and extract gradients and measurements
                    results.sort(key=lambda x: x[0])
                    grads = [r[1] for r in results]
                    measurements = [r[2] for r in results]
                    iteration_measurements = sum(measurements)
                    total_measurements += iteration_measurements
                    if verbose:
                        print(f"  Computed {len(grads)} gradients using {num_processes} processes")
                        print(f"  Measurements this iteration: {iteration_measurements}")
            except Exception as e:
                if verbose:
                    print(f"  Multiprocessing failed ({e}), falling back to sequential processing")
                use_multiprocessing = False

        if not use_multiprocessing:
            # Sequential processing fallback
            grads = []
            iteration_measurements = 0
            for i, (op, ferm_op) in enumerate(zip(pool, pool_fermion)):
                # Compute gradient using commutator measurement
                grad, measurements = compute_commutator_gradient(current_circuit, H_fermion, ferm_op, n_qubits)
                grads.append(np.abs(grad))
                iteration_measurements += measurements

                if verbose and i % 10 == 0:
                    print(f"  Gradient {i+1}/{len(pool)}: {np.abs(grad):.6e}")

            total_measurements += iteration_measurements
            if verbose:
                print(f"  Measurements this iteration: {iteration_measurements}")

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
            print(f"  Operator {i}: OpenFermion gradient = {np.abs(of_grad):.6e}")
        print()

    # Run ADAPT-VQE with commutator gradients (using multiprocessing for gradient calculation)

    # Configuration parameters
    use_parallel = True
    max_workers = None
    executor_type = 'multiprocessing'  # For this version
    molecule_name = 'h4'

    energies, params, ansatz, final_state, total_measurements = adapt_vqe_qiskit(H_qubit_op, n_qubits, n_electrons, pool, fermion_op, uccsd_pool_fermion, use_multiprocessing=use_parallel)

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
        filename='adapt_vqe_multiprocessing_results.csv'  # Different filename to distinguish from BAI version
    )
