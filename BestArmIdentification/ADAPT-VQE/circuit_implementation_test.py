import numpy as np
import pickle
import sys
import os

# Add path to import from parent directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from OperatorPools.generalized_fermionic_pool import get_all_uccsd_anti_hermitian
from utils.ferm_utils import ferm_to_qubit
from SolvableQubitHamiltonians.fermi_util import get_commutator
from Decompositions.qwc_decomposition import qwc_decomposition
from StatePreparation.reference_state_utils import get_reference_state, get_occ_no
from BestArmIdentification.SuccessiveElimination.best_arm_identification import integer_allocations_one_guaranteed

# Import functions from the main file
from adapt_vqe_qiskit_bai import (
    hf_circuit, 
    openfermion_qubitop_to_sparsepauliop, 
    measure_expectation,
    get_statevector,
    prepare_commutator_decomps,
    get_allocation_estimate_cached
)

from qiskit import QuantumCircuit, transpile, ClassicalRegister
from qiskit_aer import AerSimulator


def measure_pauli_expectation_normal_ordering(circuit, pauli_op, shots=8192):
    """Measure expectation value of a Pauli operator using NORMAL bit ordering"""
    total_expectation = 0.0

    for pauli_string, coeff in pauli_op.to_list():
        if pauli_string == 'I' * len(pauli_string):
            # Identity term contributes coefficient directly
            total_expectation += coeff
            continue

        # Create measurement circuit for this Pauli string
        meas_circuit = circuit.copy()
        meas_circuit.add_register(ClassicalRegister(len(pauli_string), 'c'))

        # Apply basis rotations for measurement
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                meas_circuit.ry(-np.pi / 2, i)
            elif pauli == 'Y':
                meas_circuit.rx(np.pi / 2, i)
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
            # Calculate parity using NORMAL bit ordering
            parity = 1
            for i, pauli in enumerate(pauli_string):
                if pauli != 'I' and bitstring[i] == '1':  # NORMAL bit ordering
                    parity *= -1
            expectation += parity * count / shots

        total_expectation += coeff * expectation

    return np.real(total_expectation)


def measure_pauli_expectation_reverse_ordering(circuit, pauli_op, shots=8192):
    """Measure expectation value of a Pauli operator using REVERSE bit ordering (original)"""
    total_expectation = 0.0

    for pauli_string, coeff in pauli_op.to_list():
        if pauli_string == 'I' * len(pauli_string):
            # Identity term contributes coefficient directly
            total_expectation += coeff
            continue

        # Create measurement circuit for this Pauli string
        meas_circuit = circuit.copy()
        meas_circuit.add_register(ClassicalRegister(len(pauli_string), 'c'))

        # Apply basis rotations for measurement
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                meas_circuit.ry(-np.pi / 2, i)
            elif pauli == 'Y':
                meas_circuit.rx(np.pi / 2, i)
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
            # Calculate parity using REVERSE bit ordering (original)
            parity = 1
            for i, pauli in enumerate(pauli_string):
                if pauli != 'I' and bitstring[-(i + 1)] == '1':  # REVERSE bit ordering
                    parity *= -1
            expectation += parity * count / shots

        total_expectation += coeff * expectation

    return np.real(total_expectation)


def test_bit_ordering_comparison():
    """Test both bit ordering conventions and compare with exact results"""
    
    print("Testing bit ordering conventions...")
    print("="*60)
    
    # Load H4 Hamiltonian from file
    hamiltonian_path = '../../ham_lib/h4_sto-3g.pkl'
    with open(hamiltonian_path, 'rb') as f:
        fermion_op = pickle.load(f)
    
    # Determine n_qubits from Hamiltonian
    qubit_op = ferm_to_qubit(fermion_op)
    n_qubits = max(idx for term in qubit_op.terms for idx, _ in term) + 1
    n_electrons = 4  # For H4
    
    print(f"System parameters:")
    print(f"  n_qubits: {n_qubits}")
    print(f"  n_electrons: {n_electrons}")
    print()
    
    # Build UCCSD operator pool
    print("Building UCCSD pool...")
    uccsd_pool_fermion = list(get_all_uccsd_anti_hermitian(n_qubits, n_electrons))
    print(f"UCCSD pool size: {len(uccsd_pool_fermion)}")
    
    # Check if arm 85 exists
    if len(uccsd_pool_fermion) <= 85:
        print(f"ERROR: Pool only has {len(uccsd_pool_fermion)} operators, but arm 85 requested")
        return
    
    # Get the 85th operator (0-indexed, so index 85)
    G_fermion = uccsd_pool_fermion[85]
    print(f"Selected operator 85 from pool")
    print()
    
    # Compute commutator [H, G]
    print("Computing commutator [H, G]...")
    commutator_fermion = get_commutator(fermion_op, G_fermion)
    
    # Convert to qubit operator
    commutator_qubit = ferm_to_qubit(commutator_fermion)
    
    # Decompose into Pauli groups
    pauli_groups = qwc_decomposition(commutator_qubit)
    print(f"Decomposed into {len(pauli_groups)} Pauli groups")
    
    # Create Hartree-Fock state circuit
    hf_circuit_state = hf_circuit(n_qubits, n_electrons)
    
    # Test first few groups with different bit ordering
    print("\nComparing bit ordering conventions for first 3 groups:")
    print("="*60)
    print(f"{'Group':>5} {'Exact':>12} {'Normal':>12} {'Reverse':>12} {'Norm-Diff':>12} {'Rev-Diff':>12}")
    print("-" * 77)
    
    shots = 1000000  # Use more shots for better precision
    
    for i, group in enumerate(pauli_groups[:3]):  # Test first 3 groups
        # Convert to Qiskit SparsePauliOp
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
        
        # Exact calculation
        exact_val = measure_expectation(hf_circuit_state, group_op, backend=None)
        
        # Normal bit ordering
        normal_val = measure_pauli_expectation_normal_ordering(hf_circuit_state, group_op, shots=shots)
        
        # Reverse bit ordering
        reverse_val = measure_pauli_expectation_reverse_ordering(hf_circuit_state, group_op, shots=shots)
        
        # Calculate differences
        norm_diff = abs(normal_val - exact_val)
        rev_diff = abs(reverse_val - exact_val)
        
        print(f"{i:>5} {exact_val:>12.8f} {normal_val:>12.8f} {reverse_val:>12.8f} {norm_diff:>12.8f} {rev_diff:>12.8f}")
    
    print("\nTesting with multiple runs to account for statistical noise...")
    print("="*60)
    
    # Test with multiple runs and average the results
    num_runs = 5
    group_to_test = 0  # Test first group
    
    if len(pauli_groups) > group_to_test:
        group_op = openfermion_qubitop_to_sparsepauliop(pauli_groups[group_to_test], n_qubits)
        exact_val = measure_expectation(hf_circuit_state, group_op, backend=None)
        
        normal_vals = []
        reverse_vals = []
        
        for run in range(num_runs):
            normal_val = measure_pauli_expectation_normal_ordering(hf_circuit_state, group_op, shots=shots)
            reverse_val = measure_pauli_expectation_reverse_ordering(hf_circuit_state, group_op, shots=shots)
            normal_vals.append(normal_val)
            reverse_vals.append(reverse_val)
            print(f"Run {run+1}: Normal={normal_val:.8f}, Reverse={reverse_val:.8f}")
        
        # Calculate statistics
        normal_mean = np.mean(normal_vals)
        normal_std = np.std(normal_vals)
        reverse_mean = np.mean(reverse_vals)
        reverse_std = np.std(reverse_vals)
        
        print(f"\nResults for group {group_to_test}:")
        print(f"Exact value:     {exact_val:.8f}")
        print(f"Normal ordering: {normal_mean:.8f} ± {normal_std:.8f}")
        print(f"Reverse ordering: {reverse_mean:.8f} ± {reverse_std:.8f}")
        print(f"Normal bias:     {abs(normal_mean - exact_val):.8f}")
        print(f"Reverse bias:    {abs(reverse_mean - exact_val):.8f}")
        
        # Determine which is closer
        if abs(normal_mean - exact_val) < abs(reverse_mean - exact_val):
            print("✓ NORMAL bit ordering appears to be correct!")
        else:
            print("✓ REVERSE bit ordering appears to be correct!")


def simulate_parallel_execution_with_hf_state():
    """Simulate what the parallel execution would do but using HF state for consistency"""
    
    print("Simulating parallel execution with HF state...")
    print("="*60)
    
    # Load H4 Hamiltonian from file
    hamiltonian_path = '../../ham_lib/h4_sto-3g.pkl'
    with open(hamiltonian_path, 'rb') as f:
        fermion_op = pickle.load(f)
    
    # Determine n_qubits from Hamiltonian
    qubit_op = ferm_to_qubit(fermion_op)
    n_qubits = max(idx for term in qubit_op.terms for idx, _ in term) + 1
    n_electrons = 4  # For H4
    
    print(f"System parameters:")
    print(f"  n_qubits: {n_qubits}")
    print(f"  n_electrons: {n_electrons}")
    print()
    
    # Build UCCSD operator pool
    print("Building UCCSD pool...")
    uccsd_pool_fermion = list(get_all_uccsd_anti_hermitian(n_qubits, n_electrons))
    print(f"UCCSD pool size: {len(uccsd_pool_fermion)}")
    
    # Check if arm 85 exists
    if len(uccsd_pool_fermion) <= 85:
        print(f"ERROR: Pool only has {len(uccsd_pool_fermion)} operators, but arm 85 requested")
        return
    
    # Create HF circuit (this is what we'll use instead of current_circuit)
    hf_circuit_state = hf_circuit(n_qubits, n_electrons)
    print(f"HF circuit created with {len(hf_circuit_state.qubits)} qubits")
    
    # Step 1: Prepare commutator decompositions (like in the main algorithm)
    print("\nStep 1: Preparing commutator decompositions...")
    commutator_dict = prepare_commutator_decomps([], uccsd_pool_fermion, fermion_op, n_qubits)
    print(f"Commutator dict computed for {len(commutator_dict)} operators")
    
    # Step 2: Get allocation estimates (like in the main algorithm)
    print("\nStep 2: Getting allocation estimates...")
    cache_filename = f"test_allocations_h4_{n_qubits}q_{n_electrons}e_{len(uccsd_pool_fermion)}pool.json"
    allocations = get_allocation_estimate_cached(uccsd_pool_fermion, fermion_op, n_qubits, n_electrons, cache_file=cache_filename)
    print(f"Allocations computed for {len(allocations)} operators")
    
    # Step 3: Simulate arm evaluation for arm 85 (like in evaluate_arm_worker)
    print("\nStep 3: Simulating arm 85 evaluation...")
    arm_id = 85
    
    # Get the fragments and allocation for arm 85
    fragments = commutator_dict[arm_id]
    allocation_probs = allocations[arm_id]
    
    # Simulate what happens in the parallel execution
    total_allocation = 1000000  # First round allocation
    actual_allocation = integer_allocations_one_guaranteed(allocation_probs, total_allocation)
    
    print(f"Arm {arm_id} has {len(fragments)} fragments")
    print(f"Allocation probabilities: {allocation_probs}")
    print(f"Actual allocation: {actual_allocation}")
    print(f"Total shots: {sum(actual_allocation)}")
    
    # Step 4: Measure expectation value for each fragment (like in evaluate_arm_worker)
    print("\nStep 4: Measuring expectation values...")
    print("-" * 50)
    
    rewards = 0
    for j, group in enumerate(fragments):
        # Use HF state instead of current_state_circuit
        expectation_val = measure_expectation(hf_circuit_state, group, backend=True, shots=actual_allocation[j])
        rewards += expectation_val
        print(f"  Fragment {j:2d}: {expectation_val:12.8f} (shots: {actual_allocation[j]})")
    
    print("-" * 50)
    print(f"Total reward (sum of fragments): {rewards:.8f}")
    print(f"Absolute value: {abs(rewards):.8f}")
    
    return rewards


def test_arm_85_commutator():
    """Test the expectation value calculation for the 85th arm's commutator [H, G]"""
    
    print("Testing arm 85 commutator expectation value calculation...")
    print("="*60)
    
    # Load H4 Hamiltonian from file
    hamiltonian_path = '../../ham_lib/h4_sto-3g.pkl'
    with open(hamiltonian_path, 'rb') as f:
        fermion_op = pickle.load(f)
    
    # Determine n_qubits from Hamiltonian
    qubit_op = ferm_to_qubit(fermion_op)
    n_qubits = max(idx for term in qubit_op.terms for idx, _ in term) + 1
    n_electrons = 4  # For H4
    
    print(f"System parameters:")
    print(f"  n_qubits: {n_qubits}")
    print(f"  n_electrons: {n_electrons}")
    print()
    
    # Build UCCSD operator pool
    print("Building UCCSD pool...")
    uccsd_pool_fermion = list(get_all_uccsd_anti_hermitian(n_qubits, n_electrons))
    print(f"UCCSD pool size: {len(uccsd_pool_fermion)}")
    
    # Check if arm 85 exists
    if len(uccsd_pool_fermion) <= 85:
        print(f"ERROR: Pool only has {len(uccsd_pool_fermion)} operators, but arm 85 requested")
        return
    
    # Get the 85th operator (0-indexed, so index 85)
    G_fermion = uccsd_pool_fermion[85]
    print(f"Selected operator 85 from pool")
    print(f"Operator type: {type(G_fermion)}")
    print()
    
    # Compute commutator [H, G]
    print("Computing commutator [H, G]...")
    commutator_fermion = get_commutator(fermion_op, G_fermion)
    print(f"Commutator computed")
    
    # Convert to qubit operator
    print("Converting to qubit operator...")
    commutator_qubit = ferm_to_qubit(commutator_fermion)
    print(f"Commutator has {len(commutator_qubit.terms)} terms")
    
    # Decompose into Pauli groups
    print("Decomposing into Pauli groups...")
    pauli_groups = qwc_decomposition(commutator_qubit)
    print(f"Decomposed into {len(pauli_groups)} Pauli groups")
    
    # Create Hartree-Fock state circuit
    print("Creating Hartree-Fock state circuit...")
    hf_circuit_state = hf_circuit(n_qubits, n_electrons)
    print(f"HF circuit created with {len(hf_circuit_state.qubits)} qubits")
    
    # Calculate expectation value for each Pauli group
    print("\nCalculating expectation values for each Pauli group:")
    print("-" * 50)
    
    total_expectation = 0.0
    
    for i, group in enumerate(pauli_groups):
        # Convert to Qiskit SparsePauliOp
        group_op = openfermion_qubitop_to_sparsepauliop(group, n_qubits)
        
        # Measure expectation value using exact statevector simulation
        expectation_val = measure_expectation(hf_circuit_state, group_op, backend=None)
        
        print(f"  Group {i:2d}: {expectation_val:12.8f} (terms: {len(group.terms)})")
        total_expectation += expectation_val
    
    print("-" * 50)
    print(f"Total expectation value: {total_expectation:.8f}")
    print(f"Absolute value: {abs(total_expectation):.8f}")
    print()
    
    # Additional verification using direct commutator calculation
    print("\n" + "="*60)
    print("VERIFICATION: Direct commutator calculation")
    print("="*60)
    
    # Get HF state as vector
    ref_occ = get_occ_no('h4', n_qubits)
    hf_state_vector = get_reference_state(ref_occ, gs_format='wfs')
    
    # Convert operators to sparse matrices
    from openfermion import get_sparse_operator
    H_sparse = get_sparse_operator(qubit_op, n_qubits)
    G_qubit = ferm_to_qubit(G_fermion)
    G_sparse = get_sparse_operator(G_qubit, n_qubits)
    
    # Direct commutator calculation: <ψ|[H,G]|ψ> = <ψ|HG - GH|ψ>
    HG_psi = H_sparse.dot(G_sparse.dot(hf_state_vector))
    GH_psi = G_sparse.dot(H_sparse.dot(hf_state_vector))
    direct_commutator = np.vdot(hf_state_vector, HG_psi) - np.vdot(hf_state_vector, GH_psi)
    
    print(f"Direct commutator calculation: {direct_commutator:.8f}")
    print(f"Pauli decomposition result: {total_expectation:.8f}")
    print(f"Difference: {abs(direct_commutator - total_expectation):.8f}")
    
    # Test if they match within numerical precision
    if abs(direct_commutator - total_expectation) < 1e-10:
        print("✓ PASS: Direct and decomposed calculations match!")
    else:
        print("✗ FAIL: Calculations don't match within tolerance")
    
    return total_expectation, direct_commutator


if __name__ == "__main__":
    # First test bit ordering
    print("STEP 1: Testing bit ordering conventions")
    print("="*80)
    test_bit_ordering_comparison()
    
    print("\n" + "="*80)
    print("STEP 2: Simulating parallel execution with HF state")
    print("="*80)
    
    # Simulate what parallel execution would do with HF state
    parallel_result = simulate_parallel_execution_with_hf_state()
    
    print("\n" + "="*80)
    print("STEP 3: Running direct calculation test")
    print("="*80)
    
    # Then run the direct calculation test
    direct_result, verification_result = test_arm_85_commutator()
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Direct calculation result:     {direct_result:.8f}")
    print(f"Parallel simulation result:   {parallel_result:.8f}")
    print(f"Verification result:          {verification_result:.8f}")
    print(f"Difference (direct vs parallel): {abs(direct_result - parallel_result):.8f}")
    
    # Check if they match (allowing for shot noise)
    if abs(direct_result - parallel_result) < 0.01:  # Allow for shot noise
        print("✓ SUCCESS: Both methods give consistent results!")
    else:
        print("✗ DISCREPANCY: Results don't match - check implementation")
