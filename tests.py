from adapt_vqe_qiskit_qubitwise_bai import get_qubit_w_commutation_group_parallel
from test_bai_setup import test_bai_with_simple_system
from SolvableQubitHamiltonians.qubit_util import get_commutator_qubit
from Decompositions.qwc_decomposition import qwc_decomposition

if __name__ == "__main__":
    test_results = test_bai_with_simple_system()
    H_qubit_op, H_sparse_pauli_op, generator_pool, n_qubits, n_electrons, fragment_group_indices_map, commutator_indices_map = test_results
    print(f"H_qubit_op: {H_qubit_op}")
    print(f"Generator pool: {generator_pool}")

    first_commutator = get_commutator_qubit(H_qubit_op, generator_pool[0])
    pauli_groups = qwc_decomposition(first_commutator)
    second_commutator = get_commutator_qubit(H_qubit_op, generator_pool[1])
    pauli_groups2 = qwc_decomposition(second_commutator)
    print(f"First commutator QWC decomposition {pauli_groups}")
    print(f"Second commutator QWC decomposition: {pauli_groups2}")
    fragment_group_indices_map, commutator_indices_map = get_qubit_w_commutation_group_parallel(H_qubit_op, generator_pool, n_qubits)
    print(fragment_group_indices_map)
    print(commutator_indices_map)
