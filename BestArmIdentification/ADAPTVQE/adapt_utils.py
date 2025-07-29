from qiskit.quantum_info import Operator
from openfermion import get_sparse_operator

def qubit_operator_to_qiskit_operator(qubit_op, n_qubits):
    """Convert a FermionOperator to a Qiskit Operator via bravyi_kitaev mapping"""
    sparse_matrix = get_sparse_operator(qubit_op, n_qubits)
    return Operator(sparse_matrix.toarray())
