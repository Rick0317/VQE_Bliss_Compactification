import pickle
from scipy.sparse.linalg import eigsh  # sparse eigensolver
from openfermion.transforms import jordan_wigner
from openfermion.linalg import get_sparse_operator

if __name__ == "__main__":
    with open('../ham_lib/lih_fer.bin', 'rb') as f:
        fermion_op = pickle.load(f)
    mol = 'lih'

    qubit_op = jordan_wigner(fermion_op)

    n_qubits = 12
    sparse_hamiltonian = get_sparse_operator(qubit_op, n_qubits)

    ground_energy, _ = eigsh(sparse_hamiltonian, k=1,
                             which='SA')  # 'SA' = smallest algebraic
    print(f"Ground state energy of {mol}: {ground_energy[0]}")

