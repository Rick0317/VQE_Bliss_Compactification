from openfermion import (
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    bravyi_kitaev,
    normal_ordered
)

from SolvableQubitHamiltonians.main_utils_partitioning import copy_hamiltonian
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
import pickle
from Decompositions.qwc_decomposition import qwc_decomposition
from BLISS.normal_bliss.customized_bliss_package import *


def abs_of_dict_value(x):
    return np.abs(x[1])


def load_hamiltonian(moltag):
    filename = f'ham_lib/{moltag}_fer.bin'
    with open(filename, 'rb') as f:
        Hfer = pickle.load(f)
    Hqub = bravyi_kitaev(Hfer)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub


def params_to_matrix_op(params, n):
    ferm_op = FermionOperator()
    matrix_param = params.reshape((n, n))
    for i in range(n):
        for j in range(n):
            ferm_op += FermionOperator(f'{i}^ {j}', matrix_param[i, j])

    return ferm_op


def ferm_to_qubit(H: FermionOperator):
    Hqub = bravyi_kitaev(H)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(
    sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub


def commutator_variance(H: FermionOperator, decomp, N, Ne):
    """
    Computes the variance of the [H, G] - K.
    :param H: The Hamiltonian to compute the variance of.
    :param decomp: The decomposition of the Hamiltonian.
    :param G: The fermion operator to define the gradient
    :param K: The Killer operator applied to the whole [H, G]
    :param N: The number of sites
    :return: The variance metric
    """
    psi = ggs(gso(H, N))[1]

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, N), psi)
    return np.sum((vars) ** (1 / 2)) ** 2


if __name__ == '__main__':

    moltag = 'lih'
    methodtag = 'fc'

    N = 8
    Ne = 4

    filename = f'../SolvableQubitHamiltonians/ham_lib/h4_sto-3g.pkl'
    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)



    ordered_hamil = normal_ordered(Hamil)
    com_name_list = [
        '3^ 2^ 0 1',
        '3^ 2^ 0 2',
        '4^ 3^ 5 2',
        '2^ 5^ 1 6',
        '0^ 7^ 5 2',
        '1^ 6^ 0 7',
        '3^ 6^ 0 5',
        '2^ 1^ 3 4',
        '1^ 6^ 6 5',
        '0^ 1^ 3 2',
        '5^ 4^ 2 1'
    ]
    # idx_list = [0, 1, 2, 3]
    anti_com_list = [
        FermionOperator('3^ 2^ 0 1') - FermionOperator('1^ 0^ 2 3'),
        FermionOperator('3^ 2^ 0 2') - FermionOperator('2^ 0^ 2 3'),
        FermionOperator('4^ 3^ 5 2') - FermionOperator('2^ 5^ 3 4'),
        FermionOperator('2^ 5^ 1 6') - FermionOperator('6^ 1^ 5 2'),
        FermionOperator('0^ 7^ 5 2') - FermionOperator('2^ 5^ 7 0'),
        FermionOperator('1^ 6^ 0 7') - FermionOperator('7^ 0^ 6 1'),
        FermionOperator('3^ 6^ 0 5') - FermionOperator('5^ 0^ 6 3'),
        FermionOperator('2^ 1^ 3 4') - FermionOperator('4^ 3^ 1 2'),
        FermionOperator('1^ 6^ 6 5') - FermionOperator('5^ 6^ 6 1'),
        FermionOperator('0^ 1^ 3 2') - FermionOperator('2^ 3^ 1 0'),
        FermionOperator('5^ 4^ 2 1') - FermionOperator('1^ 2^ 4 5'),

    ]

    G = anti_com_list[0]
    hamil_copy1 = copy_ferm_hamiltonian(Hamil)
    hamil_copy2 = copy_ferm_hamiltonian(Hamil)
    g_copy1 = copy_ferm_hamiltonian(G)
    g_copy2 = copy_ferm_hamiltonian(G)

    commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2

    H1 = normal_ordered(commutator)

    G = anti_com_list[2]
    hamil_copy1 = copy_ferm_hamiltonian(Hamil)
    hamil_copy2 = copy_ferm_hamiltonian(Hamil)
    g_copy1 = copy_ferm_hamiltonian(G)
    g_copy2 = copy_ferm_hamiltonian(G)

    commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2

    H2 = normal_ordered(commutator)

    copied_Hamil_1 = copy_ferm_hamiltonian(H1)
    copied_Hamil_2 = copy_ferm_hamiltonian(H2)

    H_q = ferm_to_qubit(copied_Hamil_1)
    print("Commutator Obtained in Fermion and Qubit space")
    H_copy = copy_hamiltonian(H_q)
    # decomp = sorted_insertion_decomposition(H_copy, methodtag)
    decomp = qwc_decomposition(H_copy)
    print("Original Decomposition Complete")
    var = commutator_variance(H_q, decomp, N, Ne)
    print(f"Original variance1: {var}")

    H_q = ferm_to_qubit(copied_Hamil_2)
    print("Commutator Obtained in Fermion and Qubit space")
    H_copy = copy_hamiltonian(H_q)
    # decomp = sorted_insertion_decomposition(H_copy, methodtag)
    decomp = qwc_decomposition(H_copy)
    print("Original Decomposition Complete")
    var = commutator_variance(H_q, decomp, N, Ne)
    print(f"Original variance2: {var}")

    summ_H = H1 + H2

    H_q = ferm_to_qubit(summ_H)
    print("Commutator Obtained in Fermion and Qubit space")
    H_copy = copy_hamiltonian(H_q)
    # decomp = sorted_insertion_decomposition(H_copy, methodtag)
    decomp = qwc_decomposition(H_copy)
    print("Original Decomposition Complete")
    var = commutator_variance(H_q, decomp, N, Ne)
    print(f"Summed Variance: {var}")

    subtract_H = H1 - H2

    H_q = ferm_to_qubit(subtract_H)
    print("Commutator Obtained in Fermion and Qubit space")
    H_copy = copy_hamiltonian(H_q)
    # decomp = sorted_insertion_decomposition(H_copy, methodtag)
    decomp = qwc_decomposition(H_copy)
    print("Original Decomposition Complete")
    var = commutator_variance(H_q, decomp, N, Ne)
    print(f"Subtracted variance1: {var}")



