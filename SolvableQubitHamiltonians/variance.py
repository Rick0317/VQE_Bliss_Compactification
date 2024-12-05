from openfermion import (
    QubitOperator as Q,
    commutator,
    anticommutator,
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    bravyi_kitaev,
    FermionOperator,
    get_majorana_operator
)

from one_norm_func_gen import generate_analytical_one_norm, generate_analytical_one_norm_2_body, construct_symmetric_matrix
import numpy as np
from fermi_util import get_commutator
from algo_utils import bliss_three_body, bliss_three_body_cheaper
from main_utils_partitioning import N_QUBITS, copy_hamiltonian, sorted_insertion_decomposition
import pickle
from qwc_decomposition import qwc_decomposition
from scipy.optimize import minimize

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


def construct_H_bliss(H, params, N, Ne):
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))
    result = H
    mu_1 = params[0]
    mu_2 = params[1]
    o_1 = params[2:]
    o1_ferm = params_to_matrix_op(o_1, N)

    result -= mu_1 * (total_number_operator - Ne)
    result -= mu_2 * (total_number_operator ** 2 - Ne ** 2)
    result -= o1_ferm * (total_number_operator - Ne)

    return result


def ferm_to_qubit(H: FermionOperator):
    Hqub = bravyi_kitaev(H)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(
    sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub


def commutator_variance(H: FermionOperator, decomp, num_modes):
    """
    Computes the variance of the [H, G] - K.
    :param H: The Hamiltonian to compute the variance of.
    :param decomp: The decomposition of the Hamiltonian.
    :param G: The fermion operator to define the gradient
    :param K: The Killer operator applied to the whole [H, G]
    :param N: The number of sites
    :return: The variance metric
    """
    psi = ggs(gso(H, num_modes))[1]
    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, num_modes), psi)
    return np.sum((vars) ** (1 / 2)) ** 2


if __name__ == '__main__':
    moltag = 'lih'
    methodtag = 'fc'

    # N = N_QUBITS[moltag]
    N = 12
    Ne = 4

    filename = f'ham_lib/lih_fer.bin'
    with open(filename, 'rb') as f:
        H = pickle.load(f)


    # G = FermionOperator('1^ 3^ 2 0') - FermionOperator('0^ 2^ 3 1')

    # ommutator = get_commutator(H, G)
    H_q = ferm_to_qubit(H)
    print("Commutator Obtained in Fermion and Qubit space")
    majo = get_majorana_operator(H)

    one_norm = 0
    for term, coeff in majo.terms.items():
        one_norm += abs(coeff)

    print("Original 1-Norm", one_norm)
    # bliss_output = bliss_three_body(commutator, N, N//2)
    #bliss_output = bliss_three_body_cheaper(commutator, N, Ne)

    one_norm_func, one_norm_expr = generate_analytical_one_norm_2_body(H, N, Ne)

    def optimization_wrapper(params):
        x_val = params[0]
        y_val = params[1]
        o_vals = params[2:]
        return one_norm_func(x_val, y_val, o_vals)


    x_val = 0
    y_val = 0
    o_val = construct_symmetric_matrix(N)

    o_val_flattened = o_val.flatten()

    initial_guess = np.concatenate((np.array([0, 0]), o_val_flattened,))
    res = minimize(optimization_wrapper, initial_guess, method='BFGS',
                   options={'gtol': 1e-300, 'disp': True, 'maxiter': 300})

    bliss_output = construct_H_bliss(H, res.x, N, Ne)

    H_bliss_q = ferm_to_qubit(bliss_output)
    print("BLISS Complete. Obtained in Fermion and Qubit space")

    decomp = sorted_insertion_decomposition(H_q, methodtag)
    # decomp = qwc_decomposition(H_q)
    print("Original Decomposition Complete")
    blissed_decomp = sorted_insertion_decomposition(H_bliss_q, methodtag)
    # blissed_decomp = qwc_decomposition(H_bliss_q)
    print("Blissed Decomposition Complete")
    var = commutator_variance(H_q, decomp, N)

    blissed_vars = commutator_variance(H_bliss_q, blissed_decomp, N)

    print(f"Original variance: {var}")
    print(f"Blissed variance: {blissed_vars}")
