from openfermion import (
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    bravyi_kitaev,
    get_majorana_operator,
    jw_get_ground_state_at_particle_number as jwggs
)

from SolvableQubitHamiltonians.main_utils_partitioning import copy_hamiltonian
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
import pickle
from Decompositions.qwc_decomposition import qwc_decomposition
from scipy.optimize import minimize
from BLISS.normal_bliss.bliss_package import *
from BLISS.normal_bliss.customized_bliss_package import *
from BLISS.normal_bliss.indices_filter import filter_indices_normal_bliss


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


def commutator_variance_subspace(H: FermionOperator, decomp, N, Ne):
    """
    Computes the variance of the [H, G] - K.
    :param H: The Hamiltonian to compute the variance of.
    :param decomp: The decomposition of the Hamiltonian.
    :param G: The fermion operator to define the gradient
    :param K: The Killer operator applied to the whole [H, G]
    :param N: The number of sites
    :return: The variance metric
    """
    psi = jwggs(gso(H, N), Ne)[1]

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, N), psi)
    return np.sum((vars) ** (1 / 2)) ** 2, psi


if __name__ == '__main__':
    moltag = 'Beh2'
    methodtag = 'qwc'

    N = 8
    Ne = 4

    filename = f'../SolvableQubitHamiltonians/ham_lib/h4_sto-3g.pkl'
    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)

    print(f"Original Hamil {Hamil}")


    two_body_list = filter_indices_normal_bliss(Hamil, N, Ne)

    H_q = ferm_to_qubit(Hamil)
    print("Commutator Obtained in Fermion and Qubit space")
    H_copy = copy_hamiltonian(H_q)
    # decomp = sorted_insertion_decomposition(H_copy, methodtag)
    decomp = qwc_decomposition(H_copy)
    print("Original Decomposition Complete")
    var = commutator_variance(H_q, decomp, N, Ne)
    print(f"Original variance: {var}")

    majo = get_majorana_operator(Hamil)

    one_norm = 0
    term_counter = 0
    for term, coeff in majo.terms.items():
        if term != ():
            term_counter += 1
            one_norm += abs(coeff)

    print("Original 1-Norm", one_norm)

    H_before_bliss1 = copy_ferm_hamiltonian(Hamil)
    optimization_wrapper, initial_guess = optimization_bliss_mu12_o1(
        H_before_bliss1, N, Ne)
    # opt_method = 'Nelder-Mead 100k Iter'
    #
    # res = minimize(optimization_wrapper, initial_guess, method='BFGS',
    #                options={'gtol': 1e-300, 'disp': True, 'maxiter': 600, 'eps': 1e-2})
    # # Try a different method that doesn't require gradients
    res = minimize(optimization_wrapper, initial_guess, method='Powell',
                   options={'disp': True, 'maxiter': 100000})
    x = res.x
    count = sum(1 for value in x if abs(value) > 1E-6)
    print(f"Number of parameters to be optimized: {count} / {term_counter}")
    # x = generate_analytical_one_norm_2_body(H_before_bliss1, N, Ne)
    if x.all() == None:
        print(f"X null")
        exit()

    H_before_modification = copy_ferm_hamiltonian(Hamil)
    bliss_output = construct_H_bliss_m12_o1(H_before_modification,
                                                      x, N, Ne,
                                                      )

    print(f"First 5 parameters: {x[:5]}")

    H_before_bliss_test = copy_ferm_hamiltonian(Hamil)
    print(f"BLISS Output: {bliss_output}")
    H_bliss_output_test = copy_ferm_hamiltonian(bliss_output)
    check_correctness(H_before_bliss_test, H_bliss_output_test, Ne)
    print("Bliss correctness check complete")

    H_bliss_output = copy_ferm_hamiltonian(bliss_output)
    H_bliss_q = ferm_to_qubit(bliss_output)
    print("BLISS Complete. Obtained in Fermion and Qubit space")
    majo_blissed = get_majorana_operator(H_bliss_output)
    blissed_one_norm = 0
    for term, coeff in majo_blissed.terms.items():
        if term != ():
            blissed_one_norm += abs(coeff)

    print("Blissed 1-Norm", blissed_one_norm)

    H_bliss_copy = copy_hamiltonian(H_bliss_q)
    # blissed_decomp = sorted_insertion_decomposition(H_bliss_copy, methodtag)
    blissed_decomp = qwc_decomposition(H_bliss_copy)
    print("Blissed Decomposition Complete")

    H_decompose_copy = copy_hamiltonian(H_bliss_q)

    blissed_vars = commutator_variance(H_decompose_copy,
                                       blissed_decomp.copy(), N, Ne)

    print(f"Blissed variance: {blissed_vars}")
