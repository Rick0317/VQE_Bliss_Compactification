from BLISS.normal_bliss.bliss_package import *
from scipy.optimize import minimize
from openfermion import FermionOperator as FO
from BLISS.normal_bliss.indices_filter import filter_indices
from BLISS.normal_bliss.customized_bliss_package import *


def copy_ferm_hamiltonian(H: FO):
    H_copy = FO().zero()

    for t, s in H.terms.items():
        H_copy += s * FO(t)

    assert (H - H_copy) == FO().zero()
    return H_copy


def bliss_two_body(H, N, Ne):
    """
    Apply BLISS to two body Hamiltonian.
    :param H:
    :param N:
    :param Ne:
    :return:
    """
    H_input = copy_ferm_hamiltonian(H)
    optimization_wrapper, initial_guess = optimization_bliss_mu12_o1(
        H_input, N, Ne)

    res = minimize(optimization_wrapper, initial_guess, method='Powell',
                   options={'disp': True, 'maxiter': 100000})

    x = res.x
    H_before_modification = copy_ferm_hamiltonian(H)
    bliss_output = construct_H_bliss_m12_o1(H_before_modification,
                                            x, N, Ne,
                                            )
    return bliss_output


def bliss_three_body_indices_filtered(H, N, Ne):
    """
    Apply BLISS to three body Hamiltonian with indices filter used.
    :param H:
    :param N:
    :param Ne:
    :return:
    """
    two_body_list = filter_indices(H, N, Ne)
    if len(two_body_list) != 0:
        H_input = copy_ferm_hamiltonian(H)

        optimization_wrapper, initial_guess = optimize_bliss_mu3_customizable(
            H_input, N, Ne, two_body_list)

        res = minimize(optimization_wrapper, initial_guess, method='Powell',
                       options={'disp': True, 'maxiter': 100000})

        H_before_modification = copy_ferm_hamiltonian(H)
        bliss_output, killer = construct_H_bliss_mu3_customizable(
            H_before_modification, res.x, N, Ne, two_body_list)

        return bliss_output
    else:
        return H
