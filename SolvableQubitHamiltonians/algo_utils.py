import numpy as np
from openfermion import (
    FermionOperator,
    normal_ordered,
    get_majorana_operator
)
import random
from scipy.optimize import minimize


def callback(xk, trunc_H, num_modes, num_elec,):
    print(f"Current 1-Norm: {bliss_cost_fn_cheaper(xk, trunc_H, num_modes, num_elec,)}")


def bliss_three_body(trunc_H: FermionOperator, num_modes, num_elec):

    initial_mu = [0, 0, 0]
    init_O = [0 for _ in range(int(num_modes * (num_modes + 1)))]
    # init_O = [random.random() for _ in range(num_modes ** 2)]
    initial_guess = np.concatenate((initial_mu, init_O))
    res = minimize(bliss_cost_fn, initial_guess, args=(trunc_H, num_modes, num_elec,), method='BFGS',
                   options={'gtol': 1e-30, 'disp': True, 'maxiter': 10}, callback=lambda xk: callback(xk, trunc_H, num_modes, num_elec))

    bliss_result = blissed_H(res.x, trunc_H, num_modes, num_elec)

    return bliss_result


def bliss_three_body_cheaper(trunc_H: FermionOperator, num_modes, num_elec):

    initial_mu = [0, 0, 0]
    init_O = [random.random() * 0.01 for _ in range(int(num_modes * (num_modes + 1) // 2))]
    # init_O = [random.random() for _ in range(num_modes ** 2)]
    initial_guess = np.concatenate((initial_mu, init_O))
    res = minimize(bliss_cost_fn_cheaper, initial_guess, args=(trunc_H, num_modes, num_elec,), method='BFGS',
                   options={'gtol': 1e-300, 'disp': True, 'maxiter': 300}, callback=lambda xk: callback(xk, trunc_H, num_modes, num_elec))
    # res = minimize(bliss_cost_fn_cheaper, initial_guess, args=(trunc_H, num_modes, num_elec,), method='nelder-mead',
    #                options={'xatol': 1e-30, 'disp': True, 'maxfev': 1000}, callback=lambda xk: callback(xk, trunc_H, num_modes, num_elec))

    bliss_result = blissed_H_cheap(res.x, trunc_H, num_modes, num_elec)

    return bliss_result


def total_num_op(num_modes):
    """
    Calculates the total number operator for a given number of modes.
    :param num_modes:
    :return:
    """
    total_number_operator = FermionOperator()
    for mode in range(num_modes):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    return total_number_operator


def construct_symmetric_fermion_operator(sym_matrix):
    """
    Constructs a symmetric FermionOperator from a symmetric matrix.

    Args:
        sym_matrix (ndarray): An n x n symmetric matrix.

    Returns:
        FermionOperator: The constructed symmetric FermionOperator.
    """
    n = sym_matrix.shape[0]

    # Check if the matrix is symmetric
    if not np.allclose(sym_matrix, sym_matrix.T):
        raise ValueError("The input matrix must be symmetric.")

    fermion_operator = FermionOperator()

    # Iterate over the matrix elements to construct the operator
    for i in range(n):
        for j in range(i, n):  # Only loop over the upper triangle (i <= j)
            if sym_matrix[i][j] != 0:
                coefficient = sym_matrix[i][j]
                if i == j:
                    # Diagonal terms (e.g., c_i^† c_i)
                    fermion_operator += FermionOperator(((i, 1), (i, 0)),
                                                        coefficient)
                else:
                    # Off-diagonal terms contribute twice (symmetric part)
                    fermion_operator += FermionOperator(((i, 1), (j, 0)),
                                                        coefficient)
                    fermion_operator += FermionOperator(((j, 1), (i, 0)),
                                                        coefficient)

    return fermion_operator


def bliss_cost_fn(params, trun_H, num_modes, num_elec):
    total_number_operator = total_num_op(num_modes)
    mu1 = params[0]
    mu2 = params[1]
    mu3 = params[2]
    # matrix = params[3:].reshape((num_modes, num_modes))
    upper_tri_indices = np.triu_indices(num_modes)
    sym_matrix = np.zeros((num_modes, num_modes))
    sym_matrix[upper_tri_indices] = params[3:3 + int(
        num_modes * (num_modes + 1) // 2)]
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))

    upper_tri_indices2 = np.triu_indices(num_modes)
    sym_matrix2 = np.zeros((num_modes, num_modes))
    sym_matrix2[upper_tri_indices2] = params[3 + int(
        num_modes * (num_modes + 1) // 2):]
    sym_matrix2 = sym_matrix2 + sym_matrix2.T - np.diag(
        np.diag(sym_matrix2))
    truncated_H = FermionOperator()
    truncated_H += trun_H

    truncated_H -= mu1 * (total_number_operator - num_elec)
    truncated_H -= mu2 * (
                total_number_operator * total_number_operator - num_elec ** 2)
    truncated_H -= mu3 * (
                total_number_operator * total_number_operator * total_number_operator - num_elec ** 3)
    O = construct_symmetric_fermion_operator(sym_matrix)
    O_2 = construct_symmetric_fermion_operator(sym_matrix2)
    # O = construct_non_symmetric_fermion_operator(matrix)
    truncated_H -= normal_ordered(O) * (total_number_operator - num_elec)
    truncated_H -= normal_ordered(O_2) * (
                total_number_operator * total_number_operator - num_elec ** 2)
    majo = get_majorana_operator(truncated_H)

    one_norm = 0
    for term, coeff in majo.terms.items():
        one_norm += abs(coeff)
    return one_norm


def bliss_cost_fn_cheaper(params, trun_H, num_modes, num_elec):
    total_number_operator = total_num_op(num_modes)
    mu1 = params[0]
    mu2 = params[1]
    mu3 = params[2]
    # matrix = params[3:].reshape((num_modes, num_modes))
    upper_tri_indices = np.triu_indices(num_modes)
    sym_matrix = np.zeros((num_modes, num_modes))
    sym_matrix[upper_tri_indices] = params[3:]
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))

    truncated_H = FermionOperator()
    truncated_H += trun_H

    truncated_H -= mu1 * (total_number_operator - num_elec)
    truncated_H -= mu2 * (
                total_number_operator * total_number_operator - num_elec ** 2)
    truncated_H -= mu3 * (
                total_number_operator * total_number_operator * total_number_operator - num_elec ** 3)
    O = construct_symmetric_fermion_operator(sym_matrix)
    # O = construct_non_symmetric_fermion_operator(matrix)
    truncated_H -= normal_ordered(O) * (total_number_operator - num_elec)
    majo = get_majorana_operator(truncated_H)

    one_norm = 0
    for term, coeff in majo.terms.items():
        one_norm += abs(coeff)
    return one_norm


def blissed_H(params, trun_H, num_modes, num_elec):
    total_number_operator = total_num_op(num_modes)
    mu1 = params[0]
    mu2 = params[1]
    mu3 = params[2]
    # matrix = params[3:].reshape((num_modes, num_modes))
    upper_tri_indices = np.triu_indices(num_modes)
    sym_matrix = np.zeros((num_modes, num_modes))
    sym_matrix[upper_tri_indices] = params[
                                    3:3 + int(num_modes * (num_modes + 1) // 2)]
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))

    upper_tri_indices2 = np.triu_indices(num_modes)
    sym_matrix2 = np.zeros((num_modes, num_modes))
    sym_matrix2[upper_tri_indices2] = params[3 + int(
        num_modes * (num_modes + 1) // 2):]
    sym_matrix2 = sym_matrix2 + sym_matrix2.T - np.diag(np.diag(sym_matrix2))

    truncated_H = FermionOperator()
    truncated_H += trun_H

    truncated_H -= mu1 * (total_number_operator - num_elec)
    truncated_H -= mu2 * (
                total_number_operator * total_number_operator - num_elec ** 2)
    truncated_H -= mu3 * (
                total_number_operator * total_number_operator * total_number_operator - num_elec ** 3)
    O = construct_symmetric_fermion_operator(sym_matrix)
    O_2 = construct_symmetric_fermion_operator(sym_matrix2)
    # O = construct_non_symmetric_fermion_operator(matrix)
    truncated_H -= normal_ordered(O) * (total_number_operator - num_elec)
    truncated_H -= normal_ordered(O_2) * (
                total_number_operator * total_number_operator - num_elec ** 2)

    return truncated_H


def blissed_H_cheap(params, trun_H, num_modes, num_elec):
    total_number_operator = total_num_op(num_modes)
    mu1 = params[0]
    mu2 = params[1]
    mu3 = params[2]
    # matrix = params[3:].reshape((num_modes, num_modes))
    upper_tri_indices = np.triu_indices(num_modes)
    sym_matrix = np.zeros((num_modes, num_modes))
    sym_matrix[upper_tri_indices] = params[3:]
    sym_matrix = sym_matrix + sym_matrix.T - np.diag(np.diag(sym_matrix))

    truncated_H = FermionOperator()
    truncated_H += trun_H

    truncated_H -= mu1 * (total_number_operator - num_elec)
    truncated_H -= mu2 * (
                total_number_operator * total_number_operator - num_elec ** 2)
    truncated_H -= mu3 * (
                total_number_operator * total_number_operator * total_number_operator - num_elec ** 3)
    O = construct_symmetric_fermion_operator(sym_matrix)
    # O = construct_non_symmetric_fermion_operator(matrix)
    truncated_H -= normal_ordered(O) * (total_number_operator - num_elec)

    return truncated_H
