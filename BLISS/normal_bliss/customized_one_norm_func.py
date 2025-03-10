import numpy as np
import random
from openfermion import FermionOperator
from BLISS.Majorana.custom_majorana_transform import get_custom_majorana_operator
import sympy as sp


def get_param_num(n):
    result = (2 * (8) ** 2 + 1) * (4 * (8) - 8) - (8 * (8) - 16) ** 2 // 2
    # return (result - 136) // 2
    return 66


def construct_symmetric_tensor_specific(n, idx_list):
    idx_len = len(idx_list)
    symm_tensor = np.zeros(idx_len)
    for i in range(idx_len):
        t = random.random() * 0.01
        # t = 0
        symm_tensor[i] = t

    return symm_tensor


def symmetric_tensor_from_triangle_specific(T, n, candidate_idx):
    """
    What symmetry do we apply?
    :param T:
    :param n:
    :return:
    """

    tensor = np.zeros((n, n, n, n), dtype=object)
    idx = 0
    # candidate_list = [10, 11] + idx_list
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # if len({i, j, k, l}) == 4:
                    #     evens_ij = sum(1 for x in [i, j] if x % 2 == 0)
                    #     evens_kl = sum(1 for x in [k, l] if x % 2 == 0)
                    #     if evens_ij == evens_kl:
                    #         if any(x in idx_list for x in [i, j, k, l]):
                    #             if not all(x in idx_list for x in [i, j, k, l]):
                    #                 if (i, j, k, l) <= (l, k, j, i):
                    #                     tensor[i, j, k, l] = T[idx]
                    #                     tensor[l, k, j, i] = T[idx]
                    #                     idx += 1
                    if (i, j, k, l) in candidate_idx:
                        tensor[i, j, k, l] = T[idx]
                        tensor[l, k, j, i] = T[idx]
                        idx += 1

    return tensor


def symmetric_tensor_array_specific(name, n, candidate_idx):
    """
    idx_list is the list of indices corresponding to A
    :param name:
    :param n:
    :param idx_list:
    :return:
    """
    symmetric_tensor = []
    # candidate_list = [10, 11] + idx_list
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    # if len({i, j, k, l}) == 4:
                    #     evens_ij = sum(1 for x in [i, j] if x % 2 == 0)
                    #     evens_kl = sum(1 for x in [k, l] if x % 2 == 0)
                    #     if evens_ij == evens_kl:
                    #         if any(x in idx_list for x in [i, j, k, l]):
                    #             if not all(x in idx_list for x in [i, j, k, l]):
                    #                 if (i, j, k, l) <= (l, k, j, i):
                    #                     symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))
                    if (i, j, k, l) in candidate_idx:
                        symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))

    print("Length of Symmetric Tensor", len(symmetric_tensor))
    print("Predicted: ", len(candidate_idx))
    return sp.Matrix(symmetric_tensor)


def tensor_to_ferm_op(tensor, N):
    ferm_op = FermionOperator()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    ferm_op += FermionOperator(f'{i}^ {j} {k}^ {l}', tensor[i, j, k, l])

    return ferm_op


def construct_specific_tensor(n, idx_list):
    idx_len = len(idx_list)
    symm_tensor = np.zeros(idx_len)
    for i in range(idx_len):
        t = random.random() * 0.01
        # t = 0
        symm_tensor[i] = t

    return symm_tensor


def construct_majorana_terms_3_body_specific(ferm_op, N, ne, z, T):
    """Construct the Majorana terms from a parameterized FermionOperator."""
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    t_ferm_op = tensor_to_ferm_op(T, N)

    param_op = (ferm_op
                - z * (total_number_operator ** 3 - ne ** 3)
                - t_ferm_op * (total_number_operator - ne)
                )

    majo = get_custom_majorana_operator(param_op)

    return majo


def generate_analytical_one_norm_3_body_specific(ferm_op, N, ne, idx_list):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    z = sp.symbols('z')
    T = symmetric_tensor_array_specific('T', N, idx_list)

    T_tensor = symmetric_tensor_from_triangle_specific(T, N, idx_list)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_3_body_specific(ferm_op, N, ne, z, T_tensor )

    invariant_terms_counter = 0

    # for term, coeff in majorana_terms.terms.items():
    #     if coeff != 0 and str(coeff)[-1] == 'I':
    #         print(term, coeff)

    # Compute the symbolic one-norm
    one_norm_expr = sum(
        sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if
        term != ())

    # coeff_matrix = []
    # rhs_vector = []
    # constant_terms = []
    # for term, coeff_expr in majorana_terms.terms.items():
    #     if term == ():
    #         continue
    #     coeffs_expanded = sp.expand(coeff_expr).as_coefficients_dict()
    #
    #     row = [float(sp.re(coeffs_expanded.get(var, 0))) for var in
    #            T]  # Real part coefficients
    #     rhs = float(sp.re(coeffs_expanded.get(1, 0)))  # Constant term
    #
    #     coeff_matrix.append(row)
    #     rhs_vector.append(rhs)
    #     constant_terms.append(rhs)
    #
    # coeff_matrix = np.array(coeff_matrix)
    # rhs_vector = np.array(rhs_vector)
    #
    # # Number of constraints
    # n_constraints = len(rhs_vector)
    # n_vars = len(T)  # Number of variables in x
    #
    # # Create auxiliary variables u_i
    # c = np.concatenate(
    #     [np.zeros(n_vars), np.ones(n_constraints)])  # Minimize sum of u_i
    #
    # # Constraints: u_i >= Ax + b and u_i >= -(Ax + b)
    # A_ub = np.vstack([
    #     np.hstack([coeff_matrix, -np.eye(n_constraints)]),  # Ax + b <= u
    #     np.hstack([-coeff_matrix, -np.eye(n_constraints)])  # -(Ax + b) <= u
    # ])
    #
    # print(f"A_ub.shape: {A_ub.shape}")
    #
    # b_ub = np.hstack([rhs_vector, -rhs_vector])
    #
    # # Solve LP
    # res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    #
    # # Output results
    # if res.success:
    #     print(f"Optimal value (1-norm): {res.fun}")
    #     print(f"Optimal u values: {res.x}")
    # else:
    #     print("Linear programming failed to find a solution.")

    one_norm_func = sp.lambdify((z, T), one_norm_expr,
                                modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr
