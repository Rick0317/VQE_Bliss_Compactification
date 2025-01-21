import random

import sympy as sp
import numpy as np
from scipy.optimize import minimize
import pickle
from openfermion import FermionOperator
from utils.custom_Majorana_operator import CustomMajoranaOperator
from utils.custom_majorana_transform import get_custom_majorana_operator
from sympy import Abs


def upper_triangle_array(name, n):
    """Create a 1D array of symbolic variables representing the upper triangle of a symmetric matrix."""
    upper_triangle = []
    for i in range(n):
        for j in range(i, n):  # Only define upper triangular elements
            upper_triangle.append(sp.Symbol(f'{name}_{i}{j}'))
    return sp.Matrix(upper_triangle)


def symmetric_tensor_array(name, n):
    symmetric_tensor = []
    for i in range(n):
        for j in range(i, n):
            for k in range(n):
                for l in range(k, n):
                    symmetric_tensor.append(sp.Symbol(f"{name}_{i}{j}{k}{l}"))

    print("Length of Symmetric Tensor", len(symmetric_tensor))

    return sp.Matrix(symmetric_tensor)

def symmetric_matrix_from_upper_triangle(O, n):
    """Construct a symmetric matrix of size n x n from a 1D array of upper triangular elements."""
    if len(O) != n * (n + 1) // 2:
        raise ValueError(
            "Length of O must be n(n+1)/2 for an n x n symmetric matrix.")

    # Map elements from O to the symmetric matrix
    matrix = sp.Matrix.zeros(n, n)
    idx = 0
    for i in range(n):
        for j in range(i, n):  # Fill upper triangular part
            matrix[i, j] = O[idx]
            matrix[j, i] = O[idx]  # Ensure symmetry
            idx += 1
    return matrix


def construct_symmetric_matrix(n):
    symm_matrix = np.zeros(int(n * (n + 1) // 2))
    for i in range(int(n * (n + 1) // 2)):
        t = random.random() * 0.01
        # t = 0
        symm_matrix[i] = t

    return symm_matrix


def construct_symmetric_tensor(n):
    symm_tensor = np.zeros(int(n * (n + 1) // 2) ** 2)
    for i in range(int(n * (n + 1) // 2) ** 2):
        t = random.random() * 0.01
        # t = 0
        symm_tensor[i] = t

    return symm_tensor

def construct_specific_tensor(n, idx_list):
    idx_len = len(idx_list)
    symm_tensor = np.zeros(int(idx_len ** 2 * (idx_len ** 2 + 1) // 2))
    for i in range(int(idx_len ** 2 * (idx_len ** 2 + 1) // 2)):
        t = random.random() * 0.01
        # t = 0
        symm_tensor[i] = t

    return symm_tensor



def matrix_to_fermion_operator(matrix, N):
    ferm_op = FermionOperator()
    for i in range(N):
        for j in range(N):
            ferm_op += FermionOperator(f'{i}^ {j}', matrix[i, j])

    return ferm_op


def tensor_to_ferm_op(tensor, N):
    ferm_op = FermionOperator()
    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    ferm_op += FermionOperator(f'{i}^ {j} {k}^ {l}', tensor[i, j, k, l])

    return ferm_op

def construct_majorana_terms(ferm_op, N, ne, x, y, z, w, O, O_2):
    """Construct the Majorana terms from a parameterized FermionOperator."""
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    o_ferm_op = matrix_to_fermion_operator(O, N)
    o2_ferm_op = matrix_to_fermion_operator(O_2, N)

    param_op = (ferm_op - x * (total_number_operator - ne)
                - y * (total_number_operator ** 2 - ne ** 2)
                - z * (total_number_operator ** 3 - ne ** 3)
                - w * (total_number_operator ** 4 - ne ** 4)
                - o_ferm_op * (total_number_operator - ne)
                - o2_ferm_op * (total_number_operator ** 2 - ne ** 2))

    majo = get_custom_majorana_operator(param_op)

    return majo


def generate_analytical_one_norm(ferm_op, N, ne):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    x, y, z, w = sp.symbols('x y z w')
    O = upper_triangle_array('O', N)
    O_2 = upper_triangle_array('O_2', N)

    O_matrix = symmetric_matrix_from_upper_triangle(O, N)
    O2_matrix = symmetric_matrix_from_upper_triangle(O_2, N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms(ferm_op, N, ne, x, y, z, w, O_matrix, O2_matrix)

    # Compute the symbolic one-norm
    one_norm_expr = sum(sp.Abs(coeff) for _, coeff in majorana_terms.terms.items())

    one_norm_func = sp.lambdify((x, y, z, w, O, O_2), one_norm_expr, modules=['numpy'])

    # with open('one_norm_function.py', 'w') as f:
    #     f.write(f"import numpy as np\n")
    #     f.write(f"import sympy as sp\n")
    #     f.write(f"x, y, z, w, O, O_2 = {sp.symbols('x y z w O O_2 O_3')}\n")
    #     f.write(f"one_norm_expr = {repr(one_norm_expr)}\n")
    #     f.write(
    #         f"one_norm_func = {sp.lambdify((x, y, z, w, O, O_2, O_3), one_norm_expr, modules=['numpy'])}\n")
    #
    # print(f"Lambdified function saved in 'one_norm_function.py'")
    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr


def construct_majorana_terms_2_body(ferm_op, N, ne, x, y, O):
    """Construct the Majorana terms from a parameterized FermionOperator."""
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    o_ferm_op = matrix_to_fermion_operator(O, N)

    param_op = (ferm_op - x * (total_number_operator - ne)
                - y * (total_number_operator ** 2 - ne ** 2)
                - o_ferm_op * (total_number_operator - ne))

    majo = get_custom_majorana_operator(param_op)

    return majo


def generate_analytical_one_norm_2_body(ferm_op, N, ne):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    x, y= sp.symbols('x y')
    O = upper_triangle_array('O', N)

    O_matrix = symmetric_matrix_from_upper_triangle(O, N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_2_body(ferm_op, N, ne, x, y, O_matrix)

    # Compute the symbolic one-norm
    one_norm_expr = sum(sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if term != ())

    one_norm_func = sp.lambdify((x, y, O), one_norm_expr, modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr

def construct_majorana_terms_3_body(ferm_op, N, ne, x, y, z, O, O_2):
    """Construct the Majorana terms from a parameterized FermionOperator."""
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    o_ferm_op = matrix_to_fermion_operator(O, N)
    o2_ferm_op = matrix_to_fermion_operator(O_2, N)

    param_op = (ferm_op - x * (total_number_operator - ne)
                - y * (total_number_operator ** 2 - ne ** 2)
                - z * (total_number_operator ** 3 - ne ** 3)
                - o_ferm_op * (total_number_operator - ne)
                - o2_ferm_op * (total_number_operator ** 2 - ne ** 2))

    majo = get_custom_majorana_operator(param_op)

    return majo


def construct_majorana_terms_3_body_simple(ferm_op, N, ne, z, T):
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


def construct_majorana_terms_3_body_cheap(ferm_op, N, ne, z, O, O2):
    """Construct the Majorana terms from a parameterized FermionOperator."""
    total_number_operator = FermionOperator()
    for mode in range(N):
        total_number_operator += FermionOperator(((mode, 1), (mode, 0)))

    o_ferm_op = matrix_to_fermion_operator(O, N)
    o2_ferm_op = matrix_to_fermion_operator(O2, N)

    param_op = (ferm_op
                - z * (total_number_operator ** 3 - ne ** 3)
                - o_ferm_op * o2_ferm_op * (total_number_operator - ne)
                )

    majo = get_custom_majorana_operator(param_op)

    return majo




def generate_analytical_one_norm_3_body_simple(ferm_op, N, ne):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    z = sp.symbols('z')
    T = symmetric_tensor_array('T', N)

    T_tensor = symmetric_tensor_from_triangle(T, N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_3_body_simple(ferm_op, N, ne, z, T_tensor)

    # Compute the symbolic one-norm
    one_norm_expr = sum(
        sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if
        term != ())

    one_norm_func = sp.lambdify((z, T), one_norm_expr,
                                modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr


def generate_analytical_one_norm_3_body_cheap(ferm_op, N, ne):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    z = sp.symbols('z')
    O = upper_triangle_array('O', N)
    O_2 = upper_triangle_array('O2', N)

    O_matrix = symmetric_matrix_from_upper_triangle(O, N)
    O2_matrix = symmetric_matrix_from_upper_triangle(O_2, N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_3_body_cheap(ferm_op, N, ne, z, O_matrix, O2_matrix)

    # Compute the symbolic one-norm
    one_norm_expr = sum(
        sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if
        term != ())

    one_norm_func = sp.lambdify((z, O, O_2), one_norm_expr,
                                modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr




def generate_analytical_one_norm_3_body(ferm_op, N, ne):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    x, y, z = sp.symbols('x y z')
    O = upper_triangle_array('O', N)
    O_2 = upper_triangle_array('O_2', N)

    O_matrix = symmetric_matrix_from_upper_triangle(O, N)
    O2_matrix = symmetric_matrix_from_upper_triangle(O_2, N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_3_body(ferm_op, N, ne, x, y, z, O_matrix, O2_matrix)

    # Compute the symbolic one-norm
    one_norm_expr = sum(sp.Abs(coeff) for term, coeff in majorana_terms.terms.items() if term != ())

    one_norm_func = sp.lambdify((x, y, z, O, O_2), one_norm_expr, modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr


if __name__ == '__main__':

    pass

