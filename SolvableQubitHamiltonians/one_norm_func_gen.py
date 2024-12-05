import random

import sympy as sp
import numpy as np
from scipy.optimize import minimize
import pickle
from openfermion import FermionOperator
from utils.custom_Majorana_operator import CustomMajoranaOperator
from utils.custom_majorana_transform import get_custom_majorana_operator
from sympy import Abs


def symmetric_matrix(name, n):
    """Create a symbolic symmetric matrix of size n x n with variable entries."""
    elements = {}
    for i in range(n):
        for j in range(i, n):  # Only define upper triangular elements
            elements[(i, j)] = sp.Symbol(f'{name}_{i}{j}')

    # Construct symmetric matrix
    matrix = sp.Matrix(n, n, lambda i, j: elements[(i, j)] if i <= j else elements[(j, i)])
    return matrix


def construct_symmetric_matrix(n):
    symm_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1):
            t = random.random() * 0.01
            # t = 0
            symm_matrix[i, j] = t
            symm_matrix[j, i] = t

    return symm_matrix



def matrix_to_fermion_operator(matrix, N):
    ferm_op = FermionOperator()
    for i in range(N):
        for j in range(N):
            ferm_op += FermionOperator(f'{i}^ {j}', matrix[i, j])

    return ferm_op


def construct_majorana_terms(ferm_op, N, ne, x, y, z, O, O_2):
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


def generate_analytical_one_norm(ferm_op, N, ne):
    """Generate the analytical one-norm of a parameterized Majorana operator."""
    x, y, z= sp.symbols('x y z')
    O = symmetric_matrix('O', N)
    O_2 = symmetric_matrix('O_2', N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms(ferm_op, N, ne, x, y, z, O, O_2)

    # Compute the symbolic one-norm
    one_norm_expr = sum(sp.Abs(coeff) for _, coeff in majorana_terms.terms.items())

    one_norm_func = sp.lambdify((x, y, z, O, O_2), one_norm_expr, modules=['numpy'])

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
    O = symmetric_matrix('O', N)

    # Construct the Majorana terms manually
    majorana_terms = construct_majorana_terms_2_body(ferm_op, N, ne, x, y, O)

    # Compute the symbolic one-norm
    one_norm_expr = sum(sp.Abs(coeff) for _, coeff in majorana_terms.terms.items())

    one_norm_func = sp.lambdify((x, y, O), one_norm_expr, modules=['numpy'])

    print("Analytical 1-Norm Complete")
    return one_norm_func, one_norm_expr


if __name__ == '__main__':

    filename = f'ham_lib/truncated_hamiltonian.pkl'
    with open(filename, 'rb') as f:
        H = pickle.load(f)

    N = 8
    ne = 4
    one_norm_func, one_norm_expr = generate_analytical_one_norm(H, N, ne)


    # Optimization wrapper
    def optimization_wrapper(params):
        x_val = params[0]
        y_val = params[1]
        z_val = params[2]
        o_vals = params[3:67]
        o_vals2 = params[67:]
        return one_norm_func(x_val, y_val, z_val, o_vals, o_vals2)

    x_val = 0
    y_val = 0
    z_val = 0
    o_val = construct_symmetric_matrix(N)
    o_val2 = construct_symmetric_matrix(N)

    o_val_flattened = o_val.flatten()
    o_val_flattened2 = o_val2.flatten()
    print(o_val_flattened.shape)
    print(f"One-norm at x = {x_val}, y = {y_val}, z= {z_val}: {one_norm_func(x_val, y_val, z_val, o_val_flattened, o_val_flattened2)}")

    # Minimize
    initial_guess = np.concatenate((np.array([0, 0, 0, 0]), o_val_flattened, o_val_flattened2))
    res = minimize(optimization_wrapper, initial_guess, method='BFGS', options={'gtol': 1e-300, 'disp': True, 'maxiter': 300})

    print(f"Optimal values: {res.x}")
    print(f"Minimum one-norm: {res.fun}")

