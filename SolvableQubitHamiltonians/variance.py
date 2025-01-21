from openfermion import (
    QubitOperator as Q,
    commutator,
    anticommutator,
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    bravyi_kitaev,
    FermionOperator,
    get_majorana_operator,
    jw_get_ground_state_at_particle_number as jwggs,
    normal_ordered
)
import os
import csv

from SolvableQubitHamiltonians.main_utils_partitioning import N_QUBITS, copy_hamiltonian, sorted_insertion_decomposition
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
import pickle
from SolvableQubitHamiltonians.qwc_decomposition import qwc_decomposition
from scipy.optimize import minimize
from utils.bliss_package import *
from utils.customized_bliss_package import *
from utils.indices_filter import filter_indices

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
    moltag = 'h2O'
    methodtag = 'fc'

    N = 14
    Ne = 10

    filename = f'ham_lib/h2o_fer.bin'
    with open(filename, 'rb') as f:
        Hamil = pickle.load(f)



    ordered_hamil = normal_ordered(Hamil)
    com_name_list = [
        '3^ 2^ 0 1',
        '6^ 3^ 7 0',
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
        FermionOperator('6^ 3^ 7 0') - FermionOperator('0^ 7^ 3 6'),
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

    for i in range(len(anti_com_list)):
        G = anti_com_list[i]
        hamil_copy1 = copy_ferm_hamiltonian(Hamil)
        hamil_copy2 = copy_ferm_hamiltonian(Hamil)
        g_copy1 = copy_ferm_hamiltonian(G)
        g_copy2 = copy_ferm_hamiltonian(G)

        commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2
        H = normal_ordered(commutator)

        two_body_h = FermionOperator()
        three_body_h = FermionOperator()
        # two_body_list = []
        # for term, coeff in H.terms.items():
        #     if len(term) == 4:
        #         parity_0 = term[0][0] % 2  # Parity of term[0][0]
        #         parity_2 = term[2][0] % 2  # Parity of term[2][0]
        #         parity_3 = term[3][0] % 2  # Parity of term[3][0]
        #         if parity_0 == parity_2:
        #             two_body_list.append((term[0][0], term[2][0], term[1][0], term[3][0]))
        #         elif parity_0 == parity_3:
        #             two_body_list.append(
        #                 (term[0][0], term[3][0], term[1][0], term[2][0]))
        #
        #         two_body_h += FermionOperator(term, coeff)
        #     if len(term) == 6:
        #         three_body_h += FermionOperator(term, coeff)

        two_body_list = filter_indices(H, N, Ne)
        if len(two_body_list) != 0:
            majo = get_majorana_operator(two_body_h)

            one_norm = 0
            for term, coeff in majo.terms.items():
                if term != ():
                    one_norm += abs(coeff)

            print("Original 1-Norm 2-body", one_norm)

            majo = get_majorana_operator(three_body_h)

            one_norm = 0
            for term, coeff in majo.terms.items():
                if term != ():
                    one_norm += abs(coeff)

            print("Original 1-Norm 3-body", one_norm)
            # H = Hamil

            H_q = ferm_to_qubit(H)
            print("Commutator Obtained in Fermion and Qubit space")
            H_copy = copy_hamiltonian(H_q)
            # decomp = sorted_insertion_decomposition(H_copy, methodtag)
            decomp = qwc_decomposition(H_copy)
            print("Original Decomposition Complete")
            var = commutator_variance(H_q, decomp, N, Ne)
            print(f"Original variance: {var}")

            majo = get_majorana_operator(H)

            one_norm = 0
            for term, coeff in majo.terms.items():
                if term != ():
                    one_norm += abs(coeff)

            print("Original 1-Norm", one_norm)

            H_before_bliss1 = copy_ferm_hamiltonian(H)
            optimization_wrapper, initial_guess = optimize_bliss_mu3_customizable(H_before_bliss1, N, Ne, two_body_list)
            opt_method = 'Nelder-Mead 100k Iter'

            # res = minimize(optimization_wrapper, initial_guess, method='BFGS',
            #                options={'gtol': 1e-300, 'disp': True, 'maxiter': 600, 'eps': 1e-2})
            # Try a different method that doesn't require gradients
            res = minimize(optimization_wrapper, initial_guess, method='Nelder-Mead',
                           options={'disp': True, 'maxiter': 100000})

            H_before_modification = copy_ferm_hamiltonian(H)
            bliss_output = construct_H_bliss_mu3_customizable(H_before_modification, res.x, N, Ne, two_body_list)

            print(f"First 5 parameters: {res.x[:5]}")

            H_before_bliss_test = copy_ferm_hamiltonian(H)
            H_bliss_output_test = copy_ferm_hamiltonian(bliss_output)
            # check_correctness(H_before_bliss_test, H_bliss_output_test, Ne)
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
            blissed_vars = commutator_variance(H_decompose_copy, blissed_decomp.copy(), N, Ne)
            print(f"Blissed variance: {blissed_vars}")


            file_name = f"bliss_commutator_result_{moltag}.csv"

            file_exists = os.path.isfile(file_name)
            # Open the file in append mode or write mode
            with open(file_name, mode='a' if file_exists else 'w', newline='',
                      encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Write the header only if the file doesn't exist
                if not file_exists:
                    writer.writerow(
                        ['Commutator', 'Original 1-Norm', 'Blissed 1-Norm', 'Original variance', 'Blissed variance', 'Optimization method', 'term'])

                # Write the data
                writer.writerow(
                    [com_name_list[i], one_norm, blissed_one_norm, var.real, blissed_vars.real, opt_method, term])

            # H_bliss_copy2 = copy_hamiltonian(H_bliss_q)
            # blissed_vars1, psi2 = commutator_variance(H_bliss_copy2, blissed_decomp.copy(),
            #                                                  N, Ne)

            # print(f"GS difference: ", np.linalg.norm(psi - psi2))
