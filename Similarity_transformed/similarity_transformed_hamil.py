import pickle
from utils.bliss_package import (
    optimize_bliss_mu123_o12,
    optimization_bliss_mu12_o1,
    construct_H_bliss_m12_o1,
    construct_H_bliss_mu123_o12)
from scipy.optimize import minimize
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
from pathlib import Path
from openfermion import get_majorana_operator, FermionOperator, normal_ordered
import numpy as np
import h5py
from utils.ferm_utils import *
from SolvableQubitHamiltonians.variance import (
    commutator_variance,
    qwc_decomposition,
    ferm_to_qubit,
    commutator_variance_subspace)

from SolvableQubitHamiltonians.main_utils_partitioning import sorted_insertion_decomposition

def transform_to_chemist_notation(operator):
    """Transform a FermionOperator into chemist notation."""
    transformed_operator = FermionOperator()
    for term, coefficient in operator.terms.items():
        # Reorder the term to chemist notation
        reordered_term = []
        chemist_order = []
        for op in term:
            if op[1] == 1:  # Creation operator
                chemist_order.append(op)
            else:  # Annihilation operator
                reordered_term += chemist_order
                chemist_order = [op]
        reordered_term += chemist_order

        # Calculate the sign change for reordering
        sign = calculate_sign(term, reordered_term)

        # Add the transformed term to the new operator
        transformed_operator += FermionOperator(tuple(reordered_term), coefficient * sign)

    return transformed_operator

def calculate_sign(original_term, reordered_term):
    """Calculate the sign difference due to reordering."""
    original_indices = [op[0] for op in original_term]
    reordered_indices = [op[0] for op in reordered_term]
    swaps = 0
    for i in range(len(original_indices)):
        for j in range(i + 1, len(original_indices)):
            if original_indices[i] > original_indices[j]:
                swaps += 1
    return (-1) ** swaps


def save_tensors_to_h5(h1e, h2e, h3e, q4e, filename="tensors.h5"):
    """
    Save h1e and h2e tensors to an HDF5 file.

    Parameters:
        h1e (numpy.ndarray): One-body interaction tensor.
        h2e (numpy.ndarray): Two-body interaction tensor.
        filename (str): Name of the output HDF5 file.
    """
    with h5py.File(filename, "w") as h5file:
        # Create a group for BLISS_HAM
        bliss_ham = h5file.create_group("BLISS_HAM")
        # Save the tensors
        bliss_ham.create_dataset("obt", data=h1e, compression="gzip",
                                 compression_opts=9)
        bliss_ham.create_dataset("tbt", data=h2e, compression="gzip",
                                 compression_opts=9)
        bliss_ham.create_dataset("threebt", data=h3e, compression="gzip",
                                 compression_opts=9)
        bliss_ham.create_dataset("fourbt", data=q4e, compression="gzip",
                                 compression_opts=9)


    print(f"Tensors saved to {filename}")


if __name__ == "__main__":
    filename = Path('../utils/ham_lib/truncated_hamiltonian.pkl')
    with open(filename, 'rb') as f:
        H = pickle.load(f)

    N = 8
    Ne = 4
    molecule = "H4"

    h1e, g2e, t3e, q4e = ferm_op_to_tensors(H, N)

    save_tensors_to_h5(h1e, g2e, t3e, q4e, filename=f"{molecule}_transformed.h5")


    # original_H1 = copy_ferm_hamiltonian(H)
    # H_q_copy = ferm_to_qubit(original_H1)
    # decomp = sorted_insertion_decomposition(H_q_copy, 'fc')
    # # decomp = qwc_decomposition(H_q_copy)
    # print("Original Decomposition Complete")
    #
    # original_H2 = copy_ferm_hamiltonian(H)
    # H_q = ferm_to_qubit(original_H2)
    # var = commutator_variance(H_q, decomp, N, Ne)
    # print(f"Original variance: {var}")
    #
    # majo = get_majorana_operator(H)
    # one_norm = 0
    # for term, coeff in majo.terms.items():
    #     if term != ():
    #         one_norm += abs(coeff)
    #
    # print("Original 1-Norm", one_norm)
    #
    # H_before_bliss1 = copy_ferm_hamiltonian(H)
    # optimization_wrapper, initial_guess = optimization_bliss_mu12_o1(H_before_bliss1,
    #                                                             N, Ne)
    #
    # res = minimize(optimization_wrapper, initial_guess, method='BFGS',
    #                options={'gtol': 1e-300, 'disp': True, 'maxiter': 600})
    #
    # H_before_modification = copy_ferm_hamiltonian(H)
    # bliss_output = construct_H_bliss_m12_o1(H_before_modification, res.x, N, Ne)
    #
    # H_bliss_output = copy_ferm_hamiltonian(bliss_output)
    # majo_blissed = get_majorana_operator(H_bliss_output)
    # one_norm = 0
    # for term, coeff in majo_blissed.terms.items():
    #     if term != ():
    #         one_norm += abs(coeff)
    #
    # print("Blissed 1-Norm", one_norm)
    #
    # H_bliss_output2 = copy_ferm_hamiltonian(bliss_output)
    # chemist_fermop = chemist_ordered(H_bliss_output2)
    #
    # H_bliss_output3 = copy_ferm_hamiltonian(bliss_output)
    # H_q_copy = ferm_to_qubit(H_bliss_output2)
    # decomp = qwc_decomposition(H_q_copy)
    # print("Blissed Decomposition Complete")
    # H_q = ferm_to_qubit(H_bliss_output2)
    # var = commutator_variance(H_q, decomp, N, Ne)
    # print(f"BLISS variance: {var}")
    #
    # h1e, g2e, t3e, q4e = ferm_op_to_tensors(chemist_fermop, N)
    #
    # save_tensors_to_h5(h1e, g2e, t3e, q4e, filename=f"{molecule}_original.h5")





