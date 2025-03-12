import openfermion as of
import numpy as np
from scipy.linalg import eigh
import scipy as sp
import math
from StatePreparation.reference_state_utils import *


def get_bk_hf_state(n_qubits, n_occ):
    """
    Given the number of qubits = spin-orbitals and the number of occupations,
    return the Hartree-Fock state as 2^n_qubits state.
    :param n_qubits:
    :param n_occ:
    :return:
    """
    state_string = ""
    for i in range(n_qubits - n_occ):
        state_string += "0"
    for i in range(n_occ):
        state_string += "1"
    bk_basis_state = get_bk_basis_states(state_string, n_qubits)
    index = find_index(bk_basis_state)
    wfs = np.zeros(2 ** n_qubits)
    wfs[index] = 1
    return wfs


if __name__ == '__main__':
    n_qubits = 4
    n_occ = 2
    bk_basis_state = get_bk_basis_states("0011", n_qubits)
    index = find_index(bk_basis_state)
    print(index)

    wfs = np.zeros(2 ** n_qubits)
    wfs[index] = 1

    print(wfs)
