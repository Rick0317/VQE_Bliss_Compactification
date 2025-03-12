import pickle

import numpy as np
from openfermion import (
normal_ordered,
bravyi_kitaev,
variance,
get_sparse_operator as gso,
expectation
)
from BLISS.normal_bliss.customized_bliss_package import *
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
from StatePreparation.hartree_fock import *

def abs_of_dict_value(x):
    return np.abs(x[1])


def ferm_to_qubit(H: FermionOperator):
    Hqub = bravyi_kitaev(H)
    return Hqub


def commutator_variance(decomp, N, psi):
    """
    Computes the variance metric of the sum of fragments with the wfs psi.
    """

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance(gso(frag, N), psi)
    return np.sum((vars) ** (1 / 2)) ** 2


def sz_operator(num_spin_orbitals):
    """Creates the total Sz operator in OpenFermion"""
    sz_op = FermionOperator()
    for i in range(num_spin_orbitals // 2):  # Iterate over spatial orbitals
        spin_up = 2 * i     # Even indices are spin-up
        spin_down = 2 * i + 1  # Odd indices are spin-down
        sz_op += 0.5 * (FermionOperator(((spin_up, 1), (spin_up, 0))) -
                        FermionOperator(((spin_down, 1), (spin_down, 0))))
    return sz_op


def get_hartree_fock(num_spin_orbitals, occupied_orbitals):
    """
    Prepare the Hartree-Fock state in qubit space.

    Args:
        num_spin_orbitals (int): Total number of spin orbitals.
        occupied_orbitals (list): List of indices of occupied spin orbitals.

    Returns:
        np.ndarray: The Hartree-Fock state vector.
    """
    state = np.zeros(2 ** num_spin_orbitals, dtype=np.complex128)
    binary_index = sum(2 ** q for q in occupied_orbitals)
    state[binary_index] = 1

    return state


if __name__ == '__main__':
    N = 8
    Ne = 4
    mol_name = "h4"

    filename = f'../ham_lib/h4_sto-3g.pkl'

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
        hamil_copy1 = copy_ferm_hamiltonian(ordered_hamil)
        hamil_copy2 = copy_ferm_hamiltonian(ordered_hamil)
        g_copy1 = copy_ferm_hamiltonian(G)
        g_copy2 = copy_ferm_hamiltonian(G)

        commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2
        H = normal_ordered(commutator)
        H_in_q = ferm_to_qubit(H)

        hf_state = get_bk_hf_state(N, Ne)

        number_operator = FermionOperator()
        for mode in range(N):
            number_operator += FermionOperator(f"{mode}^ {mode}")  # a_i† a_i

        # Convert to qubit operator using Jordan-Wigner transformation
        number_operator_qubit = bravyi_kitaev(number_operator)

        print(
            f"Particle number Qubit: {expectation(gso(number_operator_qubit), hf_state)}")

        print(
            f"Sz Qubit: {expectation(gso(ferm_to_qubit(sz_operator(N)), N), hf_state)}")
