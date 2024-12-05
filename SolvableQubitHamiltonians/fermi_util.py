from openfermion import *


def get_commutator(hamil_term: FermionOperator, ferm_op: FermionOperator) -> FermionOperator:
    """
    Return the normal ordered commutator of hamil_term and ferm_op.
    :param hamil_term:
    :param ferm_op:
    :return:
    """
    return normal_ordered(hamil_term * ferm_op - ferm_op * hamil_term)





def get_pauli_strings(fermi_op: FermionOperator):
    qubit_op = jordan_wigner(fermi_op)
    pauli_sum = qubit_operator_to_pauli_sum(qubit_op)


