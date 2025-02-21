"""
In this file, we have code that finds fragments of the Hamiltonian
that allows overlapping Pauli strings amongs multiple fragments.
"""

from openfermion import QubitOperator
import numpy as np
import openfermion as of
from copy import deepcopy


def overlapping_sorted_insertion_li_grouping(H: QubitOperator, gs, n_qubits, ev_dict, commutativity='fc'):
    pws = qubu.get_pauliword_list(H)
    pws.sort(key=qubu.get_pauli_word_coefficients_size, reverse=True)
    n_qubits = of.count_qubits(H)
    no_groups, ev_dict = sorted_insertion_li_grouping(H, gs, n_qubits, ev_dict, commutativity)
    o_groups = deepcopy(no_groups)
    for pw in pws: # Loop through pw.
         for group in o_groups:
             if group_commutes_w_pw(pw, group, commutativity) and pw_linear_independent_w_group(pw, group, n_qubits):
                 if not (pw in group): group.append(pw)
    return no_groups, o_groups, ev_dict


def get_pauli_word_coefficient(P: QubitOperator, ghosts=None):
    """Given a single pauli word P, extract its coefficient.
    """
    if ghosts is not None:
       if P in ghosts:
          coeffs = [0.0]
       else:
          coeffs = list(P.terms.values())
    else:
       coeffs = list(P.terms.values())
    return coeffs[0]


def get_pauli_word_coefficients_size(P: QubitOperator):
    """Given a single pauli word P, extract the size of its coefficient.
    """
    return np.abs(get_pauli_word_coefficient(P))


def get_pauliword_list(H: QubitOperator, ignore_identity=True):
    """Obtain a list of pauli words in H.
    """
    pws = []
    for pw, val in H.terms.items():
        if ignore_identity:
            if len(pw) == 0:
                continue
        pws.append(QubitOperator(term=pw, coefficient=val))
    return pws


def overlapping_si_fragmentation(H: QubitOperator, decomposition):
    """
    Create an overlapping fragmentation of the Hamiltonian H such that there
    are Pauli String shared among multiple fragments.
    :param H:
    :param decomposition: The original decomposition of the Hamiltonian
    without overlapping
    :return:
    """
    pws = get_pauliword_list(H)
    pws.sort(key=get_pauli_word_coefficients_size, reverse=True)

    n_qubits = of.count_qubits(H)

    o_groups = deepcopy(decomposition)

    # Loop through the list of Pauli words
    for pw in pws:

        # The list of fragments
        for group in o_groups:

            # Check whether the Pauli word commute with the fragment
            if (group_commutes_w_pw(pw, group, "fc") and
                    pw_linear_independent_w_group(pw, group, n_qubits)):

                if not (pw in group):
                    group.append(pw)

    return o_groups

def group_commutes_w_pw(pw, gp, commutativity):
    commuting = True
    for pw2 in gp:
        if not(is_commuting(pw, pw2, commutativity)):
            commuting = False
            break
    return commuting


def is_commuting(ipw, jpw, condition='fc'):
    """Check whether ipw and jpw are FC or QWC.
    Args:
        ipw, jpw (QubitOperator): Single pauli-words to be checked for commutativity.
        condition (str): "qwc" or "fc", indicates the type of commutativity to check.

    Returns:
        is_commuting (bool): Whether ipw and jpw commute by the specified condition.
    """
    return of.commutator(get_pauli_word(ipw), get_pauli_word(jpw)) == QubitOperator.zero()


def get_pauli_word(P: QubitOperator):
    """Given a single pauli word P, extract the same word with coefficient 1.
    """
    words = list(P.terms.keys())
    if len(words) != 1:
        raise(ValueError("P given is not a single pauli word"))
    return QubitOperator(words[0])
