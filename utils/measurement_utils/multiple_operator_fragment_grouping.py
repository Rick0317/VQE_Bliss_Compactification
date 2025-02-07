"""
Group Pauli strings into fragments that are shared by multiple operators.
"""
from SolvableQubitHamiltonians.qwc_decomposition import qwc_decomposition


def choose_best(list1, list2):
    """
    Given two Pauli lists, construct the list that captures the most of the fragments
    :param list1: Fragment decomposition list 1
    :param list2: Fragment decomposition list 2
    :return:
    """
    common_items = list(set(list1) & set(list2))
    

def get_shared_fragment(op_list):
    """
    Group Pauli strings into fragments that are shared by multiple operators.
    :param op_list:
    :return:
    """
    shared_fragments = qwc_decomposition(op_list[0])
    for operator in op_list[1:]:


