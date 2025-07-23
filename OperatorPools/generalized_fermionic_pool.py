"""
The Generalized Fermionic Pool
"""
from openfermion import FermionOperator


def get_anti_hermitian_one_body(indices: tuple):
    """
    Get the one-body anti-Hermitian generator
    :param indices:
    :return:
    """
    p, q = indices
    return FermionOperator(f'{p}^ {q}') - FermionOperator(f'{q}^ {p}')


def get_anti_hermitian_two_body(indices: tuple):
    """
    Get the two-body anti-Hermitian generator
    :param indices:
    :return:
    """
    p, q, r, s = indices
    return FermionOperator(f'{p}^ {q}^ {r} {s}') - FermionOperator(f'{s}^ {r}^ {q} {p}')


def get_all_anti_hermitian(site: int):
    """
    quartic size of the generator pool.
    :param site:
    :return:
    """
    operator_pool = []
    for p in range(site):
        for q in range(site):
            operator_pool.append(get_anti_hermitian_one_body((p, q)))
            for r in range(site):
                for s in range(site):
                    operator_pool.append(get_anti_hermitian_two_body((p, q, r, s)))

    return operator_pool


def get_all_uccsd_anti_hermitian(site: int, n_elec:int):
    """.
    quartic size of the generator pool.
    :param site:
    :return:
    """
    operator_pool = []
    for p in range(n_elec, site):
        for q in range(n_elec):
            operator_pool.append(get_anti_hermitian_one_body((p, q)))
            for r in range(n_elec, site):
                for s in range(n_elec):
                    operator_pool.append(get_anti_hermitian_two_body((p, r, s, q)))

    return operator_pool
