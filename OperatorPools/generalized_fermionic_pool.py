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
    operator_pool = set()
    for p in range(site):
        for q in range(site):
            operator_pool.add(get_anti_hermitian_one_body((p, q)))
            for r in range(site):
                for s in range(site):
                    operator_pool.add(get_anti_hermitian_two_body((p, q, r, s)))

    return operator_pool
