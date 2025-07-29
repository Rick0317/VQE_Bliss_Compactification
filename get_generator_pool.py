from OperatorPools.generalized_fermionic_pool import get_spin_considered_uccsd_anti_hermitian
from OperatorPools.qubit_pool import get_qubit_adapt_pool_operators_only

def get_generator_pool(pool_type, n_site, n_elec):
    if pool_type == 'uccsd_pool':
        return get_spin_considered_uccsd_anti_hermitian(n_site, n_elec)

    if pool_type =='qubit_pool':
        return get_qubit_adapt_pool_operators_only(n_site, n_elec)


    return get_spin_considered_uccsd_anti_hermitian(n_site, n_elec)
