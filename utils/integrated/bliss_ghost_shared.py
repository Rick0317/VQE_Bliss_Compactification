import pickle
from openfermion import (
normal_ordered,
FermionOperator,
QubitOperator,
get_majorana_operator,
bravyi_kitaev,
variance,
get_ground_state as ggs,
get_sparse_operator as gso,
expectation
)
from SolvableQubitHamiltonians.qwc_decomposition import qwc_decomposition
from scipy.optimize import minimize
from utils.customized_bliss_package import *
from utils.indices_filter import filter_indices
from SolvableQubitHamiltonians.utils_basic import copy_ferm_hamiltonian
from SolvableQubitHamiltonians.main_utils_partitioning import copy_hamiltonian
from utils.measurement_utils.ghost_paulis import update_decomp_w_ghost_paulis
from utils.measurement_utils.shared_paulis import (
    get_sharable_paulis,
    get_share_pauli_only_decomp,
    get_pw_grp_idxes_no_fix_len,
    get_overlapping_decomp,
    get_sharable_only_decomp,
    get_coefficient_orderings,
    get_all_pw_indices,
    get_pauli_coeff_map,
    qubit_op_to_list
)

from utils.measurement_utils.coefficient_optimizer import (
get_split_measurement_variance_unconstrained,
get_meas_alloc,
optimize_coeffs,

)

def abs_of_dict_value(x):
    return np.abs(x[1])


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

N = 8
Ne = 4

filename = f'../../SolvableQubitHamiltonians/ham_lib/h4_sto-3g.pkl'
with open(filename, 'rb') as f:
    Hamil = pickle.load(f)

ordered_hamil = normal_ordered(Hamil)


G = FermionOperator('3^ 2^ 0 1') - FermionOperator('1^ 0^ 2 3')
hamil_copy1 = copy_ferm_hamiltonian(Hamil)
hamil_copy2 = copy_ferm_hamiltonian(Hamil)
g_copy1 = copy_ferm_hamiltonian(G)
g_copy2 = copy_ferm_hamiltonian(G)

commutator = hamil_copy1 * g_copy1 - g_copy2 * hamil_copy2
H = normal_ordered(commutator)


psi = ggs(gso(H, N))[1]
original_exp = expectation(gso(H, N), psi)
print(f"Original Expectation value: {original_exp}")


two_body_h = FermionOperator()
three_body_h = FermionOperator()


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

    blissed_vars = commutator_variance(H_decompose_copy,
                                       blissed_decomp.copy(), N, Ne)

    print(f"Blissed variance: {blissed_vars}")

    # Apply Ghost Pauli method
    psi = ggs(gso(H_bliss_copy, N))[1]
    update_decomp_w_ghost_paulis(psi, N, blissed_decomp)

    blissed_expectation = expectation(gso(H_bliss_copy), psi)
    print(f"Blissed expectation: {blissed_expectation}")

    blissed_ghost_vars = commutator_variance(H_decompose_copy, blissed_decomp.copy(), N, Ne)
    print(f"Blissed Ghost variance: {blissed_ghost_vars}")

    # Shared Pauli

    coeff_map = get_pauli_coeff_map(decomp)

    # 1
    sharable_paulis_dict, sharable_paulis_list, sharable_pauli_indices_list = get_sharable_paulis(
        decomp)
    sharable_paulis_fixed_list = [indices[-1] for indices in
                                  sharable_pauli_indices_list]

    # 2
    fragment_idx_to_sharable_paulis, pw_grp_idxes_no_fix, pw_grp_idxes_fix = get_share_pauli_only_decomp(
        sharable_paulis_dict)

    pw_grp_idxes_no_fix_len = get_pw_grp_idxes_no_fix_len(pw_grp_idxes_no_fix)

    # 3
    all_sharable_contained_decomp, all_sharable_no_fixed_decomp = get_overlapping_decomp(
        sharable_paulis_dict, decomp, pw_grp_idxes_fix)

    # 4
    sharable_only_decomp, sharable_only_no_fixed_decomp = get_sharable_only_decomp(
        sharable_paulis_dict, decomp, pw_grp_idxes_fix, coeff_map)

    # 5
    fixed_grp, all_sharable_contained_no_fixed_decomp, new_sharable_pauli_indices_list, new_grp_len_list, new_grp_idx_start \
        = get_coefficient_orderings(sharable_only_decomp, sharable_paulis_list,
                                    sharable_pauli_indices_list, coeff_map)

    # 6
    pw_indices = get_all_pw_indices(sharable_paulis_list,
                                    sharable_only_no_fixed_decomp,
                                    pw_grp_idxes_no_fix, new_grp_idx_start)

    meas_alloc = get_meas_alloc(decomp)

    # Get the linear equation for finding the gradient descent direction
    matrix, b = optimize_coeffs(
        pw_grp_idxes_fix,
        pw_grp_idxes_no_fix_len,
        meas_alloc,
        sharable_only_decomp,
        sharable_only_no_fixed_decomp,
        pw_indices, psi, N, decomp, alpha=0.001)

    sol = np.linalg.lstsq(matrix, b.T, rcond=None)
    x0 = sol[0]
    coeff = x0.T[0]

    # Update the fragment by modifying the coefficients of shared Pauli operators.
    variance, meas_alloc, var, fixed_grp_coefficients, measured_groups = (
        get_split_measurement_variance_unconstrained(
            coeff,
            decomp,
            sharable_paulis_list,
            sharable_paulis_fixed_list,
            sharable_only_no_fixed_decomp,
            pw_grp_idxes_no_fix,
            new_grp_idx_start,
            H_q,
            N,
            psi
        ))

    expectation_v = 0
    for fragment in measured_groups:
        expectation_v += expectation(gso(fragment, N), psi)
    print(expectation_v)
    assert np.isclose(expectation_v.real, original_exp.real,
                      atol=1E-6), "Expectation value shouldn't change"

    print(f"Updated variance {variance}")
