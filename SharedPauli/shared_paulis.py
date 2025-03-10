"""
Implementation of the shared Pauli technique
"""
import numpy as np
from openfermion import (
    get_sparse_operator as gso,
    get_ground_state as ggs,
    variance,
    expectation,
    bravyi_kitaev,
    QubitOperator
)
from copy import deepcopy
from utils.frag_utils import does_term_frag_commute
from Decompositions.qwc_decomposition import qwc_decomposition
from paulis import PauliString


def update_decomp_w_shared_paulis(psi, N, original_decomp):
    """
    Update the commuting decomposition by introducing ghost paulis into some commuting sets.
    :param original_decomp: The original Pauli decomposition of the input Hamiltonian
    :return: The updated Pauli decomposition with ghost paulis in each set
    """
    new_decomp = original_decomp.copy()

    frag_combs = select_combs(psi, N, original_decomp)

    pauli_added_combs = select_sharable_paulis(frag_combs, original_decomp, N, psi)

    for combination in pauli_added_combs[:1]:
        (c, qubit_op, index_a, index_b) = combination
        share = 0.01
        new_decomp[index_a] -= share * qubit_op
        new_decomp[index_b] += share * qubit_op

    return new_decomp

def select_sharable_paulis(frag_combs, original_decomp, N, psi):
    """
    Given a set of fragments, we find the shareable Pauli operatoars between
    pairs of fragments so that we can transfer some coefficient to the smaller
    fragment.
    :param frag_combs: The pairs of fragments we consider
    :param original_decomp: The decomposition of the input Hamiltonian
    :param N: The number of sites in the Hamiltonian
    :param psi: The quantum state found by CISD/ For experiments, we use the exact groundstate.
    :return: [(c, pauli, frag_a, frag_b), ...], where c is the total coefficient
    """

    variance_sum = 0
    for fragement in original_decomp:
        variance_sum += np.sqrt(variance(gso(fragement, N), psi))

    pauli_added_combs = []
    for combination in frag_combs:
        (index_a, index_b) = combination
        frag_a = original_decomp[index_a]
        frag_b = original_decomp[index_b]
        size_a = len(frag_a.terms.items())
        size_b = len(frag_b.terms.items())

        if size_b > size_a:
            frag_a, frag_b = frag_b, frag_a

        sharable_pauli_a2b = []

        # From the two fragments, find the Pauli operator that could be shared.
        for term_a, coeff_a in frag_a.terms.items():
            for term_b, coeff_b in frag_b.terms.items():
                # If term_a commutes with frag_b or term_b commutes with frag_a
                # We add them to the sharable_pauli
                if does_term_frag_commute(QubitOperator(term=term_a), frag_b):
                    sharable_pauli_a2b.append((coeff_a, QubitOperator(term=term_a)))
                if does_term_frag_commute(QubitOperator(term=term_b), frag_a):
                    sharable_pauli_a2b.append((coeff_b, QubitOperator(term=term_b)))

        for sharable_pauli in sharable_pauli_a2b:

            pauli_added_combs.append((sharable_pauli[0], sharable_pauli[1], index_a, index_b))

    return pauli_added_combs


def select_combs(psi, N, original_decomp):
    """
    Given a decomposition of the input Hamiltonian, select the set of fragment combinations
    :param original_decomp:
    :return: A list of fragment combinations as indices in the decomposition: [(a, b)]
    """

    n_frag = len(original_decomp)
    vars = np.zeros(n_frag, dtype=np.complex128)
    for i, frag in enumerate(original_decomp):
        vars[i] = variance(gso(frag, N), psi)

    p = 4 * sum(np.sqrt(vars))

    score_board = {}
    for a in range(n_frag):
        for b in range(a+1, n_frag):
            var_a = vars[a]
            var_b = vars[b]

            # The score is the upper-bound of the variance metric reduction
            # Eq (15) of the ghost Pauli paper.
            score = p * (np.sqrt(var_a * var_b)) / (np.sqrt(var_a) + np.sqrt(var_b))
            score_board[(a, b)] = score

    sorted_items = sorted(score_board.items(), key=lambda item: item[1],
                          reverse=True)

    top_50_count = len(score_board) // 2 + (
        1 if len(score_board) % 2 != 0 else 0)

    top_keys = [key for key, value in sorted_items[:top_50_count]]
    print(top_keys)

    return top_keys


def variance_metric(H, decomp, N):
    psi = ggs(gso(H, N))[1]

    vars = np.zeros(len(decomp), dtype=np.complex128)
    for i, frag in enumerate(decomp):
        vars[i] = variance( gso(frag, N), psi )
    return np.sum((vars)**(1/2))**2


def abs_of_dict_value(x):
    return np.abs(x[1])


def copy_hamiltonian(H):
    H_copy = QubitOperator().zero()

    for t, s in H.terms.items():
        H_copy += s * QubitOperator(t)

    assert (H - H_copy) == QubitOperator().zero()
    return H_copy


def load_hamiltonian(moltag):
    filename = f'../../SolvableQubitHamiltonians/ham_lib/lih_fer.bin'
    with open(filename, 'rb') as f:
        Hfer = pickle.load(f)
    Hqub = bravyi_kitaev(Hfer)
    Hqub -= Hqub.constant
    Hqub.compress()
    Hqub.terms = dict(sorted(Hqub.terms.items(), key=abs_of_dict_value, reverse=True))
    return Hqub


def get_sharable_paulis(original_decomp):
    """
    Get the data of sharable Pauli words.
    :param frag_combs:
    :param original_decomp:
    :param N:
    :param psi:
    :return: Dictionary that maps sharable Pauli in the decomposition to
    its appearance in fragments.
    """
    all_pauli_dict = {}
    for frag_idx, fragment in enumerate(original_decomp):
        for idx, term in enumerate(fragment.terms):
            all_pauli_dict[term] = [frag_idx]
            pauli_word = QubitOperator(term=term)
            for frag_idx_compared, fragment_compared in enumerate(original_decomp):
                # Check if term can belong to the fragment
                # We have to check qubit-wise commutation
                if frag_idx_compared != frag_idx:
                    commute = True
                    for idx, compared_term in enumerate(fragment_compared.terms):
                        if not PauliString(compared_term).qubit_wise_commute(PauliString(term)):
                            commute = False
                    if commute:
                        all_pauli_dict[term].append(frag_idx_compared)

    sharable_pauli_dict = {}
    sharable_pauli_list = []
    sharable_pauli_indices_list = []
    for pauli_idx, pauli_term in enumerate(all_pauli_dict):
        shared_frag_idx_list = all_pauli_dict[pauli_term]
        if len(shared_frag_idx_list) > 1:
            sharable_pauli_list.append(pauli_term)
            sharable_pauli_indices_list.append(shared_frag_idx_list)
            sharable_pauli_dict[pauli_term] = shared_frag_idx_list

    return sharable_pauli_dict, sharable_pauli_list, sharable_pauli_indices_list


def get_share_pauli_only_decomp(sharable_pauli_dict):
    """
    Get the fragmentation of the Hamiltonian with only the sharable fragments included.
    We also
    :param sharable_pauli_dict:
    :return:
    """
    fragment_idx_to_sharable_paulis = {}
    pw_grp_idxes_no_fix = {}
    pw_grp_idxes_fix = {}
    for pauli_idx, pauli_term in enumerate(sharable_pauli_dict):
        pw_grp_idxes_no_fix[pauli_term] = sharable_pauli_dict[pauli_term][:-1]
        pw_grp_idxes_fix[pauli_term] = sharable_pauli_dict[pauli_term][-1]
        for frag_idx in sharable_pauli_dict[pauli_term][:-1]:
            if frag_idx not in fragment_idx_to_sharable_paulis:
                fragment_idx_to_sharable_paulis[frag_idx] = [pauli_term]
            else:
                fragment_idx_to_sharable_paulis[frag_idx].append(pauli_term)

    return fragment_idx_to_sharable_paulis, pw_grp_idxes_no_fix, pw_grp_idxes_fix


def get_overlapping_decomp(sharable_pauli_dict, original_decomp, pw_grp_idxes_fix):
    """
    Give the overlapping allowed decomposition
    :param sharable_pauli_dict:
    :return:
    """
    all_sharable_contained_decomp = deepcopy(original_decomp)
    all_sharable_no_fixed_decomp = deepcopy(original_decomp)

    for pauli_idx, pauli in enumerate(sharable_pauli_dict):
        pauli_appearance_indices_lst = sharable_pauli_dict[pauli]
        for appear_idx in pauli_appearance_indices_lst:
            if appear_idx not in pw_grp_idxes_fix:
                all_sharable_no_fixed_decomp[appear_idx] += QubitOperator(term=pauli)
            all_sharable_contained_decomp[appear_idx] += QubitOperator(term=pauli)

    return all_sharable_contained_decomp, all_sharable_no_fixed_decomp


def get_sharable_only_decomp(sharable_pauli_dict, original_decomp, pw_grp_idxes_fix, coeff_map):
    """
    Give the overlapping allowed decomposition
    :param sharable_pauli_dict:
    :return:
    """
    sharable_only_decomp = [QubitOperator().zero() for _ in range(len(original_decomp))]
    sharable_only_no_fixed_decomp = [QubitOperator().zero() for _ in range(len(original_decomp))]

    for pauli_idx, pauli in enumerate(sharable_pauli_dict):
        pauli_appearance_indices_lst = sharable_pauli_dict[pauli]
        for appear_idx in pauli_appearance_indices_lst:
            if appear_idx != pw_grp_idxes_fix[pauli]:
                sharable_only_no_fixed_decomp[appear_idx] += QubitOperator(term=pauli) * coeff_map[pauli]
            sharable_only_decomp[appear_idx] += QubitOperator(term=pauli) * coeff_map[pauli]

    return sharable_only_decomp, sharable_only_no_fixed_decomp


def get_all_pw_indices(sharable_pauli_list, sharable_no_fixed_decomp, pw_grp_idxes_no_fix, new_grp_idx_start):

    """
    Creates a dictionary mapping all Pauli words to their indices in the coefficient vector corresponding to the coefficients of that Pauli words in the respective group.
    Args:
        pw_list(List[QubitOperator]): List of all pauli words in multiple measurement groups.
        new_overlapped_group (List[List[QubitOperator]]): Overlapped Grouping with the fixed group Pauli removed (last group that the Pauli appears in).
        new_pw_grp_idxs (List[List[int]]): List of list of corresponding group indexes that each Pauli word appears in new_overlapped_group.
        new_grp_idx_start (List[int]): List of the starting index of each group (in the coefficient vector).
    Returns:
        pw_indices (Dict[QubitOperator Tuple, Dict]): Dictionary mapping all Pauli words to their indices in the coefficient vector corresponding to the coefficients of that Pauli words in the respective group.
    """

    pw_indices = {}
    for pw_idx, pw in enumerate(sharable_pauli_list):
        current_idxs = {}
        for grp_idx in pw_grp_idxes_no_fix[pw]:
            # The total number of appearances for previous pauli words + The appearance index of the Pauli word in the group of grp_idx
            idx = new_grp_idx_start[grp_idx] + qubit_op_to_list(sharable_no_fixed_decomp[grp_idx]).index(pw)
            current_idxs[grp_idx] = idx

        pw_indices[pw] = current_idxs

    return pw_indices


def get_pw_grp_idxes_no_fix_len(pw_grp_idxes_no_fix: dict):
    """
    Get the list of number of appearances except for the last index for each Pauli
    :param pw_grp_idxes_no_fix:
    :return:
    """
    pw_grp_idxes_no_fix_len = []
    for pw_idx, pw in enumerate(pw_grp_idxes_no_fix):
        pw_grp_idxes_no_fix_len.append(len(pw_grp_idxes_no_fix[pw]))
    return pw_grp_idxes_no_fix_len


def get_coefficient_orderings(sharable_only_decomp, sharable_pauli_list, sharable_pauli_indices_list, coeff_map):
    """
    Compute the fixed group for each Pauli word and get overlapped grouping with the fixed Pauli removed. Coefficients in coefficient splitting are based on new_overlapped_group (e.g. 0th component of coefficient vector corresponds to coefficient of 0th Pauli Word in 0th group).
    Args:
        sharable_only_decomp (List[List[QubitOperator]]): Overlapped Grouping.
        sharable_pauli_list(List[QubitOperator]): List of all pauli words in multiple measurement groups.
        sharable_pauli_indices_list (List[List[int]]): List of list of corresponding group indexes that each Pauli word appears in
    Returns:
        fixed_grp: List[int]: List of the fixed group index of each Pauli, corresponding to pw_list[idx].
        new_overlapped_group (List[List[QubitOperator]]): Overlapped Grouping with the fixed group Pauli removed (last group that the Pauli appears in).
        new_pw_grp_idxs (List[List[int]]): List of list of corresponding group indexes that each Pauli word appears in new_overlapped_group.
        new_grp_len_list (List[int]): List of length of each group in new_overlapped_group.
        new_grp_idx_start (List[int]): List of the starting index of each group (in the coefficient vector).
    """

    fixed_grp = []
    sharable_contained_no_fixed_decomp = deepcopy(sharable_only_decomp)
    new_sharable_pauli_indices_list = deepcopy(sharable_pauli_indices_list)
    for pw_idx, pw in enumerate(sharable_pauli_list):
        #fixed_grp.append(pw_grp_idxs[pw_idx][-1])
        if len(sharable_pauli_indices_list[pw_idx]) > 0:
           # Removing the fixed indices of each Pauli Word (Last index of appearance)
           fixed_grp.append(sharable_pauli_indices_list[pw_idx][len(sharable_pauli_indices_list[pw_idx]) -1])
           sharable_contained_no_fixed_decomp[fixed_grp[pw_idx]] -= QubitOperator(pw) * coeff_map[pw]
           new_sharable_pauli_indices_list[pw_idx].remove(fixed_grp[pw_idx])
        else:
           fixed_grp.append(None)

    new_grp_len_list = np.array([len(fragment.terms) for fragment in sharable_contained_no_fixed_decomp])
    new_grp_idx_start = [sum(new_grp_len_list[0:i:1]) for i in range(len(sharable_contained_no_fixed_decomp))]

    return fixed_grp, sharable_contained_no_fixed_decomp, new_sharable_pauli_indices_list, new_grp_len_list, new_grp_idx_start


def qubit_op_to_list(qubit_op: QubitOperator):
    term_list = []
    for idx, term in enumerate(qubit_op.terms):
        term_list.append(term)
    return term_list



def get_pauli_coeff_map(decomp):
    pauli_coeff_map = {}
    for fragment in decomp:
        for idx, term in enumerate(fragment.terms):
            if term not in pauli_coeff_map:
                pauli_coeff_map[term] = fragment.terms[term]

    return pauli_coeff_map



def distribute_coeff(optimized_coeff, original_coeff, all_sharable_contained_decomp, fragment_idx_to_sharable_paulis):
    for sharable_contained_decomp in all_sharable_contained_decomp:
        pass


if __name__ == '__main__':

    import pickle

    N = 12

    Hqub = load_hamiltonian("")

    sparse = gso(Hqub)
    psi = ggs(sparse)[1]

    H_q = copy_hamiltonian(Hqub)

    print(f"Correct Expectation value: {expectation(gso(H_q, N), psi)}")
    # decomp = qwc_decomposition(H_q)

    H_q2 = copy_hamiltonian(Hqub)
    decomp = qwc_decomposition(H_q2)
    # print(f"Decomposition: {decomp}")
    print(f"Number of terms {len(decomp)}")

    coeff_map = get_pauli_coeff_map(decomp)


    # Get the sharable Pauli information.
    # 1. Dictionary that maps each sharable Pauli to the corresponding fragment indices
    # 2. The list of sharable Paulis
    # 3. The list of list of indices of appearance of each sharable Pauli
    sharable_paulis_dict, sharable_paulis_list, sharable_pauli_indices_list = get_sharable_paulis(decomp)
    fixed_list = [indices[-1] for indices in sharable_pauli_indices_list]
    assert len(sharable_paulis_dict) == len(sharable_paulis_list), "Sharable creation failed"
    assert len(sharable_paulis_list) == len(sharable_pauli_indices_list), "Sharable creation failed"

    # Get the Hamiltonian decomposition with only the sharable Paulis
    # 1. Mapping of each fragment to sharable Pauli list
    # 2. The fragment indices the Pauli appears without the last appearance
    # 3. The fragment index with the last appearance
    fragment_idx_to_sharable_paulis, pw_grp_idxes_no_fix, pw_grp_idxes_fix = get_share_pauli_only_decomp(sharable_paulis_dict)
    total_appearances = 0
    for pw_idx, pw in enumerate(pw_grp_idxes_no_fix):
        total_appearances += len(pw_grp_idxes_no_fix[pw])

    num_fixes = 0
    for pw_idx, pw in enumerate(pw_grp_idxes_fix):
        total_appearances += 1
        num_fixes += 1

    assert len(sharable_paulis_list) == num_fixes, "Number of sharable Pauli should match len(fixes)"

    no_fix_appearance = total_appearances - num_fixes

    print(f"Total number of terms: {total_appearances}")
    print(f"Total number no fixes: {no_fix_appearance}")

    # For all sharable Paulis, we let it appear in all the terms that
    # they can belong to.
    # 1. The modified decomposition with sharable Pauli appears in each term
    # it can appear
    # 2. The modified decomposition with sharable Pauli appears in each term
    # it can appear except for the last term of appearance
    all_sharable_contained_decomp, all_sharable_no_fixed_decomp = get_overlapping_decomp(sharable_paulis_dict, decomp, pw_grp_idxes_fix)

    assert len(all_sharable_contained_decomp) == len(decomp), "Number of terms should match"
    assert len(all_sharable_no_fixed_decomp) == len(decomp), "Number of terms should match"


    # Get the decomposition of the Hamiltonian with only sharable Paulis
    # 1. all_sharable_contained_decomp - non_sharable Paulis
    # 2. all_sharable_no_fixed_decomp - non_sharable Paulis
    sharable_only_decomp, sharable_only_no_fixed_decomp = get_sharable_only_decomp(sharable_paulis_dict, decomp, pw_grp_idxes_fix, coeff_map)
    assert len(sharable_only_decomp) == len(decomp), "Number of terms should match"
    assert len(sharable_only_no_fixed_decomp) == len(decomp), "Number of terms should match"

    num_terms = 0
    for fragment in sharable_only_decomp:
        for idx, term in enumerate(fragment.terms.items()):
            num_terms += 1

    num_no_fixed_terms = 0
    for fragment in sharable_only_no_fixed_decomp:
        for idx, term in enumerate(fragment.terms.items()):
            num_no_fixed_terms += 1

    assert num_terms == total_appearances, "Number of terms should match"
    assert num_no_fixed_terms == no_fix_appearance, "Number of terms should match"

    # Get the coefficient ordering
    # 1. The list of fixed indices for each Pauli word. The order is based on the sharable_pauli_list
    # 2. Sharable Pauli included decomposition without the fixed indices.
    # 3. The list of list of indices of appearance of each sharable Pauli without the last index
    # 4. The list of numbers of sharable Paulis in each fragment without the fixed ones
    # 5. The indices of the start of each fragment
    fixed_grp, all_sharable_contained_no_fixed_decomp, new_sharable_pauli_indices_list, new_grp_len_list, new_grp_idx_start \
        = get_coefficient_orderings(sharable_only_decomp, sharable_paulis_list, sharable_pauli_indices_list, coeff_map)

    assert len(fixed_grp) == len(sharable_paulis_list), "Number of fixed should match the number of Paulis"
    assert len(all_sharable_contained_no_fixed_decomp) == len(decomp), "Number of fragments should match"

    num_no_fixed_terms2 = 0
    for fragment in all_sharable_contained_no_fixed_decomp:
        for idx, term in enumerate(fragment.terms.items()):
            num_no_fixed_terms2 += 1


    assert num_no_fixed_terms2 == num_no_fixed_terms, "Number of terms should match"
    assert num_no_fixed_terms == sum([len(indices) for indices in new_sharable_pauli_indices_list]), "Number of terms no fixed should match"

    assert sum(new_grp_len_list) == num_no_fixed_terms, "Number of terms without fixes should match"



    pw_grp_idxes_no_fix_len = get_pw_grp_idxes_no_fix_len(pw_grp_idxes_no_fix)

    assert sum(pw_grp_idxes_no_fix_len) == num_no_fixed_terms, "Number of terms without fixes should match"

    # Get the indices in the coefficient space for each Pauli word
    # For each fragment index, we assign the number that shows the order of appearance of the sharable Pauli + the sum of previous indices
    pw_indices = get_all_pw_indices(sharable_paulis_list, sharable_only_no_fixed_decomp, pw_grp_idxes_no_fix, new_grp_idx_start)
    print(new_grp_len_list, new_grp_idx_start)
    print(len(qubit_op_to_list(sharable_only_no_fixed_decomp[153])))
    exit()
    for fragment in all_sharable_contained_decomp:
        print(fragment)
    exit()
    print(type(decomp[0]))
    exectation_original = 0
    for fragment in decomp:
        exectation_original += expectation(gso(fragment, N), psi)
    print(f"Expectation value original: {exectation_original}")

    H_q3 = copy_hamiltonian(Hqub)
    original_variance = variance_metric(H_q3, decomp, N)
    print(f"Original variance: {original_variance}")

    new_decomp = update_decomp_w_shared_paulis(psi, N, decomp)

    # The new expectation value should match the original value
    expectation_new = 0
    for fragment in new_decomp:
        expectation_new += expectation(gso(fragment, N), psi)
    print(f"Expectation value new: {expectation_new}")

    H_q4 = copy_hamiltonian(Hqub)
    # print(f"New decomposition: {new_decomp}")
    new_variance = variance_metric(H_q4, new_decomp, N)
    print(f"New variance: {new_variance}")

    print("Original Decomposition Complete")
