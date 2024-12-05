import numpy as np
from paulis import PauliString
from openfermion import QubitOperator


def qwc_decomposition(Hqub):
    sorted_terms = sorted(Hqub.terms, key=lambda x: np.abs(Hqub.terms[x]), reverse=True)
    groups = []

    for term in sorted_terms:
        pauli_str = PauliString(term)  # Create PauliString from the term only.
        coeff = Hqub.terms[term]
        found_group = False

        for idx, group in enumerate(groups):
            commute = True
            for pauli_term, _ in group:  # Unpack (term, coeff) tuple, only need term here.
                group_pauli = PauliString(pauli_term)
                if not pauli_str.qubit_wise_commute(group_pauli):
                    commute = False
                    break

            if commute:
                groups[idx].append((term, coeff))
                found_group = True
                break

        if not found_group:
            groups.append([(term, coeff)])

    decomp = []
    for group in groups:
        qubit_op = QubitOperator()
        for pauli_term, coeff in group:
            qubit_op += PauliString(pauli_term).to_qubit_operator(coeff)

        decomp.append(qubit_op)

    return decomp
