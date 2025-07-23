import numpy as np
from entities.paulis import PauliString
from openfermion import QubitOperator, FermionOperator, bravyi_kitaev


def fc_decomposition(Hqub):
    groups = []
    for term in Hqub.terms:
        pauli_str = PauliString(term)
        coeff = Hqub.terms[term]
        found_group = False
        for idx, group in enumerate(groups):
            commute = True
            for pauli_term, _ in group:
                group_pauli = PauliString(pauli_term)
                if not pauli_str.fully_commute(group_pauli):
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


if __name__ == '__main__':
    qubit_op1 = QubitOperator('Z0 Z1 Z2')
    qubit_op2 = QubitOperator('Z0 X1 Z2')

    commutators = qubit_op1 * qubit_op2 - qubit_op2 * qubit_op1

    print(commutators)
