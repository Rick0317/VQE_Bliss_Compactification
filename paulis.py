from openfermion import QubitOperator


class PauliOp:
    def __init__(self, op):
        self.index, self.pauli = op

    def commutes(self, other):
        if self.index != other.index:
            return True  # Different qubits always commute.
        # Commutation logic: same Pauli or identity always commutes.
        if self.pauli == other.pauli or self.pauli == 'I' or other.pauli == 'I':
            return True
        # Anti-commuting pairs: X/Y, Y/Z, Z/X.
        return False  # X vs Y, Y vs Z, Z vs X anti-commute.


class PauliString:
    def __init__(self, pauli_ops: tuple):
        self.pauli_ops = {index: pauli for index, pauli in pauli_ops}

    def qubit_wise_commute(self, other):
        for index in set(self.pauli_ops.keys()).intersection(other.pauli_ops.keys()):
            if not PauliOp((index, self.pauli_ops[index])).commutes(PauliOp((index, other.pauli_ops[index]))):
                return False
        return True  # All pairs commute if no anti-commuting pairs are found.

    def to_qubit_operator(self, coeff=1.0):
        qubit_terms = tuple(
            (index, pauli) for index, pauli in self.pauli_ops.items())
        return QubitOperator(qubit_terms, coeff)

    def __add__(self, other):
        if not isinstance(other, PauliString):
            raise TypeError("Can only add PauliString objects.")

        # Combine the two QubitOperators.
        result_operator = self.to_qubit_operator() + other.to_qubit_operator()
        return result_operator

    def __str__(self):
        return str(self.pauli_ops)


if __name__ == '__main__':
    p1 = PauliString(((0, 'X'), (4, 'Z')))
    p2 = PauliString(((0, 'X'), (1, 'Z'), (2, 'Z'), (3, 'Y')))
    print(p1 + p2)
