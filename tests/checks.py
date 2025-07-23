from openfermion import FermionOperator, is_hermitian, normal_ordered, bravyi_kitaev

if __name__ == "__main__":
    killer = (FermionOperator("2^ 1") + FermionOperator("1^ 2")) * FermionOperator("1^ 1")
    qubit_op = bravyi_kitaev(killer)
    print(normal_ordered(killer))
    print(qubit_op)
    print(is_hermitian(killer))
