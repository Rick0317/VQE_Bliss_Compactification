from openfermion import FermionOperator, normal_ordered
from random import random


if __name__ == '__main__':

    n_1 = FermionOperator("1^ 1")
    n_2 = FermionOperator("2^ 2")

    a = 0.3
    b = 0.7

    term1 = a * n_1 + b * n_2

    print(f"Term: {term1}")

    quadratic = term1 * term1

    print(f"Original: {quadratic}")
    print(f"Normal Ordered: {normal_ordered(quadratic)}")

    cubic = term1 * term1 * term1

    print(f"Original: {cubic}")
    print(f"Normal Ordered: {normal_ordered(cubic)}")




