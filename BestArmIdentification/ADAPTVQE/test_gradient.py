import os
import pickle


if __name__ == "__main__":

    with open(os.path.join(os.path.dirname(__file__), '../../ham_lib/h4_sto-3g.pkl'), 'rb') as f:
        H_ferm = pickle.load(f)
    n_electrons = 4


