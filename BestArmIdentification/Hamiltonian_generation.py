from pyscf import gto, scf
from openfermionpyscf import run_pyscf
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermion.linalg import get_sparse_operator
from openfermionpyscf import generate_molecular_hamiltonian
import pickle
import zipfile
import os

def generat_with_of():
    from openfermion import QubitOperator, FermionOperator, jordan_wigner, \
        commutator, MolecularData, get_fermion_operator, count_qubits

    n_hydrogens = 3
    bond_length = 2
    basis = 'sto-3g'
    multiplicity = 2
    spin_ord = 'udud'

    molecule = MolecularData(
        geometry=[('H', (0, 0, i * bond_length)) for i in range(n_hydrogens)],
        charge=0,
        basis=basis,
        multiplicity=multiplicity,
        description=f"linear_r-{bond_length}")

    molecule = run_pyscf(molecule, run_scf=True, run_cisd=True, run_fci=True)
    print("PyScf Calculation complete. Hartree-Fock energy:",
          molecule.hf_energy,
          "\nFCI energy:", molecule.fci_energy)

    mh = molecule.get_molecular_hamiltonian()
    H = get_fermion_operator(mh)

    return H

if __name__ == "__main__":
    mol = "h3st"
    # Create folder (optional, but useful)
    output_folder = 'output_data'
    os.makedirs(output_folder, exist_ok=True)

    # File paths
    bin_path = os.path.join(output_folder, f'{mol}_fer.bin')
    zip_path = os.path.join(output_folder, f'{mol}_fer.zip')

    # # Define geometry: H4 in a square or linear layout
    # geometry = [
    #     ('H', (0.0, 0.0, 0.0)),
    #     ('H', (0.0, 0.0, 2.0)),
    #     ('H', (0.0, 0.0, 4.0)),
    #     ('H', (0.0, 0.0, 6.0))
    # ]
    #
    # basis = 'sto-3g'
    # multiplicity = 1  # Singlet state (2S + 1)
    # charge = 0
    #
    # # Create and build molecule
    # molecule = gto.Mole()
    # molecule.atom = geometry
    # molecule.basis = basis
    # molecule.charge = charge
    # molecule.spin = multiplicity - 1  # spin = 2S
    # molecule.build()
    #
    # # Run Hartree-Fock
    # mean_field = scf.RHF(molecule)
    # mean_field.kernel()
    #
    # # Generate Hamiltonian with required multiplicity
    # molecular_hamiltonian = generate_molecular_hamiltonian(
    #     geometry=geometry,
    #     basis=basis,
    #     multiplicity=multiplicity
    # )
    #
    # # Convert to FermionOperator
    # fermionic_hamiltonian = get_fermion_operator(molecular_hamiltonian)
    fermionic_hamiltonian = generat_with_of()

    # Optionally convert to qubit operator
    qubit_hamiltonian = jordan_wigner(fermionic_hamiltonian)

    # Print the result
    print("Fermionic Hamiltonian:")
    print(fermionic_hamiltonian)

    with open(bin_path, 'wb') as f:
        pickle.dump(fermionic_hamiltonian, f)

    # Compress it into a zip archive
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(bin_path, arcname=f'{mol}_fer.bin')

    # Optional: remove the original .bin file after zipping
    os.remove(bin_path)

    print(f'FermionOperator saved to {zip_path}')

    print("\nQubit Hamiltonian (Jordan-Wigner):")
    print(qubit_hamiltonian)
