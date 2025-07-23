from openfermion import FermionOperator, bravyi_kitaev, get_sparse_operator, expectation
import pickle
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StatePreparation.reference_state_utils import get_cisd_gs, get_occ_no, get_reference_state
from utils.ferm_utils import ferm_to_qubit


def create_ucc_generator(n_qubits, n_electrons):
    """
    Create UCC generator G = T - T† where T contains single and double excitations
    """
    # Get occupied and virtual orbital indices
    occupied = list(range(n_electrons))
    virtual = list(range(n_electrons, n_qubits))

    ucc_generator = FermionOperator()

    # Single excitations: T1 = sum_ia t_ia * a†_i a_a
    print("Adding single excitations...")
    for i in occupied:
        for a in virtual:
            # T1 term: a†_a a_i (create virtual, annihilate occupied)
            single_excitation = FermionOperator(f'{a}^ {i}', 1.0)
            # T1† term: a†_i a_a (create occupied, annihilate virtual)
            single_excitation_dagger = FermionOperator(f'{i}^ {a}', 1.0)

            # Add T - T† for this excitation
            ucc_generator += single_excitation - single_excitation_dagger

    # Double excitations: T2 = sum_ijab t_ijab * a†_i a†_j a_b a_a
    print("Adding double excitations...")
    for i in occupied:
        for j in occupied:
            if i < j:  # Avoid double counting
                for a in virtual:
                    for b in virtual:
                        if a < b:  # Avoid double counting
                            # T2 term: a†_a a†_b a_j a_i
                            double_excitation = FermionOperator(f'{a}^ {b}^ {j} {i}', 1.0)
                            # T2† term: a†_i a†_j a_b a_a
                            double_excitation_dagger = FermionOperator(f'{i}^ {j}^ {b} {a}', 1.0)

                            # Add T - T† for this excitation
                            ucc_generator += double_excitation - double_excitation_dagger


    return ucc_generator


def create_specific_ucc_generator(a, b, j, i):
    double_excitation = FermionOperator(f'{a}^ {b}^ {j} {i}', 1.0)
    # T2† term: a†_i a†_j a_b a_a
    double_excitation_dagger = FermionOperator(f'{i}^ {j}^ {b} {a}', 1.0)
    return double_excitation - double_excitation_dagger

def compute_gradient_expectation(H_sparse, pauli_sparse, wavefunction):
    """
    Compute <ψ|[H, P]|ψ> = <ψ|HP - PH|ψ>
    """
    # Compute HP|ψ> and PH|ψ>
    HP_psi = H_sparse.dot(pauli_sparse.dot(wavefunction))
    PH_psi = pauli_sparse.dot(H_sparse.dot(wavefunction))

    # Compute <ψ|[H,P]|ψ> = <ψ|HP|ψ> - <ψ|PH|ψ>
    gradient = np.vdot(wavefunction, HP_psi) - np.vdot(wavefunction, PH_psi)

    return gradient


def create_perturbed_state_ucc(hf_wavefunction, ucc_generator_sparse, theta=0.01):
    """
    Create a perturbed state using UCC rotation: |ψ⟩ = e^(θG)|ψ_CISD⟩
    """
    from scipy.sparse.linalg import expm_multiply

    print(f"Applying UCC rotation with θ = {theta} to HF")
    # Apply e^(θG) to the CISD state
    perturbed_state = expm_multiply(theta * ucc_generator_sparse, hf_wavefunction)

    # Normalize
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    return perturbed_state


def create_perturbed_state_noise(cisd_wavefunction, noise_level=0.05):
    """
    Create a perturbed state by adding controlled noise to CISD coefficients
    """
    print(f"Adding random noise with level = {noise_level}")

    # Add random noise to each coefficient
    noise = np.random.normal(0, noise_level, cisd_wavefunction.shape)
    perturbed_state = cisd_wavefunction + noise

    # Normalize
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    return perturbed_state


def create_perturbed_state_mixing(cisd_wavefunction, hf_wavefunction, mixing_coeff=0.1):
    """
    Create a perturbed state by mixing CISD with Hartree-Fock state
    """
    print(f"Mixing CISD with HF state, mixing coefficient = {mixing_coeff}")

    # Mix the two states
    perturbed_state = (1 - mixing_coeff) * cisd_wavefunction + mixing_coeff * hf_wavefunction

    # Normalize
    perturbed_state = perturbed_state / np.linalg.norm(perturbed_state)

    return perturbed_state


def analyze_state_difference(state1, state2, label1="State 1", label2="State 2"):
    """
    Analyze the difference between two quantum states
    """
    # Fidelity
    fidelity = abs(np.vdot(state1, state2))**2

    # Norm difference
    norm_diff = np.linalg.norm(state1 - state2)

    print(f"\nState comparison ({label1} vs {label2}):")
    print(f"  Fidelity |⟨ψ₁|ψ₂⟩|²: {fidelity:.6f}")
    print(f"  Norm difference: {norm_diff:.6f}")
    print(f"  States are {'very similar' if fidelity > 0.99 else 'different'}")

    return fidelity, norm_diff


def create_hf_state(ref_occ, n_qubits):
    """
    Create Hartree-Fock reference state directly from occupation string
    """
    # Convert occupation string to binary index
    binary_index = 0
    for i, occ in enumerate(ref_occ):
        if occ == '1':
            binary_index += 2**(n_qubits - 1 - i)

    # Create wavefunction vector
    hf_state = np.zeros(2**n_qubits)
    hf_state[binary_index] = 1.0

    return hf_state


def analyze_ucc_combinations_separate_generators(H_sparse, n_qubits, n_electrons, hf_wavefunction, cisd_wavefunction, theta=0.2, gradient_threshold=1e-10):
    """
    Systematically analyze different UCC generator combinations where:
    - One generator is used for rotation (creating perturbed state)
    - Different generators are used for gradient calculation
    - Only store results with non-zero gradients
    """
    # Define occupied and virtual orbital indices
    occupied = list(range(n_electrons))  # [0, 1, 2, 3] for LiH
    virtual = list(range(n_electrons, n_qubits))  # [4, 5, 6, 7, 8, 9, ...] for LiH

    # Generate all valid combinations
    valid_combinations = []
    for i in occupied:
        for j in occupied:
            if i < j:  # Avoid double counting for occupied pairs
                for a in virtual:
                    for b in virtual:
                        if a < b:  # Avoid double counting for virtual pairs
                            valid_combinations.append((i, j, a, b))

    print(f"Analyzing UCC combinations with separate generators (storing only non-zero gradients):")
    print(f"Occupied orbitals: {occupied}")
    print(f"Virtual orbitals: {virtual}")
    print(f"Valid combinations found: {len(valid_combinations)}")
    print(f"Gradient threshold: {gradient_threshold}")
    print(f"Total rotation × gradient tests: {len(valid_combinations)} × {len(valid_combinations)} = {len(valid_combinations)**2}")
    print("-" * 80)

    results = []
    test_count = 0
    total_gradients_computed = 0
    non_zero_gradients_found = 0

    # Test all pairs of combinations
    for rot_idx, (i_rot, j_rot, a_rot, b_rot) in enumerate(valid_combinations):
        print(f"\nROTATION Generator {rot_idx+1}/{len(valid_combinations)}: i={i_rot}, j={j_rot}, a={a_rot}, b={b_rot}")

        try:
            # Create rotation generator and perturbed state
            rotation_gen_ferm = create_specific_ucc_generator(a_rot, b_rot, j_rot, i_rot)
            rotation_gen_qubit = ferm_to_qubit(rotation_gen_ferm)
            rotation_gen_sparse = get_sparse_operator(rotation_gen_qubit, n_qubits)

            # Create perturbed state using rotation generator
            perturbed_state = cisd_wavefunction # create_perturbed_state_ucc(hf_wavefunction, rotation_gen_sparse, theta)
            print(f"Perturbed state is {perturbed_state}")
            print(f"  Rotation generator terms: {len(rotation_gen_qubit.terms)}")
            print(f"  Testing against {len(valid_combinations)} gradient generators...")

            rotation_non_zero_count = 0

            # Test against all gradient generators
            for grad_idx, (i_grad, j_grad, a_grad, b_grad) in enumerate(valid_combinations):
                test_count += 1

                try:
                    # Create gradient generator
                    gradient_gen_ferm = create_specific_ucc_generator(a_grad, b_grad, j_grad, i_grad)
                    gradient_gen_qubit = ferm_to_qubit(gradient_gen_ferm)

                    # Store gradient values for each Pauli operator in gradient generator
                    gradients_for_test = {}
                    test_has_nonzero = False

                    for pauli_idx, (pauli_term, coeff) in enumerate(gradient_gen_qubit.terms.items()):
                        from openfermion import QubitOperator
                        pauli_op = QubitOperator(pauli_term, 1.0)
                        pauli_sparse = get_sparse_operator(pauli_op, n_qubits)

                        # Compute gradient for this perturbed state
                        gradient = compute_gradient_expectation(H_sparse, pauli_sparse, perturbed_state)
                        total_gradients_computed += 1

                        # Only store if gradient is non-zero
                        if abs(gradient) > gradient_threshold:
                            gradients_for_test[f'P{pauli_idx+1}'] = {
                                'pauli_term': pauli_term,
                                'gradient': gradient,
                                'coeff': coeff
                            }
                            test_has_nonzero = True
                            non_zero_gradients_found += 1

                    # Only store result if there are non-zero gradients
                    if test_has_nonzero:
                        result = {
                            'test_number': test_count,
                            'rotation_combination': f'i={i_rot}, j={j_rot}, a={a_rot}, b={b_rot}',
                            'gradient_combination': f'i={i_grad}, j={j_grad}, a={a_grad}, b={b_grad}',
                            'rotation_indices': {'i': i_rot, 'j': j_rot, 'a': a_rot, 'b': b_rot},
                            'gradient_indices': {'i': i_grad, 'j': j_grad, 'a': a_grad, 'b': b_grad},
                            'n_gradient_pauli_terms': len(gradient_gen_qubit.terms),
                            'n_nonzero_gradients': len(gradients_for_test),
                            'gradients': gradients_for_test
                        }

                        results.append(result)
                        rotation_non_zero_count += 1

                        # Print summary for this test (show all non-zero results)
                        print(f"    ✓ GRADIENT Generator {grad_idx+1}: i={i_grad}, j={j_grad}, a={a_grad}, b={b_grad}")
                        print(f"      Non-zero gradients: {len(gradients_for_test)}/{len(gradient_gen_qubit.terms)}")
                        for pauli_name, pauli_data in gradients_for_test.items():
                            print(f"      UCC_Perturbed: <ψ|[H,{pauli_name}]|ψ> = {pauli_data['gradient']}")

                except Exception as e:
                    print(f"    Error with gradient generator {grad_idx+1}: {e}")
                    continue

            print(f"  → Found {rotation_non_zero_count} gradient generators with non-zero results")

        except Exception as e:
            print(f"  Error with rotation generator: {e}")
            continue

    print(f"\n" + "="*60)
    print(f"FILTERING SUMMARY:")
    print(f"Total gradients computed: {total_gradients_computed}")
    print(f"Non-zero gradients found: {non_zero_gradients_found}")
    print(f"Percentage non-zero: {(non_zero_gradients_found/total_gradients_computed*100):.2f}%")
    print(f"Tests with non-zero gradients: {len(results)}/{test_count}")
    print("="*60)

    return results


def save_and_analyze_nonzero_results(results, filename="ucc_nonzero_gradient_results.txt"):
    """
    Save and analyze results containing only non-zero gradients
    """
    with open(filename, 'w') as f:
        f.write("UCC Non-Zero Gradient Analysis Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total tests with non-zero gradients: {len(results)}\n\n")

        for result in results:
            f.write(f"Test {result['test_number']}:\n")
            f.write(f"Rotation Generator: {result['rotation_combination']}\n")
            f.write(f"Gradient Generator: {result['gradient_combination']}\n")
            f.write(f"Non-zero gradients: {result['n_nonzero_gradients']}/{result['n_gradient_pauli_terms']}\n")
            f.write("-" * 60 + "\n")

            for pauli_name, pauli_data in result['gradients'].items():
                f.write(f"{pauli_name}: {pauli_data['pauli_term']}\n")
                f.write(f"  Gradient: {pauli_data['gradient']}\n")
                f.write(f"  Coefficient: {pauli_data['coeff']}\n")
            f.write("\n")

    # Enhanced analysis for non-zero results
    print(f"\nNON-ZERO GRADIENT ANALYSIS:")
    print(f"Total meaningful tests: {len(results)}")

    if len(results) == 0:
        print("No non-zero gradients found!")
        return

    # Find test with largest gradient magnitude
    max_gradient_mag = 0
    max_test = None
    max_pauli = None

    for result in results:
        for pauli_name, pauli_data in result['gradients'].items():
            grad_mag = abs(pauli_data['gradient'])
            if grad_mag > max_gradient_mag:
                max_gradient_mag = grad_mag
                max_test = result
                max_pauli = pauli_name

    print(f"Largest gradient magnitude: {max_gradient_mag:.6f}")
    print(f"  Rotation generator: {max_test['rotation_combination']}")
    print(f"  Gradient generator: {max_test['gradient_combination']}")
    print(f"  Pauli operator: {max_pauli}")

    # Gradient magnitude distribution
    all_gradients = []
    for result in results:
        for pauli_data in result['gradients'].values():
            all_gradients.append(abs(pauli_data['gradient']))

    all_gradients = np.array(all_gradients)
    print(f"\nGradient magnitude statistics:")
    print(f"  Mean: {np.mean(all_gradients):.6f}")
    print(f"  Std:  {np.std(all_gradients):.6f}")
    print(f"  Min:  {np.min(all_gradients):.6f}")
    print(f"  Max:  {np.max(all_gradients):.6f}")

    # Most productive generator combinations
    rotation_counts = {}
    gradient_counts = {}

    for result in results:
        rot_combo = result['rotation_combination']
        grad_combo = result['gradient_combination']

        if rot_combo not in rotation_counts:
            rotation_counts[rot_combo] = 0
        rotation_counts[rot_combo] += result['n_nonzero_gradients']

        if grad_combo not in gradient_counts:
            gradient_counts[grad_combo] = 0
        gradient_counts[grad_combo] += result['n_nonzero_gradients']

    print(f"\nMost productive rotation generators:")
    sorted_rot = sorted(rotation_counts.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sorted_rot[:3]:
        print(f"  {combo}: {count} non-zero gradients")

    print(f"\nMost responsive gradient generators:")
    sorted_grad = sorted(gradient_counts.items(), key=lambda x: x[1], reverse=True)
    for combo, count in sorted_grad[:3]:
        print(f"  {combo}: {count} non-zero gradients")

    print(f"\nResults saved to: {filename}")


def create_perturbed_state_double_ucc(hf_wavefunction, ucc_generator1_sparse, ucc_generator2_sparse, theta1=0.01, theta2=0.01):
    """
    Create a perturbed state using two sequential UCC rotations: |ψ⟩ = e^(θ₂G₂)e^(θ₁G₁)|ψ_HF⟩
    """
    from scipy.sparse.linalg import expm_multiply

    print(f"Applying double UCC rotation with θ₁ = {theta1}, θ₂ = {theta2} to HF")

    # First rotation: e^(θ₁G₁)|ψ_HF⟩
    intermediate_state = expm_multiply(theta1 * ucc_generator1_sparse, hf_wavefunction)
    intermediate_state = intermediate_state / np.linalg.norm(intermediate_state)

    # Second rotation: e^(θ₂G₂)|ψ_intermediate⟩
    final_state = expm_multiply(theta2 * ucc_generator2_sparse, intermediate_state)
    final_state = final_state / np.linalg.norm(final_state)

    return final_state


def analyze_double_ucc_combinations(H_sparse, n_qubits, n_electrons, hf_wavefunction, theta1=0.1, theta2=0.1, gradient_threshold=1e-10):
    """
    Systematically analyze UCC combinations with two sequential rotations:
    - First generator for first rotation
    - Second generator for second rotation
    - Third generator for gradient calculation
    """
    # Define occupied and virtual orbital indices
    occupied = list(range(n_electrons))  # [0, 1, 2, 3] for LiH
    virtual = list(range(n_electrons, n_qubits))  # [4, 5, 6, 7, 8, 9, ...] for LiH

    # Generate all valid combinations
    valid_combinations = []
    for i in occupied:
        for j in occupied:
            if i < j:  # Avoid double counting for occupied pairs
                for a in virtual:
                    for b in virtual:
                        if a < b:  # Avoid double counting for virtual pairs
                            valid_combinations.append((i, j, a, b))

    print(f"Analyzing DOUBLE UCC combinations (storing only non-zero gradients):")
    print(f"Occupied orbitals: {occupied}")
    print(f"Virtual orbitals: {virtual}")
    print(f"Valid combinations found: {len(valid_combinations)}")
    print(f"Rotation angles: θ₁ = {theta1}, θ₂ = {theta2}")
    print(f"Gradient threshold: {gradient_threshold}")
    print(f"Total tests: {len(valid_combinations)}³ = {len(valid_combinations)**3} (rot1 × rot2 × grad)")
    print("-" * 80)

    results = []
    test_count = 0
    total_gradients_computed = 0
    non_zero_gradients_found = 0

    # Test all triplets of combinations
    for rot1_idx, (i_rot1, j_rot1, a_rot1, b_rot1) in enumerate(valid_combinations):
        print(f"\nROTATION 1 Generator {rot1_idx+1}/{len(valid_combinations)}: i={i_rot1}, j={j_rot1}, a={a_rot1}, b={b_rot1}")

        try:
            # Create first rotation generator
            rotation1_gen_ferm = create_specific_ucc_generator(a_rot1, b_rot1, j_rot1, i_rot1)
            rotation1_gen_qubit = ferm_to_qubit(rotation1_gen_ferm)
            rotation1_gen_sparse = get_sparse_operator(rotation1_gen_qubit, n_qubits)

            rot1_non_zero_count = 0

            for rot2_idx, (i_rot2, j_rot2, a_rot2, b_rot2) in enumerate(valid_combinations):
                if rot2_idx < 3 or rot2_idx == len(valid_combinations) - 1:  # Limit output
                    print(f"  ROTATION 2 Generator {rot2_idx+1}: i={i_rot2}, j={j_rot2}, a={a_rot2}, b={b_rot2}")
                elif rot2_idx == 3:
                    print(f"  ... (testing rotation 2 generators 4-{len(valid_combinations)-1}) ...")

                try:
                    # Create second rotation generator
                    rotation2_gen_ferm = create_specific_ucc_generator(a_rot2, b_rot2, j_rot2, i_rot2)
                    rotation2_gen_qubit = ferm_to_qubit(rotation2_gen_ferm)
                    rotation2_gen_sparse = get_sparse_operator(rotation2_gen_qubit, n_qubits)

                    # Create double-rotated state
                    double_rotated_state = create_perturbed_state_double_ucc(
                        hf_wavefunction, rotation1_gen_sparse, rotation2_gen_sparse, theta1, theta2
                    )

                    rot2_non_zero_count = 0

                    # Test against all gradient generators
                    for grad_idx, (i_grad, j_grad, a_grad, b_grad) in enumerate(valid_combinations):
                        test_count += 1

                        try:
                            # Create gradient generator
                            gradient_gen_ferm = create_specific_ucc_generator(a_grad, b_grad, j_grad, i_grad)
                            gradient_gen_qubit = ferm_to_qubit(gradient_gen_ferm)

                            # Store gradient values for each Pauli operator in gradient generator
                            gradients_for_test = {}
                            test_has_nonzero = False

                            for pauli_idx, (pauli_term, coeff) in enumerate(gradient_gen_qubit.terms.items()):
                                from openfermion import QubitOperator
                                pauli_op = QubitOperator(pauli_term, 1.0)
                                pauli_sparse = get_sparse_operator(pauli_op, n_qubits)

                                # Compute gradient for this double-rotated state
                                gradient = compute_gradient_expectation(H_sparse, pauli_sparse, double_rotated_state)
                                total_gradients_computed += 1

                                # Only store if gradient is non-zero
                                if abs(gradient) > gradient_threshold:
                                    gradients_for_test[f'P{pauli_idx+1}'] = {
                                        'pauli_term': pauli_term,
                                        'gradient': gradient,
                                        'coeff': coeff
                                    }
                                    test_has_nonzero = True
                                    non_zero_gradients_found += 1

                            # Only store result if there are non-zero gradients
                            if test_has_nonzero:
                                result = {
                                    'test_number': test_count,
                                    'rotation1_combination': f'i={i_rot1}, j={j_rot1}, a={a_rot1}, b={b_rot1}',
                                    'rotation2_combination': f'i={i_rot2}, j={j_rot2}, a={a_rot2}, b={b_rot2}',
                                    'gradient_combination': f'i={i_grad}, j={j_grad}, a={a_grad}, b={b_grad}',
                                    'rotation1_indices': {'i': i_rot1, 'j': j_rot1, 'a': a_rot1, 'b': b_rot1},
                                    'rotation2_indices': {'i': i_rot2, 'j': j_rot2, 'a': a_rot2, 'b': b_rot2},
                                    'gradient_indices': {'i': i_grad, 'j': j_grad, 'a': a_grad, 'b': b_grad},
                                    'n_gradient_pauli_terms': len(gradient_gen_qubit.terms),
                                    'n_nonzero_gradients': len(gradients_for_test),
                                    'theta1': theta1,
                                    'theta2': theta2,
                                    'gradients': gradients_for_test
                                }

                                results.append(result)
                                rot1_non_zero_count += 1
                                rot2_non_zero_count += 1

                                # Print summary for significant results only
                                if rot2_idx < 2 and grad_idx < 2:  # Only show first few to avoid spam
                                    print(f"      ✓ GRADIENT Generator {grad_idx+1}: i={i_grad}, j={j_grad}, a={a_grad}, b={b_grad}")
                                    print(f"        Non-zero gradients: {len(gradients_for_test)}/{len(gradient_gen_qubit.terms)}")
                                    for pauli_name, pauli_data in gradients_for_test.items():
                                        print(f"        Double_UCC_Perturbed: <ψ|[H,{pauli_name}]|ψ> = {pauli_data['gradient']}")

                        except Exception as e:
                            if rot2_idx < 2:  # Only show errors for first few
                                print(f"      Error with gradient generator {grad_idx+1}: {e}")
                            continue

                    if rot2_idx < 3 or rot2_idx == len(valid_combinations) - 1:
                        print(f"    → Found {rot2_non_zero_count} gradient generators with non-zero results")

                except Exception as e:
                    if rot2_idx < 3:
                        print(f"    Error with rotation 2 generator: {e}")
                    continue

            print(f"  → Total non-zero results for this rotation 1: {rot1_non_zero_count}")

        except Exception as e:
            print(f"  Error with rotation 1 generator: {e}")
            continue

    print(f"\n" + "="*60)
    print(f"DOUBLE ROTATION FILTERING SUMMARY:")
    print(f"Total gradients computed: {total_gradients_computed}")
    print(f"Non-zero gradients found: {non_zero_gradients_found}")
    print(f"Percentage non-zero: {(non_zero_gradients_found/total_gradients_computed*100):.2f}%")
    print(f"Tests with non-zero gradients: {len(results)}/{test_count}")
    print("="*60)

    return results


def save_and_analyze_double_results(results, filename="ucc_double_rotation_results.txt"):
    """
    Save and analyze results from double rotation analysis
    """
    with open(filename, 'w') as f:
        f.write("UCC Double Rotation Analysis Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total tests with non-zero gradients: {len(results)}\n\n")

        for result in results:
            f.write(f"Test {result['test_number']}:\n")
            f.write(f"Rotation 1 Generator: {result['rotation1_combination']}\n")
            f.write(f"Rotation 2 Generator: {result['rotation2_combination']}\n")
            f.write(f"Gradient Generator: {result['gradient_combination']}\n")
            f.write(f"Rotation angles: θ₁={result['theta1']}, θ₂={result['theta2']}\n")
            f.write(f"Non-zero gradients: {result['n_nonzero_gradients']}/{result['n_gradient_pauli_terms']}\n")
            f.write("-" * 60 + "\n")

            for pauli_name, pauli_data in result['gradients'].items():
                f.write(f"{pauli_name}: {pauli_data['pauli_term']}\n")
                f.write(f"  Gradient: {pauli_data['gradient']}\n")
                f.write(f"  Coefficient: {pauli_data['coeff']}\n")
            f.write("\n")

    # Enhanced analysis for double rotation results
    print(f"\nDOUBLE ROTATION GRADIENT ANALYSIS:")
    print(f"Total meaningful tests: {len(results)}")

    if len(results) == 0:
        print("No non-zero gradients found!")
        return

    # Find test with largest gradient magnitude
    max_gradient_mag = 0
    max_test = None
    max_pauli = None

    for result in results:
        for pauli_name, pauli_data in result['gradients'].items():
            grad_mag = abs(pauli_data['gradient'])
            if grad_mag > max_gradient_mag:
                max_gradient_mag = grad_mag
                max_test = result
                max_pauli = pauli_name

    print(f"Largest gradient magnitude: {max_gradient_mag:.6f}")
    print(f"  Rotation 1 generator: {max_test['rotation1_combination']}")
    print(f"  Rotation 2 generator: {max_test['rotation2_combination']}")
    print(f"  Gradient generator: {max_test['gradient_combination']}")
    print(f"  Pauli operator: {max_pauli}")

    # Compare with single rotation results (if available)
    all_gradients = []
    for result in results:
        for pauli_data in result['gradients'].values():
            all_gradients.append(abs(pauli_data['gradient']))

    all_gradients = np.array(all_gradients)
    print(f"\nDouble rotation gradient magnitude statistics:")
    print(f"  Mean: {np.mean(all_gradients):.6f}")
    print(f"  Std:  {np.std(all_gradients):.6f}")
    print(f"  Min:  {np.min(all_gradients):.6f}")
    print(f"  Max:  {np.max(all_gradients):.6f}")

    print(f"\nResults saved to: {filename}")


if __name__ == '__main__':

    # with open('../ham_lib/h4_sto-3g.pkl', 'rb') as f:
    #     lih_hamiltonian = pickle.load(f)
    # Load LiH Hamiltonian
    filename = '../ham_lib/beh2_fer.bin'
    with open(filename, 'rb') as f:
        lih_hamiltonian = pickle.load(f)

    # Convert to qubit Hamiltonian
    H_qubit = ferm_to_qubit(lih_hamiltonian)

    # Determine the actual number of qubits by finding the maximum qubit index
    max_qubit_index = 0
    for term in H_qubit.terms:
        for qubit_index, pauli_op in term:
            max_qubit_index = max(max_qubit_index, qubit_index)

    n_qubits = max_qubit_index + 1  # Add 1 because indices are 0-based
    print('n_qubits', n_qubits)
    n_electrons = 4  # LiH has 4 electrons

    print(f"LiH Hamiltonian loaded: {len(H_qubit.terms)} terms")
    print(f"Actual number of qubits needed: {n_qubits}")

    # Get reference occupation number for LiH (4 electrons)
    ref_occ = get_occ_no('lih', n_qubits)
    print(f"Reference occupation for LiH: {ref_occ}")

    # Get CISD ground state
    cisd_energy, cisd_wavefunction = get_cisd_gs('h4', H_qubit, n_qubits, gs_format='wfs', tf='bk')

    print(f"CISD Energy for LiH: {cisd_energy}")
    print(f"CISD Wavefunction shape: {cisd_wavefunction.shape}")
    print(f"Wavefunction norm: {sum(abs(cisd_wavefunction)**2)}")

    # Create Hartree-Fock reference state
    hf_wavefunction = create_hf_state(ref_occ, n_qubits)

    # Get sparse representations for efficient computation
    H_sparse = get_sparse_operator(H_qubit, n_qubits)

    # Choose analysis type
    analysis_type = "single"  # Change to "single" for single rotation analysis

    if analysis_type == "double":
        # Double rotation analysis
        print("\n" + "="*80)
        print("SYSTEMATIC DOUBLE UCC ANALYSIS - NON-ZERO GRADIENTS ONLY")
        print("="*80)

        results = analyze_double_ucc_combinations(H_sparse, n_qubits, n_electrons, hf_wavefunction,
                                                theta1=0.1, theta2=0.1, gradient_threshold=1e-10)

        save_and_analyze_double_results(results)

        print("\nDouble rotation analysis complete!")
        print("Check 'ucc_double_rotation_results.txt' for detailed results.")

    else:
        # Single rotation analysis (original)
        print("\n" + "="*80)
        print("SYSTEMATIC UCC ANALYSIS - NON-ZERO GRADIENTS ONLY")
        print("="*80)

        results = analyze_ucc_combinations_separate_generators(H_sparse, n_qubits, n_electrons, hf_wavefunction, cisd_wavefunction,
                                                             theta=0.2, gradient_threshold=1e-10)

        save_and_analyze_nonzero_results(results)

        print("\nSingle rotation analysis complete!")
        print("Check 'ucc_nonzero_gradient_results.txt' for detailed results.")
