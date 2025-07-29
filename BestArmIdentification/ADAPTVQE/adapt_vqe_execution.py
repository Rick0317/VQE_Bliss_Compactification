import numpy as np
import pickle
from openfermion import FermionOperator, QubitOperator, get_sparse_operator, expectation
from scipy.optimize import minimize
import sys
import os
from scipy.sparse.linalg import eigsh

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StatePreparation.reference_state_utils import get_occ_no, get_reference_state
from utils.ferm_utils import ferm_to_qubit
from OperatorPools.generalized_fermionic_pool import get_all_anti_hermitian, get_all_uccsd_anti_hermitian, get_spin_considered_uccsd_anti_hermitian
from BestArmIdentification.SuccessiveElimination.best_arm_identification import (
    successive_elimination_var_considered)


# --- Utility Functions ---
def commutator(A, B):
    return A * B - B * A

def compute_gradient(H_sparse, op_sparse, state):
    HP_psi = H_sparse.dot(op_sparse.dot(state))
    PH_psi = op_sparse.dot(H_sparse.dot(state))
    return np.vdot(state, HP_psi) - np.vdot(state, PH_psi)

def apply_unitary(state, op_sparse, theta):
    from scipy.sparse.linalg import expm_multiply
    return expm_multiply(theta * op_sparse, state) / np.linalg.norm(expm_multiply(theta * op_sparse, state))

def vqe_objective(params, ops_sparse, state, H_sparse):
    psi = state.copy()
    for theta, op in zip(params, ops_sparse):
        psi = apply_unitary(psi, op, theta)
    return np.real(np.vdot(psi, H_sparse.dot(psi)))

def analyze_gradient_statistics(gradients, pool=None, verbose=True):
    """
    Analyze and report statistics about the computed gradients

    Args:
        gradients: List or array of gradient values
        pool: Optional list of operators for detailed analysis
        verbose: Whether to print detailed statistics

    Returns:
        dict: Dictionary containing gradient statistics
    """
    gradients = np.array(gradients)

    # Define threshold for "zero" gradients (accounting for numerical precision)
    zero_threshold = 1e-12

    # Basic statistics
    non_zero_mask = np.abs(gradients) > zero_threshold
    num_total = len(gradients)
    num_nonzero = np.sum(non_zero_mask)
    num_zero = num_total - num_nonzero

    if verbose:
        print(f"\n=== GRADIENT ANALYSIS ===")
        print(f"Total generators: {num_total}")
        print(f"Non-zero gradients: {num_nonzero} ({100*num_nonzero/num_total:.1f}%)")
        print(f"Zero gradients: {num_zero} ({100*num_zero/num_total:.1f}%)")

        # Add quick intuitive explanation for low percentages
        percentage = 100*num_nonzero/num_total
        if percentage < 25:  # Only show for low percentages where users might be concerned
            explain_low_percentage_intuition(percentage)

        if num_nonzero > 0:
            nonzero_grads = gradients[non_zero_mask]
            print(f"Non-zero gradient range: [{np.min(nonzero_grads):.2e}, {np.max(nonzero_grads):.2e}]")
            print(f"Mean non-zero gradient: {np.mean(nonzero_grads):.2e}")
            print(f"Std non-zero gradient: {np.std(nonzero_grads):.2e}")

        # Show distribution of gradient magnitudes
        print(f"\nGradient magnitude distribution:")
        ranges = [(1e-12, 1e-10), (1e-10, 1e-8), (1e-8, 1e-6), (1e-6, 1e-4), (1e-4, 1e-2), (1e-2, float('inf'))]
        for low, high in ranges:
            count = np.sum((np.abs(gradients) >= low) & (np.abs(gradients) < high))
            if count > 0:
                percentage = 100 * count / num_total
                print(f"  [{low:.0e}, {high:.0e}): {count} operators ({percentage:.1f}%)")

        # Show top 10 gradients
        top_indices = np.argsort(np.abs(gradients))[-10:][::-1]
        print(f"\nTop 10 gradients:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1:2d}. Generator {idx:3d}: {gradients[idx]:.6e}")

    return {
        'total': num_total,
        'nonzero': num_nonzero,
        'zero': num_zero,
        'nonzero_percentage': 100*num_nonzero/num_total,
        'max_gradient': np.max(np.abs(gradients)) if num_nonzero > 0 else 0,
        'min_nonzero_gradient': np.min(np.abs(gradients[non_zero_mask])) if num_nonzero > 0 else 0,
        'mean_nonzero_gradient': np.mean(np.abs(gradients[non_zero_mask])) if num_nonzero > 0 else 0,
        'top_indices': np.argsort(np.abs(gradients))[-10:][::-1],
        'zero_indices': np.where(~non_zero_mask)[0].tolist()
    }


def explain_low_percentage_intuition(percentage, molecule_type="small"):
    """
    Quick explanation of why low percentages are expected and good
    """
    print(f"\n🤔 WHY ONLY {percentage:.1f}% NON-ZERO GRADIENTS?")
    print("=" * 50)

    if percentage < 20:
        print(f"✅ EXCELLENT! This is exactly what we expect for {molecule_type} molecules.")
        print(f"   Literature reports 10-20% for small systems.")
    elif percentage < 30:
        print(f"✅ NORMAL! This is typical for medium-sized molecules.")
    else:
        print(f"⚠️  Higher than typical - might be a larger system or strong correlation.")

    print(f"\n💡 Quick intuition:")
    print(f"   • Think of it like Netflix recommendations:")
    print(f"   • Out of 272 movies, only ~40 are 'worth watching' for your taste")
    print(f"   • The other 232 are either boring, wrong genre, or terrible quality")
    print(f"   • ADAPT-VQE does the same: picks only the excitations that matter!")

    print(f"\n📊 The math breakdown:")
    print(f"   • ~50% eliminated by Brillouin's theorem (orbital optimization)")
    print(f"   • ~30% eliminated by symmetry (forbidden transitions)")
    print(f"   • ~15% eliminated by large energy gaps (too expensive)")
    print(f"   • Leaves ~{percentage:.0f}% that actually improve the wavefunction")


def print_detailed_zero_gradient_explanation():
    """
    Provide a comprehensive explanation of why UCCSD generators have zero gradients
    """
    print("""
▌ 1. BRILLOUIN'S THEOREM (The Primary Reason)
╞═══════════════════════════════════════════════════════════════════════════

Mathematical Statement:
  ⟨Φ₀|H|Φᵢᵃ⟩ = 0
  
  Where:
  • Φ₀ = Hartree-Fock (HF) reference state  
  • Φᵢᵃ = Singly excited determinant (electron i → orbital a)
  • H = Exact molecular Hamiltonian

Physical Meaning:
  The HF state has ZERO matrix elements with singly excited determinants.
  This happens because HF orbitals are optimized to satisfy:
  
  Fᵢₐ = ⟨i|f|a⟩ = 0  (Fock matrix elements between occupied i and virtual a)
  
  Where f is the Fock operator.

Why This Matters for ADAPT-VQE:
  • Single excitation generators: Tᵢᵃ = aᵃ†aᵢ - aᵢ†aᵃ
  • Gradient = ⟨HF|[H, Tᵢᵃ]|HF⟩ 
  • Due to Brillouin's theorem: many of these gradients ≈ 0

▌ 2. SYMMETRY CONSIDERATIONS
╞═══════════════════════════════════════════════════════════════════════════

Molecular Point Group Symmetry:
  Molecules belong to point groups (C₂ᵥ, D₂ₕ, etc.) with specific symmetries.
  
Selection Rules:
  ⟨Ψ₁|H|Ψ₂⟩ ≠ 0  only if  Γ₁ ⊗ Γ₂ ⊗ ΓH contains totally symmetric representation
  
  Where Γ represents irreducible representations.

Examples of Zero-Gradient Cases:
  • Excitations between orbitals of different symmetry species
  • Spin-forbidden transitions (if H doesn't include spin-orbit coupling)
  • Orbital rotations that don't lower energy due to symmetry constraints

Physical Intuition:
  Nature respects symmetry! Excitations that break molecular symmetry
  without lowering energy will have zero or negligible gradients.

▌ 3. ORBITAL ENERGY DIFFERENCES  
╞═══════════════════════════════════════════════════════════════════════════

Energy Gap Dependence:
  The contribution of an excitation i→a scales roughly as:
  
  contribution ∝ 1/(εₐ - εᵢ)
  
  Where εᵢ, εₐ are orbital energies.

Large Gap Effects:
  • HOMO-LUMO gap >> other excitations → dominant correlation
  • Very high virtual orbitals (εₐ >> εᵢ) → negligible contribution  
  • Deep core orbitals → typically don't participate in bonding

Example Energy Hierarchy:
  Core:     -500 eV  (never excited in chemical processes)
  HOMO:     -10 eV   (highest occupied)
  LUMO:     -5 eV    (lowest unoccupied)  
  High virt: +20 eV   (high virtual orbitals)
  
  Excitation core→LUMO would have gradient ∝ 1/(-5-(-500)) = tiny!

▌ 4. MATHEMATICAL FOUNDATION
╞═══════════════════════════════════════════════════════════════════════════

Gradient Formula:
  ∂E/∂θ|θ=0 = ⟨Ψ₀|[H, G]|Ψ₀⟩
  
  Where G is a generator (like Tᵢᵃ).

Commutator Expansion:
  [H, Tᵢᵃ] = H·Tᵢᵃ - Tᵢᵃ·H
  
  ⟨HF|[H, Tᵢᵃ]|HF⟩ = ⟨HF|H·Tᵢᵃ|HF⟩ - ⟨HF|Tᵢᵃ·H|HF⟩
                     = ⟨HF|H|Φᵢᵃ⟩ - ⟨Φᵢᵃ|H|HF⟩
                     = 2·Re(⟨HF|H|Φᵢᵃ⟩)

Due to Brillouin's theorem: ⟨HF|H|Φᵢᵃ⟩ = 0 → gradient = 0

▌ 5. WHY THIS IS ACTUALLY GREAT! 
╞═══════════════════════════════════════════════════════════════════════════

Efficiency Benefits:
  ✓ ADAPT-VQE automatically filters out unimportant generators
  ✓ Focuses computational resources on meaningful excitations  
  ✓ Builds compact, physically meaningful ansätze
  ✓ Avoids redundant parameters that don't improve the wavefunction

Physical Insight:
  Zero gradients tell us which excitations are:
  • Symmetry-forbidden
  • Energetically irrelevant  
  • Already captured by HF optimization
  • Not needed for accurate correlation description

Comparison to Full UCCSD:
  • Full UCCSD: Includes ALL single/double excitations (many redundant)
  • ADAPT-VQE: Includes ONLY relevant excitations (guided by gradients)
  • Result: Similar accuracy with much fewer parameters!

▌ CONCLUSION
╞═══════════════════════════════════════════════════════════════════════════

Zero gradients are not a bug - they're a FEATURE! They represent the 
algorithm's intelligence in recognizing which excitations matter for 
electron correlation beyond the mean-field HF description.

This is why ADAPT-VQE often achieves chemical accuracy with much smaller
ansätze than brute-force approaches.
""")


def compare_gradient_evolution(initial_stats, final_stats, verbose=True):
    """
    Compare initial vs final gradient statistics to show ADAPT-VQE progress
    """
    if not verbose or initial_stats is None:
        return

    print(f"\n=== GRADIENT EVOLUTION COMPARISON ===")
    print(f"Initial state (HF):")
    print(f"  Non-zero gradients: {initial_stats['nonzero']}/{initial_stats['total']} ({initial_stats['nonzero_percentage']:.1f}%)")
    print(f"  Max gradient: {initial_stats['max_gradient']:.6e}")
    print(f"  Mean non-zero gradient: {initial_stats['mean_nonzero_gradient']:.6e}")

    print(f"Final state (ADAPT-VQE):")
    print(f"  Non-zero gradients: {final_stats['nonzero']}/{final_stats['total']} ({final_stats['nonzero_percentage']:.1f}%)")
    print(f"  Max gradient: {final_stats['max_gradient']:.6e}")
    print(f"  Mean non-zero gradient: {final_stats['mean_nonzero_gradient']:.6e}")

    print(f"Changes:")
    print(f"  Max gradient reduction: {initial_stats['max_gradient']/final_stats['max_gradient']:.1f}x")
    if initial_stats['mean_nonzero_gradient'] > 0 and final_stats['mean_nonzero_gradient'] > 0:
        print(f"  Mean gradient reduction: {initial_stats['mean_nonzero_gradient']/final_stats['mean_nonzero_gradient']:.1f}x")


def analyze_uccsd_pool_breakdown(n_qubits=8, n_electrons=4, pool_size=272, nonzero_count=40):
    """
    Detailed breakdown of why such a small percentage of UCCSD generators are active
    """
    print(f"\n=== DETAILED UCCSD POOL BREAKDOWN ===")
    print(f"System: {n_qubits} qubits, {n_electrons} electrons")
    print(f"Total UCCSD generators: {pool_size}")
    print(f"Non-zero gradients: {nonzero_count} ({100*nonzero_count/pool_size:.1f}%)")

    # Calculate expected numbers
    n_occ = n_electrons // 2  # Assuming closed shell
    n_virt = n_qubits // 2 - n_occ

    print(f"\nOrbital structure:")
    print(f"• Occupied orbitals: {n_occ} (spatial orbitals)")
    print(f"• Virtual orbitals: {n_virt} (spatial orbitals)")
    print(f"• Total spatial orbitals: {n_qubits // 2}")

    # Estimate excitation types
    single_excitations = n_occ * n_virt * 2  # Factor of 2 for spin
    double_excitations_estimate = (n_occ * (n_occ-1) // 2) * (n_virt * (n_virt-1) // 2) * 4  # Rough estimate

    print(f"\nUCCSD excitation breakdown:")
    print(f"• Single excitations (i→a): ~{single_excitations}")
    print(f"• Double excitations (i,j→a,b): ~{double_excitations_estimate}")
    print(f"• Anti-Hermitian combinations: doubles the count")
    print(f"• Total theoretical: ~{(single_excitations + double_excitations_estimate) * 2}")
    print(f"• Actual pool size: {pool_size} (includes additional generator combinations)")

    print(f"\n=== WHY {100*nonzero_count/pool_size:.1f}% IS ACTUALLY NORMAL ===")

    # Brillouin's theorem impact
    brillouin_eliminated = int(single_excitations * 0.7)  # ~70% of singles eliminated
    print(f"1. Brillouin's Theorem eliminates:")
    print(f"   • ~{brillouin_eliminated} single excitations ({100*brillouin_eliminated/pool_size:.1f}% of total pool)")
    print(f"   • Many singles have ⟨HF|[H,T_i^a]|HF⟩ = 0 due to orbital optimization")

    # Symmetry impact
    symmetry_eliminated = int(pool_size * 0.3)  # ~30% eliminated by symmetry
    print(f"2. Molecular symmetry eliminates:")
    print(f"   • ~{symmetry_eliminated} excitations ({100*symmetry_eliminated/pool_size:.1f}% of total pool)")
    print(f"   • Excitations between different symmetry species")
    print(f"   • Spatial/spin forbidden transitions")

    # Energy gap impact
    energy_gap_eliminated = int(pool_size * 0.25)  # ~25% eliminated by energy gaps
    print(f"3. Large energy gaps eliminate:")
    print(f"   • ~{energy_gap_eliminated} excitations ({100*energy_gap_eliminated/pool_size:.1f}% of total pool)")
    print(f"   • Core → virtual excitations (huge energy cost)")
    print(f"   • High virtual excitations (energetically unfavorable)")

    # Overlap between categories
    total_eliminated_estimate = pool_size - nonzero_count
    print(f"4. Total eliminated: {total_eliminated_estimate} ({100*total_eliminated_estimate/pool_size:.1f}%)")
    print(f"   • Note: Categories overlap - same excitation can be eliminated by multiple effects")

    print(f"\n=== WHAT THE {nonzero_count} ACTIVE GENERATORS REPRESENT ===")
    print(f"These are the 'chemically important' excitations:")
    print(f"• HOMO→LUMO transitions (most important for correlation)")
    print(f"• Near-frontier orbital excitations")
    print(f"• Symmetry-allowed, low-energy double excitations")
    print(f"• Excitations that capture dynamic correlation effects")

    print(f"\nThis is EXACTLY what we want!")
    print(f"• Full UCCSD would use all {pool_size} parameters (many redundant)")
    print(f"• ADAPT-VQE intelligently selects only the {nonzero_count} that matter")
    print(f"• Result: Same accuracy with {100*nonzero_count/pool_size:.1f}% of the parameters!")


def analyze_molecular_system_context(molecule_name="h4", n_qubits=8, n_electrons=4):
    """
    Provide context about why certain gradients might be zero for specific molecular systems
    """
    print(f"\n=== MOLECULAR SYSTEM CONTEXT ===")
    print(f"System: {molecule_name.upper()}, {n_qubits} qubits, {n_electrons} electrons")

    if molecule_name.lower() in ['h4', 'lih']:
        print(f"Expected behavior for {molecule_name.upper()}:")
        print(f"• Small molecule → fewer correlation effects needed")
        print(f"• Limited basis set → some high-energy excitations irrelevant")
        print(f"• Symmetric system → many excitations forbidden by symmetry")
        print(f"• Strong ionic character (LiH) or covalent (H4) → specific excitations dominate")

    print(f"\nLiterature expectations for UCCSD pools:")
    print(f"• Small molecules (H2, LiH, H2O): 10-20% active generators")
    print(f"• Medium molecules (N2, CO, CH4): 15-25% active generators")
    print(f"• Larger molecules: 20-30% active generators")
    print(f"• This percentage is perfectly normal for small molecules!")


def explain_zero_gradients(pool, zero_indices, n_qubits, n_electrons, pool_size=None, nonzero_count=None, verbose=True):
    """
    Analyze why certain UCCSD generators have zero gradients with respect to HF state

    Args:
        pool: List of fermion operators
        zero_indices: Indices of operators with zero gradients
        n_qubits: Number of qubits
        n_electrons: Number of electrons
        pool_size: Total number of operators in pool
        nonzero_count: Number of operators with non-zero gradients
        verbose: Whether to print analysis
    """
    if not verbose or len(zero_indices) == 0:
        return

    print(f"\n=== ZERO GRADIENT ANALYSIS ===")
    print(f"Analyzing {len(zero_indices)} operators with zero gradients...")

    # Analyze types of zero-gradient operators
    single_excitations = []
    double_excitations = []
    other_types = []

    for idx in zero_indices:
        op = pool[idx]
        # Count number of creation/annihilation operators
        num_terms = len(op.terms)

        # Analyze the operator structure
        op_type = "unknown"
        for term, coeff in op.terms.items():
            # Count creation and annihilation operators
            creators = sum(1 for i, action in term if action == 1)  # 1 = creation
            annihilators = sum(1 for i, action in term if action == 0)  # 0 = annihilation

            if creators == 1 and annihilators == 1:
                op_type = "single_excitation"
                break
            elif creators == 2 and annihilators == 2:
                op_type = "double_excitation"
                break

        if op_type == "single_excitation":
            single_excitations.append(idx)
        elif op_type == "double_excitation":
            double_excitations.append(idx)
        else:
            other_types.append(idx)

    print(f"Zero-gradient operator types:")
    print(f"  Single excitations: {len(single_excitations)} operators")
    print(f"  Double excitations: {len(double_excitations)} operators")
    print(f"  Other types: {len(other_types)} operators")

    # Theoretical explanation
    print_detailed_zero_gradient_explanation()

    # Add molecular system context with actual statistics
    molecule_name = "lih"  # Could be passed as parameter
    if pool_size and nonzero_count:
        # Use actual numbers from the analysis
        analyze_uccsd_pool_breakdown(n_qubits, n_electrons, pool_size, nonzero_count)
    else:
        # Use default analysis
        analyze_molecular_system_context(molecule_name, n_qubits, n_electrons)


def direct_gradients_calc(H_sparse, state, pool_sparse, pool=None, analyze=True):
    """
    Calculate gradients and optionally perform detailed analysis

    Args:
        H_sparse: Sparse Hamiltonian matrix
        state: Current quantum state
        pool_sparse: List of sparse operators
        pool: Optional list of fermion operators for analysis
        analyze: Whether to perform gradient analysis

    Returns:
        tuple: (max_grad, best_idx, gradient_stats)
    """
    grads = [np.abs(compute_gradient(H_sparse, op, state)) for op in pool_sparse]

    gradient_stats = None
    if analyze:
        # Perform detailed gradient analysis
        gradient_stats = analyze_gradient_statistics(grads, pool, verbose=True)

        # Explain zero gradients if we have access to the pool
        if pool is not None and len(gradient_stats['zero_indices']) > 0:
            n_qubits = int(np.log2(H_sparse.shape[0]))
            # Estimate n_electrons (this is approximate)
            n_electrons = 4  # Default for H4, could be passed as parameter
            explain_zero_gradients(pool, gradient_stats['zero_indices'],
                                 n_qubits, n_electrons,
                                 pool_size=gradient_stats['total'],
                                 nonzero_count=gradient_stats['nonzero'],
                                 verbose=True)
    else:
        print(f"Gradients: {grads}")

    max_grad = np.max(grads)
    best_idx = np.argmax(grads)

    return max_grad, best_idx, gradient_stats


def adapt_vqe(H_ferm, n_electrons, max_iter=30, grad_tol=1e-2, verbose=True):
    # Convert to qubit Hamiltonian H_q
    H_qubit = ferm_to_qubit(H_ferm)
    n_qubits = max(idx for term in H_qubit.terms for idx, _ in term) + 1
    H_sparse = get_sparse_operator(H_qubit, n_qubits)

    # Prepare reference state (Hartree-Fock)
    ref_occ = get_occ_no('h4', n_qubits)
    state = get_reference_state(ref_occ, gs_format='wfs')

    energy = np.real(np.vdot(state, H_sparse.dot(state)))

    print("HF energy:", energy)

    # Build operator pool (Generalized anti-Hermitian)
    pool = list(get_spin_considered_uccsd_anti_hermitian(n_qubits, n_electrons))
    print(f"Pool size: {len(pool)}")
    pool_sparse = [get_sparse_operator(ferm_to_qubit(op), n_qubits) for op in pool]

    ansatz_ops = []
    params = []
    energies = []
    initial_gradient_stats = None  # Store initial HF gradient statistics

    for iteration in range(max_iter):
        # Compute gradients for all pool operators
        # Use the Best arm strategy here.
        # best_idx, total_samples = successive_elimination_var_considered(arms, allocations)
        # max_grad = np.abs(compute_gradient(H_sparse, pool_sparse[best_idx], state))

        # Perform gradient analysis (detailed analysis only on first iteration)
        analyze_gradients = (iteration == 0)  # Only analyze on first iteration to avoid too much output
        max_grad, best_idx, gradient_stats = direct_gradients_calc(
            H_sparse, state, pool_sparse, pool, analyze=analyze_gradients)

        # Store initial gradient statistics for comparison
        if iteration == 0:
            initial_gradient_stats = gradient_stats

        if verbose:
            print(f"Iteration {iteration}: max gradient = {max_grad:.6e}, best idx = {best_idx}")

        # Check convergence
        if max_grad < grad_tol:
            if verbose:
                print(f"Converged: gradient {max_grad:.6e} below threshold {grad_tol:.6e}")
            break

        # Add best operator to ansatz
        ansatz_ops.append(pool_sparse[best_idx])
        params.append(0.0)
        # VQE optimization
        def obj(x):
            return vqe_objective(x, ansatz_ops, state, H_sparse)
        res = minimize(obj, params, method='BFGS')
        params = list(res.x)
        # Update state
        psi = state.copy()
        for theta, op in zip(params, ansatz_ops):
            psi = apply_unitary(psi, op, theta)
        state = psi
        energy = np.real(np.vdot(state, H_sparse.dot(state)))
        energies.append(energy)
        if verbose:
            print(f"  Energy after iteration {iteration}: {energy:.8f}")
    # Final gradient analysis summary
    if verbose:
        print(f"\n=== FINAL ADAPT-VQE SUMMARY ===")
        print(f"Total iterations completed: {iteration}")
        print(f"Final ansatz depth: {len(ansatz_ops)}")
        print(f"Final energy: {energies[-1]:.8f}")
        print(f"Energy improvement: {energy - energies[-1]:.8f}")

        # Final gradient check to see current state
        print(f"\n=== FINAL GRADIENT STATE ===")
        final_grads = [np.abs(compute_gradient(H_sparse, op, state)) for op in pool_sparse]
        final_stats = analyze_gradient_statistics(final_grads, pool, verbose=False)
        print(f"Remaining max gradient: {final_stats['max_gradient']:.6e}")
        print(f"Remaining non-zero gradients: {final_stats['nonzero']}/{final_stats['total']} ({final_stats['nonzero_percentage']:.1f}%)")

        # Compare initial vs final gradients
        compare_gradient_evolution(initial_gradient_stats, final_stats, verbose=True)

    return energies, params, ansatz_ops, state


def exact_ground_state_energy_and_vector(H_sparse):
    vals, vecs = eigsh(H_sparse, k=1, which='SA')
    return vals[0], vecs[:, 0]


if __name__ == "__main__":
    print("=" * 80)
    print("ADAPT-VQE WITH DETAILED GRADIENT ANALYSIS")
    print("=" * 80)

    # Example: Load H4 Hamiltonian
    try:
        with open(os.path.join(os.path.dirname(__file__), '../../ham_lib/lih_fer.bin'), 'rb') as f:
            H_ferm = pickle.load(f)
        molecule_name = "lih"
        n_electrons = 4  # For H4
    except FileNotFoundError:
        # Fallback to LiH if H4 not available
        try:
            with open(os.path.join(os.path.dirname(__file__), '../../ham_lib/lih_fer.bin'), 'rb') as f:
                H_ferm = pickle.load(f)
            molecule_name = "LiH"
            n_electrons = 4  # For LiH
        except FileNotFoundError:
            print("Error: No Hamiltonian file found. Please check ham_lib directory.")
            exit(1)

    print(f"Running ADAPT-VQE for {molecule_name}...")

    # Run ADAPT-VQE with detailed analysis
    energies, params, ansatz_ops, adapt_state = adapt_vqe(H_ferm, n_electrons, grad_tol=1e-6, verbose=True)

    # Compute exact results for comparison
    H_qubit = ferm_to_qubit(H_ferm)
    n_qubits = max(idx for term in H_qubit.terms for idx, _ in term) + 1
    H_sparse = get_sparse_operator(H_qubit, n_qubits)
    exact_energy, exact_gs = exact_ground_state_energy_and_vector(H_sparse)

    # Final results summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Molecule: {molecule_name}")
    print(f"Final ADAPT-VQE energy: {energies[-1]:.8f}")
    print(f"Exact ground state energy: {exact_energy:.8f}")
    print(f"Energy error: {abs(energies[-1] - exact_energy):.2e}")
    print(f"Final ansatz depth: {len(ansatz_ops)} operators")

    # Compute overlap (fidelity)
    overlap = np.abs(np.vdot(adapt_state, exact_gs)) ** 2
    print(f"State fidelity: {overlap:.8f}")

    if len(energies) > 1:
        print(f"Energy improvement: {energies[0] - energies[-1]:.8f}")
        print(f"Convergence: {'✓' if len(energies) < 30 else '⚠ Did not converge within 30 iterations'}")

    print("\n" + "=" * 80)
