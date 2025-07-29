from openfermion import QubitOperator
import numpy as np
from itertools import combinations, product


def generate_single_qubit_excitations(n_qubits, pauli_types=['X', 'Y']):
    """
    Generate single qubit excitation operators
    :param n_qubits: Number of qubits
    :param pauli_types: Types of Pauli operators to include
    :return: List of QubitOperator objects
    """
    single_excitations = []
    
    for qubit in range(n_qubits):
        for pauli in pauli_types:
            # Create single Pauli operator on this qubit
            op = QubitOperator(f'{pauli}{qubit}')
            single_excitations.append({
                'operator': op,
                'type': 'single',
                'qubits': [qubit],
                'pauli_pattern': pauli,
                'description': f'{pauli} on qubit {qubit}'
            })
    
    return single_excitations


def generate_two_qubit_excitations(n_qubits, pauli_types=['X', 'Y'], include_mixed=True):
    """
    Generate two-qubit excitation operators
    :param n_qubits: Number of qubits
    :param pauli_types: Types of Pauli operators to include
    :param include_mixed: Whether to include mixed Pauli patterns (XY, YX)
    :return: List of QubitOperator objects
    """
    two_excitations = []
    
    # Generate all unique pairs of qubits
    for q1, q2 in combinations(range(n_qubits), 2):
        # Same Pauli operators (XX, YY)
        for pauli in pauli_types:
            op = QubitOperator(f'{pauli}{q1} {pauli}{q2}')
            two_excitations.append({
                'operator': op,
                'type': 'double_same',
                'qubits': [q1, q2],
                'pauli_pattern': f'{pauli}{pauli}',
                'description': f'{pauli}{pauli} on qubits {q1},{q2}'
            })
        
        # Mixed Pauli operators (XY, YX) if requested
        if include_mixed and len(pauli_types) >= 2:
            for pauli1, pauli2 in product(pauli_types, repeat=2):
                if pauli1 != pauli2:  # Only mixed patterns
                    op = QubitOperator(f'{pauli1}{q1} {pauli2}{q2}')
                    two_excitations.append({
                        'operator': op,
                        'type': 'double_mixed',
                        'qubits': [q1, q2],
                        'pauli_pattern': f'{pauli1}{pauli2}',
                        'description': f'{pauli1}{pauli2} on qubits {q1},{q2}'
                    })
    
    return two_excitations


def generate_three_qubit_excitations(n_qubits, pauli_types=['X', 'Y'], max_operators=50):
    """
    Generate selected three-qubit excitation operators
    :param n_qubits: Number of qubits
    :param pauli_types: Types of Pauli operators to include
    :param max_operators: Maximum number of three-qubit operators to generate
    :return: List of QubitOperator objects
    """
    three_excitations = []
    count = 0
    
    # Generate all unique triplets of qubits
    for q1, q2, q3 in combinations(range(n_qubits), 3):
        if count >= max_operators:
            break
            
        # Same Pauli operators (XXX, YYY)
        for pauli in pauli_types:
            if count >= max_operators:
                break
            op = QubitOperator(f'{pauli}{q1} {pauli}{q2} {pauli}{q3}')
            three_excitations.append({
                'operator': op,
                'type': 'triple_same',
                'qubits': [q1, q2, q3],
                'pauli_pattern': f'{pauli}{pauli}{pauli}',
                'description': f'{pauli}{pauli}{pauli} on qubits {q1},{q2},{q3}'
            })
            count += 1
    
    return three_excitations


def generate_nearest_neighbor_excitations(n_qubits, pauli_types=['X', 'Y']):
    """
    Generate excitation operators only between nearest neighbor qubits
    :param n_qubits: Number of qubits
    :param pauli_types: Types of Pauli operators to include
    :return: List of QubitOperator objects
    """
    nn_excitations = []
    
    # Adjacent pairs only
    for q in range(n_qubits - 1):
        q1, q2 = q, q + 1
        
        # Same Pauli operators
        for pauli in pauli_types:
            op = QubitOperator(f'{pauli}{q1} {pauli}{q2}')
            nn_excitations.append({
                'operator': op,
                'type': 'nearest_neighbor',
                'qubits': [q1, q2],
                'pauli_pattern': f'{pauli}{pauli}',
                'description': f'{pauli}{pauli} on adjacent qubits {q1},{q2}'
            })
        
        # Mixed Pauli operators
        if len(pauli_types) >= 2:
            for pauli1, pauli2 in product(pauli_types, repeat=2):
                if pauli1 != pauli2:
                    op = QubitOperator(f'{pauli1}{q1} {pauli2}{q2}')
                    nn_excitations.append({
                        'operator': op,
                        'type': 'nearest_neighbor_mixed',
                        'qubits': [q1, q2],
                        'pauli_pattern': f'{pauli1}{pauli2}',
                        'description': f'{pauli1}{pauli2} on adjacent qubits {q1},{q2}'
                    })
    
    return nn_excitations


def generate_spin_flip_excitations(n_qubits):
    """
    Generate spin-flip excitation operators (X operators)
    :param n_qubits: Number of qubits
    :return: List of QubitOperator objects
    """
    spin_flips = []
    
    # Single spin flips
    for q in range(n_qubits):
        op = QubitOperator(f'X{q}')
        spin_flips.append({
            'operator': op,
            'type': 'spin_flip_single',
            'qubits': [q],
            'pauli_pattern': 'X',
            'description': f'Spin flip on qubit {q}'
        })
    
    # Two-qubit spin flips (XX)
    for q1, q2 in combinations(range(n_qubits), 2):
        op = QubitOperator(f'X{q1} X{q2}')
        spin_flips.append({
            'operator': op,
            'type': 'spin_flip_double',
            'qubits': [q1, q2],
            'pauli_pattern': 'XX',
            'description': f'Double spin flip on qubits {q1},{q2}'
        })
    
    return spin_flips


def generate_qubit_excitation_pool(n_qubits, pool_type='full', **kwargs):
    """
    Generate a qubit excitation pool based on the specified type
    
    :param n_qubits: Number of qubits
    :param pool_type: Type of pool to generate
        - 'full': Single and double excitations with X,Y
        - 'singles_only': Only single qubit excitations  
        - 'nearest_neighbor': Only nearest neighbor interactions
        - 'spin_flip': Only X operators (spin flips)
        - 'custom': Custom configuration via kwargs
    :param kwargs: Additional parameters for custom pool generation
    :return: List of QubitOperator dictionaries
    """
    
    pool = []
    
    if pool_type == 'full':
        # Generate comprehensive pool with singles and doubles
        singles = generate_single_qubit_excitations(n_qubits, ['X', 'Y'])
        doubles = generate_two_qubit_excitations(n_qubits, ['X', 'Y'], include_mixed=True)
        pool.extend(singles)
        pool.extend(doubles)
        
    elif pool_type == 'singles_only':
        # Only single qubit excitations
        pauli_types = kwargs.get('pauli_types', ['X', 'Y'])
        singles = generate_single_qubit_excitations(n_qubits, pauli_types)
        pool.extend(singles)
        
    elif pool_type == 'nearest_neighbor':
        # Only nearest neighbor interactions
        pauli_types = kwargs.get('pauli_types', ['X', 'Y'])
        nn_ops = generate_nearest_neighbor_excitations(n_qubits, pauli_types)
        pool.extend(nn_ops)
        
    elif pool_type == 'spin_flip':
        # Only spin flip operators (X)
        spin_flips = generate_spin_flip_excitations(n_qubits)
        pool.extend(spin_flips)
        
    elif pool_type == 'custom':
        # Custom pool based on kwargs
        include_singles = kwargs.get('include_singles', True)
        include_doubles = kwargs.get('include_doubles', True)
        include_triples = kwargs.get('include_triples', False)
        pauli_types = kwargs.get('pauli_types', ['X', 'Y'])
        include_mixed = kwargs.get('include_mixed', True)
        max_triple_ops = kwargs.get('max_triple_ops', 50)
        
        if include_singles:
            singles = generate_single_qubit_excitations(n_qubits, pauli_types)
            pool.extend(singles)
            
        if include_doubles:
            doubles = generate_two_qubit_excitations(n_qubits, pauli_types, include_mixed)
            pool.extend(doubles)
            
        if include_triples:
            triples = generate_three_qubit_excitations(n_qubits, pauli_types, max_triple_ops)
            pool.extend(triples)
    
    else:
        raise ValueError(f"Unknown pool_type: {pool_type}")
    
    # Add metadata
    for i, op_dict in enumerate(pool):
        op_dict['pool_index'] = i
        op_dict['pool_type'] = pool_type
        
    print(f"Generated qubit excitation pool with {len(pool)} operators (type: {pool_type})")
    
    return pool


def get_qubit_excitation_operators_only(n_qubits, pool_type='full', **kwargs):
    """
    Simplified version that returns only the QubitOperator objects
    :param n_qubits: Number of qubits
    :param pool_type: Type of pool to generate
    :return: List of QubitOperator objects
    """
    full_pool = generate_qubit_excitation_pool(n_qubits, pool_type, **kwargs)
    return [item['operator'] for item in full_pool]


def analyze_qubit_excitation_pool(pool):
    """
    Analyze the composition of the qubit excitation pool
    :param pool: List of operator dictionaries
    """
    if not pool:
        print("Empty pool!")
        return
    
    print(f"\n=== QUBIT EXCITATION POOL ANALYSIS ===")
    print(f"Total operators: {len(pool)}")
    
    # Count by type
    type_counts = {}
    for op_dict in pool:
        op_type = op_dict['type']
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
    
    print(f"\nOperator types:")
    for op_type, count in sorted(type_counts.items()):
        print(f"  {op_type}: {count}")
    
    # Count by Pauli pattern
    pauli_counts = {}
    for op_dict in pool:
        pattern = op_dict['pauli_pattern']
        pauli_counts[pattern] = pauli_counts.get(pattern, 0) + 1
    
    print(f"\nPauli patterns:")
    for pattern, count in sorted(pauli_counts.items()):
        print(f"  {pattern}: {count}")
    
    # Show examples
    print(f"\nExample operators:")
    for i, op_dict in enumerate(pool[:5]):
        print(f"  {i+1}. {op_dict['description']}")
    
    # Show qubit usage statistics
    qubit_usage = {}
    for op_dict in pool:
        for qubit in op_dict['qubits']:
            qubit_usage[qubit] = qubit_usage.get(qubit, 0) + 1
    
    if qubit_usage:
        print(f"\nQubit usage (how many operators act on each qubit):")
        for qubit in sorted(qubit_usage.keys()):
            print(f"  Qubit {qubit}: {qubit_usage[qubit]} operators")


# Example usage and test function
if __name__ == "__main__":
    print("Testing qubit excitation pool generation...")
    
    # Test different pool types
    test_cases = [
        ("Full pool (4 qubits)", 4, 'full'),
        ("Singles only (6 qubits)", 6, 'singles_only'),
        ("Nearest neighbor (4 qubits)", 4, 'nearest_neighbor'),
        ("Spin flip (4 qubits)", 4, 'spin_flip'),
    ]
    
    for description, n_qubits, pool_type in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {description}")
        pool = generate_qubit_excitation_pool(n_qubits, pool_type)
        analyze_qubit_excitation_pool(pool)
    
    # Test custom pool
    print(f"\n{'='*60}")
    print("Testing custom pool (6 qubits, with triples)")
    custom_pool = generate_qubit_excitation_pool(
        6, 'custom',
        include_singles=True,
        include_doubles=True, 
        include_triples=True,
        pauli_types=['X', 'Y'],
        include_mixed=False,  # Only XX, YY (no XY, YX)
        max_triple_ops=20
    )
    analyze_qubit_excitation_pool(custom_pool) 