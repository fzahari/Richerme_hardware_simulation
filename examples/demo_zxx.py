#!/usr/bin/env python3
"""
Richerme Ion Analog Quantum Gate Synthesis - Demonstration Script

This script demonstrates the quantum gate synthesis capabilities of the
Richerme Ion Analog library, showcasing the UMQ-Rz-UMQ construction
for multi-qubit Pauli string operations on trapped-ion hardware.
"""

import sys
from pathlib import Path
# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import List
from richerme_ion_analog import (
    Z1X2X3, n_body_string, target_pauli_string_unitary,
    unitary_distance, UMQ, Jij_from_multimode
)

# Constants for demonstration
HIGH_PRECISION = 1e-12


def demonstrate_basic_synthesis():
    """Demonstrate basic Z1X2X3 gate synthesis."""
    print("=" * 60)
    print("Basic Gate Synthesis: Z₁X₂X₃ Operation")
    print("=" * 60)
    
    evolution_time = 0.314
    print(f"Evolution parameter: t = {evolution_time}")
    
    # Synthesize using hardware-native operations
    synthesized_gate = Z1X2X3(evolution_time)
    print(f"Synthesized gate shape: {synthesized_gate.shape}")
    
    # Generate ideal target for comparison
    target_gate = target_pauli_string_unitary('ZXX', evolution_time)
    print(f"Target gate shape: {target_gate.shape}")
    
    # Validate synthesis accuracy
    fidelity_error = unitary_distance(target_gate, synthesized_gate)
    print(f"Synthesis fidelity error: {fidelity_error:.2e}")
    
    if fidelity_error < HIGH_PRECISION:
        print("✓ Synthesis achieved machine precision accuracy!")
    else:
        print("⚠ Synthesis accuracy below machine precision")
    
    print()


def demonstrate_parameter_sweep():
    """Demonstrate synthesis accuracy across parameter range."""
    print("=" * 60)
    print("Parameter Sweep Analysis")
    print("=" * 60)
    
    time_points = np.linspace(0.1, 1.0, 10)
    fidelity_errors = []
    
    print("Testing synthesis accuracy across evolution times...")
    for i, t in enumerate(time_points):
        synthesized = Z1X2X3(t)
        target = target_pauli_string_unitary('ZXX', t)
        error = unitary_distance(target, synthesized)
        fidelity_errors.append(error)
        
        print(f"  t = {t:.3f}: error = {error:.2e}")
    
    max_error = np.max(fidelity_errors)
    mean_error = np.mean(fidelity_errors)
    
    print(f"\nSummary Statistics:")
    print(f"  Maximum error: {max_error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  All errors < 1e-12: {all(e < HIGH_PRECISION for e in fidelity_errors)}")
    print()


def demonstrate_custom_pauli_strings():
    """Demonstrate synthesis of custom Pauli string operations."""
    print("=" * 60)
    print("Custom Pauli String Synthesis")
    print("=" * 60)
    
    test_cases = [
        (['Z', 'X', 'X'], 'ZXX', 0.25),
        (['Y', 'X', 'X'], 'YXX', 0.35),
    ]
    
    for pauli_axes, pauli_string, time_param in test_cases:
        print(f"\nSynthesizing {pauli_string} operation:")
        print(f"  Pauli axes: {pauli_axes}")
        print(f"  Evolution time: {time_param}")
        
        # Synthesize using the n_body_string function
        synthesized = n_body_string(pauli_axes, time_param)
        target = target_pauli_string_unitary(pauli_string, time_param)
        
        error = unitary_distance(target, synthesized)
        print(f"  Fidelity error: {error:.2e}")
        
        if error < HIGH_PRECISION:
            print(f"  ✓ {pauli_string} synthesis successful")
        else:
            print(f"  ⚠ {pauli_string} synthesis has elevated error")
    
    print()


def demonstrate_umq_operation():
    """Demonstrate the UMQ (global entangling) operation."""
    print("=" * 60)
    print("UMQ Global Entangling Operation")
    print("=" * 60)
    
    n_qubits = 3
    interaction_strength = np.pi / 4
    
    print(f"System size: {n_qubits} qubits")
    print(f"Interaction strength: χ = π/4")
    
    # Generate UMQ operation
    umq_gate = UMQ(n_qubits, interaction_strength)
    print(f"UMQ gate dimensions: {umq_gate.shape}")
    
    # Verify unitarity
    identity = np.eye(2**n_qubits, dtype=complex)
    unitarity_check = umq_gate @ umq_gate.conj().T
    unitarity_error = np.linalg.norm(unitarity_check - identity)
    
    print(f"Unitarity check error: {unitarity_error:.2e}")
    
    if unitarity_error < HIGH_PRECISION:
        print("✓ UMQ operation is unitary")
    
    # Test with zero interaction
    umq_zero = UMQ(n_qubits, 0.0)
    identity_error = np.linalg.norm(umq_zero - identity)
    print(f"Zero interaction → identity error: {identity_error:.2e}")
    
    print()


def demonstrate_hardware_coupling():
    """Demonstrate realistic hardware coupling calculation."""
    print("=" * 60)
    print("Hardware Coupling Matrix Calculation")
    print("=" * 60)
    
    # 3-ion linear chain parameters
    num_ions = 3
    print(f"Ion chain size: {num_ions} ions")
    
    # Simplified normal mode matrix
    normal_modes = np.eye(num_ions)
    print(f"Normal mode matrix: {num_ions}×{num_ions} identity (simplified)")
    
    # Realistic parameter values
    mode_frequencies = np.array([1.0, 1.5, 2.0])  # MHz
    drive_detunings = np.array([0.8, 1.2])  # MHz  
    rabi_frequencies = np.array([15.0, 20.0])  # kHz
    recoil_frequency = 0.02  # kHz
    
    print(f"Mode frequencies: {mode_frequencies} MHz")
    print(f"Drive detunings: {drive_detunings} MHz")
    print(f"Rabi frequencies: {rabi_frequencies} kHz")
    print(f"Recoil frequency: {recoil_frequency} kHz")
    
    # Calculate coupling matrix
    coupling_matrix = Jij_from_multimode(
        normal_modes, mode_frequencies, drive_detunings,
        rabi_frequencies, recoil_frequency
    )
    
    print(f"\nCoupling matrix shape: {coupling_matrix.shape}")
    print(f"Matrix symmetry check: {np.allclose(coupling_matrix, coupling_matrix.T)}")
    print(f"Diagonal elements (should be zero): {np.diag(coupling_matrix)}")
    
    max_coupling = np.max(np.abs(coupling_matrix))
    print(f"Maximum coupling strength: {max_coupling:.6f} kHz")
    
    if max_coupling > 1e-6:
        print("✓ Non-trivial coupling strengths generated")
    
    print()


def demonstrate_performance_scaling():
    """Demonstrate computational scaling with system size."""
    print("=" * 60)
    print("Performance Scaling Analysis")
    print("=" * 60)
    
    import time
    
    system_sizes = [2, 3, 4]
    
    for n_qubits in system_sizes:
        print(f"\nTesting {n_qubits}-qubit system:")
        print(f"  Hilbert space dimension: 2^{n_qubits} = {2**n_qubits}")
        
        # Time the UMQ operation
        start_time = time.time()
        umq_gate = UMQ(n_qubits, np.pi / 6)
        umq_time = time.time() - start_time
        
        print(f"  UMQ generation time: {umq_time*1000:.2f} ms")
        print(f"  Memory usage: {umq_gate.nbytes / 1024:.1f} KB")
    
    print()


def main():
    """Run all demonstration examples."""
    print("Richerme Ion Analog Quantum Gate Synthesis")
    print("Comprehensive Demonstration Suite")
    print("" + "=" * 80 + "")
    
    # Run all demonstrations
    demonstrate_basic_synthesis()
    demonstrate_parameter_sweep()
    demonstrate_custom_pauli_strings()
    demonstrate_umq_operation()
    demonstrate_hardware_coupling()
    demonstrate_performance_scaling()
    
    print("=" * 80)
    print("All demonstrations completed successfully!")
    print("For more advanced usage, see the comprehensive test suite.")
    print("=" * 80)


if __name__ == "__main__":
    main()

