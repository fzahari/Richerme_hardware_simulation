"""
Demonstration of Extended Features
===================================
Shows the extended capabilities of the richerme_ion_analog library
"""
import sys
from pathlib import Path
# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from richerme_ion_analog import *

def demo_arbitrary_pauli_strings():
    """
    ENHANCEMENT 1: Demonstrate arbitrary Pauli string synthesis
    """
    print("="*70)
    print("DEMO 1: Arbitrary Pauli String Synthesis")
    print("="*70)

    test_cases = [
        (['Z', 'Y', 'Z'], 0.5, "ZYZ string (impossible with original)"),
        (['Y', 'Y', 'X'], 0.3, "YYX string"),
        (['Z', 'X', 'Y'], 0.4, "ZXY string"),
        (['X', 'X', 'X'], 0.2, "XXX string (works with both)"),
    ]

    for axes, t, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Pauli string: {'⊗'.join(axes)}, t={t}")

        # Generate using new arbitrary method
        U_new = n_body_string_arbitrary(axes, t)

        # Generate target
        U_target = target_pauli_string_unitary(''.join(axes), t)

        # Compute fidelity
        distance = unitary_distance(U_target, U_new)
        print(f"  Fidelity error: {distance:.2e}")

        if distance < 1e-12:
            print(f"  ✓ Perfect synthesis!")
        else:
            print(f"  ✗ Error too large")

def demo_accessibility_checker():
    """
    ENHANCEMENT 2: Demonstrate accessibility checking
    """
    print("\n" + "="*70)
    print("DEMO 2: Accessibility Checker")
    print("="*70)

    N = 5
    # Use sinusoidal modes (for equispaced ions)
    B = compute_sinusoidal_modes(N)

    print(f"\nTesting {N}-ion system with sinusoidal modes")
    print(f"Mode matrix B shape: {B.shape}")

    # Test 1: All-to-all interaction (should be accessible)
    print("\nTest 1: All-to-all interaction")
    J_all_to_all = np.ones((N, N)) - np.eye(N)
    accessible = is_accessible(J_all_to_all, B)
    print(f"  Matrix: {N}×{N} all-to-all")
    print(f"  Accessible: {accessible}")
    if accessible:
        weights = get_mode_weights_if_accessible(J_all_to_all, B)
        print(f"  Required mode weights: {weights}")

    # Test 2: Nearest-neighbor ring (check accessibility)
    print("\nTest 2: Nearest-neighbor ring")
    J_nn = np.zeros((N, N))
    for i in range(N):
        j = (i + 1) % N
        J_nn[i, j] = J_nn[j, i] = 1.0
    # Make Laplacian form
    for i in range(N):
        J_nn[i, i] = -np.sum(J_nn[i, :])

    accessible = is_accessible(J_nn, B)
    print(f"  Matrix: {N}-site ring")
    print(f"  Accessible: {accessible}")

    # Test 3: Random interaction (likely not accessible)
    print("\nTest 3: Random interaction matrix")
    np.random.seed(42)
    J_random = np.random.randn(N, N)
    J_random = 0.5 * (J_random + J_random.T)  # Symmetrize
    np.fill_diagonal(J_random, 0)  # Zero diagonal

    accessible = is_accessible(J_random, B)
    print(f"  Matrix: {N}×{N} random symmetric")
    print(f"  Accessible: {accessible}")

def demo_mode_weight_optimization():
    """
    ENHANCEMENT 3: Demonstrate mode weight optimization
    """
    print("\n" + "="*70)
    print("DEMO 3: Mode Weight Optimization")
    print("="*70)

    N = 7
    B = compute_sinusoidal_modes(N)

    # Try to approximate power-law interactions
    print(f"\nOptimizing for power-law interaction (α=1.0) with {N} ions")

    # Build desired power-law matrix
    J_desired = np.zeros((N, N))
    alpha = 1.0
    for i in range(N):
        for j in range(i+1, N):
            J_desired[i, j] = J_desired[j, i] = 1.0 / abs(i - j)**alpha
    # Laplacian form
    for i in range(N):
        J_desired[i, i] = -np.sum(J_desired[i, :])

    # Optimize
    result = optimize_mode_weights(J_desired, B, method='least_squares')

    print(f"\nResults:")
    print(f"  Accessible (exact): {result['accessible']}")
    print(f"  Infidelity: {result['infidelity']:.6f}")
    print(f"  Mode weights: {result['weights']}")

    # Show comparison
    print(f"\nComparison (first row, off-diagonal):")
    print(f"  Desired:  {J_desired[0, 1:4]}")
    print(f"  Achieved: {result['J_achieved'][0, 1:4]}")

def demo_anharmonic_potentials():
    """
    ENHANCEMENT 4: Demonstrate anharmonic potential calculations
    """
    print("\n" + "="*70)
    print("DEMO 4: Anharmonic Potentials (Equispaced Ions)")
    print("="*70)

    N = 10
    print(f"\nComputing potential for {N} equispaced ions")

    # Get potential coefficients
    potential = compute_equispaced_potential(N, spacing=5.0, omega_z_scale=2*np.pi*0.1e6)

    print(f"\nPotential coefficients (β_n):")
    for n, beta in potential['beta_coefficients'].items():
        print(f"  β_{n} = {beta:.4f}")

    # Compute sinusoidal modes
    B = compute_sinusoidal_modes(N)

    print(f"\nMode structure comparison:")
    print(f"  COM mode (k=0): {B[:, 0][:5]}... (should be uniform)")
    print(f"  Zig-zag mode (k={N-1}): {B[:, N-1][:5]}... (should alternate)")

    # Check mode orthonormality
    gram = B.T @ B
    identity_error = np.linalg.norm(gram - np.eye(N))
    print(f"\nMode orthonormality check:")
    print(f"  ||B^T·B - I|| = {identity_error:.2e}")

    # Test nearest-neighbor with sinusoidal modes
    print(f"\nNearest-neighbor interaction with sinusoidal modes:")
    J_nn = np.zeros((N, N))
    for i in range(N-1):
        J_nn[i, i+1] = J_nn[i+1, i] = 1.0

    result = optimize_mode_weights(J_nn, B)
    print(f"  Infidelity: {result['infidelity']:.6f}")
    print(f"  Note: Infidelity → 0 as N → ∞ (Kyprianidis 2024, Fig 9a)")

def demo_hardware_parameters():
    """
    ENHANCEMENT 5: Hardware-specific parameters
    """
    print("\n" + "="*70)
    print("DEMO 5: Hardware Parameters (171Yb+)")
    print("="*70)

    hw = IonTrapHardware()

    print(f"\n171Yb+ Specifications:")
    print(f"  Hyperfine splitting: {hw.hyperfine_splitting/1e9:.1f} GHz")
    print(f"  T2 coherence time: {hw.T2_coherence:.1f} s")
    print(f"  T1 coherence time: {'∞' if np.isinf(hw.T1_coherence) else hw.T1_coherence}")
    print(f"  Single-qubit fidelity: {hw.single_qubit_fidelity*100:.1f}%")
    print(f"  Two-qubit fidelity: {hw.two_qubit_fidelity*100:.1f}%")

    print(f"\nTrap frequencies:")
    print(f"  ωz (axial): {hw.omega_z/(2*np.pi*1e6):.1f} MHz")
    print(f"  ωx (radial): {hw.omega_x/(2*np.pi*1e6):.1f} MHz")
    print(f"  ωy (radial): {hw.omega_y/(2*np.pi*1e6):.1f} MHz")

    print(f"\nLaser parameters:")
    print(f"  Wavelength: {hw.wavelength*1e9:.1f} nm")
    print(f"  Recoil frequency: {hw.recoil_frequency()/1e3:.2f} kHz")

    # Demonstrate Jij calculation with realistic parameters
    print(f"\nRealistic Jij calculation:")
    N = 5
    B = compute_sinusoidal_modes(N)
    omega = hw.omega_x + np.linspace(0, 0.5e6, N) * 2*np.pi  # Spread over 500 kHz

    # Single tone, slightly detuned from COM
    mus = np.array([omega[0] + 2*np.pi*100e3])  # 100 kHz red detuning
    Omegas = np.array([2*np.pi*50e3])  # 50 kHz Rabi frequency

    R_recoil = hw.recoil_frequency()

    J = Jij_from_multimode(B, omega, mus, Omegas, R_recoil)

    print(f"  {N} ions, single tone at μ = ωCOM - 100 kHz")
    print(f"  Coupling matrix J (first row):")
    print(f"    {J[0, :]}")

def demo_library_features():
    """
    Show the complete capabilities of the unified library
    """
    print("\n" + "="*70)
    print("UNIFIED LIBRARY FEATURES")
    print("="*70)

    print("\nCore Gate Synthesis:")
    print("  ✓ n_body_string(): ZXX, YXX, XXX patterns")
    print("  ✓ n_body_string_arbitrary(): Any Pauli pattern (ZYZ, XYX, etc.)")
    print("  ✓ UMQ-Rz-UMQ construction pattern")

    print("\nInteraction Engineering:")
    print("  ✓ is_accessible(): Check if graph is realizable")
    print("  ✓ optimize_mode_weights(): Find best approximation")
    print("  ✓ compute_sinusoidal_modes(): Modes for equispaced ions")

    print("\nHardware Specifications:")
    print("  ✓ IonTrapHardware: 171Yb+ parameters")
    print("  ✓ Jij_from_multimode(): Realistic coupling calculations")
    print("  ✓ compute_equispaced_potential(): Anharmonic traps")

    print("\nKey Research Formulas:")
    print("  • Equation 14 (Kyprianidis): B^T·J·B diagonal ⟺ accessible")
    print("  • Equation 4 (Richerme): Jij = Σ_k Σ_m (Ω²R)/(μ²-ω²) B_ik B_jk")
    print("  • Equation 18 (Kyprianidis): Sinusoidal modes for equispaced ions")

if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "RICHERME ION ANALOG: EXTENDED FEATURES" + " "*15 + "║")
    print("╚" + "="*68 + "╝")

    demo_arbitrary_pauli_strings()
    demo_accessibility_checker()
    demo_mode_weight_optimization()
    demo_anharmonic_potentials()
    demo_hardware_parameters()
    demo_library_features()

    print("\n" + "="*70)
    print("All demonstrations completed!")
    print("="*70)
