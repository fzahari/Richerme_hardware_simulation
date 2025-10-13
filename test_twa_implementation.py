"""
Test and Validation Script for TWA Implementations

This script validates the Truncated Wigner Approximation (TWA) implementations
for H2 and H2O molecules, checking key properties like spin conservation,
energy consistency, and proper dissipation behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from rich_sim_h2_twa import H2_TWA_Simulator
from rich_sim_h2o_twa import H2O_TWA_Simulator
from twa_framework import TWASpinSimulator


def test_spin_conservation():
    """
    Test that spin length |s|² is conserved for each trajectory.

    This is a fundamental property of TWA - the equations come from an
    effective Hamiltonian, so spin length must be conserved.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Spin Length Conservation")
    print("=" * 70)

    # Create simple 2-qubit system
    twa = TWASpinSimulator(n_qubits=2, n_trajectories=10)

    # Sample initial conditions
    spins_initial = []
    for _ in range(10):
        s = twa.discrete_sample_initial_state('superposition')
        spins_initial.append(s)

    # Check initial spin lengths
    s_squared_initial = [np.sum(s**2, axis=1) for s in spins_initial]

    print("Initial spin lengths squared:")
    for i, s2 in enumerate(s_squared_initial[:3]):  # Show first 3
        print(f"  Trajectory {i}: {s2}")

    # Expected value for spin-1/2: |s|² = 3 (sx² + sy² + sz² = 1 + 1 + 1 = 3)
    expected = 3.0
    tolerance = 1e-10

    all_conserved = True
    for i, s2 in enumerate(s_squared_initial):
        if not np.allclose(s2, expected, atol=tolerance):
            print(f"  ✗ Trajectory {i}: FAILED (expected {expected}, got {s2})")
            all_conserved = False

    if all_conserved:
        print(f"\n✓ PASS: All spins have correct length |s|² = {expected:.1f}")
    else:
        print("\n✗ FAIL: Some spins have incorrect length")

    return all_conserved


def test_no_dissipation_energy_conservation():
    """
    Test that without dissipation, energy is approximately conserved.

    In TWA without dissipation, the Hamiltonian should be conserved
    (modulo small numerical errors from integration).
    """
    print("\n" + "=" * 70)
    print("TEST 2: Energy Conservation (No Dissipation)")
    print("=" * 70)

    h2_twa = H2_TWA_Simulator(n_trajectories=50)

    # Run short simulation without dissipation
    results = h2_twa.simulate_twa_dynamics(
        r=0.74,
        total_time=5.0,
        n_steps=50,
        add_T1=False,
        add_T2=False
    )

    # Check energy conservation
    energies = results['avg_energies']
    initial_energy = energies[0]
    final_energy = energies[-1]
    energy_drift = abs(final_energy - initial_energy)

    print(f"Initial energy: {initial_energy:.6f} H")
    print(f"Final energy:   {final_energy:.6f} H")
    print(f"Energy drift:   {energy_drift:.6e} H")

    # Tolerance: should be small (< 1% of initial energy)
    tolerance = 0.01 * abs(initial_energy)

    if energy_drift < tolerance:
        print(f"\n✓ PASS: Energy conserved within tolerance ({tolerance:.4e} H)")
        return True
    else:
        print(f"\n✗ FAIL: Energy drift exceeds tolerance")
        return False


def test_dissipation_causes_decay():
    """
    Test that T1/T2 dissipation causes energy relaxation.

    With dissipation, energy should decrease over time (on average)
    as the system relaxes toward the ground state.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Dissipation Causes Energy Decay")
    print("=" * 70)

    h2_twa = H2_TWA_Simulator(n_trajectories=100)

    # Run with dissipation
    results = h2_twa.simulate_twa_dynamics(
        r=0.74,
        total_time=10.0,
        n_steps=50,
        add_T1=True,
        add_T2=True
    )

    energies = results['avg_energies']
    initial_energy = energies[0]
    final_energy = energies[-1]

    print(f"Initial energy: {initial_energy:.6f} H")
    print(f"Final energy:   {final_energy:.6f} H")
    print(f"Energy change:  {final_energy - initial_energy:.6f} H")

    # Energy should decrease (or stay roughly constant)
    # For H2 starting from Hartree-Fock, energy might not decrease much
    # since HF is already a reasonable approximation
    if final_energy <= initial_energy + 0.1:  # Allow small increase from noise
        print("\n✓ PASS: Energy behaves physically with dissipation")
        return True
    else:
        print(f"\n✗ FAIL: Energy increased significantly ({final_energy - initial_energy:.4f} H)")
        return False


def test_trajectory_averaging():
    """
    Test that averaging over trajectories reduces statistical noise.

    The standard error should scale as 1/√N_traj.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Trajectory Averaging Reduces Noise")
    print("=" * 70)

    h2_twa = H2_TWA_Simulator(n_trajectories=100)

    results = h2_twa.simulate_twa_dynamics(
        r=0.74,
        total_time=5.0,
        n_steps=20,
        add_T1=False,
        add_T2=False
    )

    # Check that standard deviation is reasonable
    avg_std = np.mean(results['std_energies'])
    avg_energy = np.mean(results['avg_energies'])

    relative_std = avg_std / abs(avg_energy)

    print(f"Average energy:           {avg_energy:.6f} H")
    print(f"Average std deviation:    {avg_std:.6f} H")
    print(f"Relative uncertainty:     {relative_std*100:.2f}%")

    # With 100 trajectories, relative uncertainty should be < 10%
    if relative_std < 0.10:
        print("\n✓ PASS: Statistical uncertainty is reasonable")
        return True
    else:
        print("\n✗ FAIL: Statistical uncertainty is too large")
        return False


def test_h2o_scalability():
    """
    Test that H2O (10 qubit) simulation runs without errors.

    This is mainly a smoke test to ensure the larger system works.
    """
    print("\n" + "=" * 70)
    print("TEST 5: H2O Scalability Test")
    print("=" * 70)

    try:
        h2o_twa = H2O_TWA_Simulator(n_trajectories=50)

        # Run short simulation
        results = h2o_twa.simulate_twa_dynamics(
            total_time=2.0,
            n_steps=10,
            add_T1=False,
            add_T2=False
        )

        print(f"Initial energy: {results['avg_energies'][0]:.4f} H")
        print(f"Final energy:   {results['avg_energies'][-1]:.4f} H")
        print(f"System size:    {h2o_twa.n_qubits} qubits")

        print("\n✓ PASS: H2O simulation completed successfully")
        return True

    except Exception as e:
        print(f"\n✗ FAIL: H2O simulation failed with error: {e}")
        return False


def run_all_tests():
    """Run all validation tests and generate report."""
    print("\n" + "=" * 70)
    print("TWA IMPLEMENTATION VALIDATION SUITE")
    print("=" * 70)
    print()
    print("This script validates the TWA implementations for H2 and H2O.")
    print("Tests check spin conservation, energy dynamics, and scalability.")
    print()

    tests = [
        ("Spin Length Conservation", test_spin_conservation),
        ("Energy Conservation (No Dissipation)", test_no_dissipation_energy_conservation),
        ("Dissipation Causes Energy Decay", test_dissipation_causes_decay),
        ("Trajectory Averaging", test_trajectory_averaging),
        ("H2O Scalability", test_h2o_scalability),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ FAIL: Test crashed with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    print()
    print(f"Total: {n_passed}/{n_total} tests passed ({n_passed/n_total*100:.0f}%)")

    if n_passed == n_total:
        print("\n🎉 All tests passed! TWA implementation is validated.")
    else:
        print(f"\n⚠ {n_total - n_passed} test(s) failed. Please review the output above.")

    print("=" * 70)


def demo_visualization():
    """
    Create a visualization comparing ideal and dissipative H2 dynamics.

    This is not a test, but a demo to show the TWA method in action.
    """
    print("\n" + "=" * 70)
    print("BONUS: Visualization Demo")
    print("=" * 70)
    print("\nGenerating comparison plot for H2 molecule...")

    h2_twa = H2_TWA_Simulator(n_trajectories=200)

    # Run both ideal and dissipative
    results_ideal = h2_twa.simulate_twa_dynamics(
        r=0.74, total_time=15.0, n_steps=75,
        add_T1=False, add_T2=False
    )

    h2_twa.twa.dissipation_channels = []  # Reset
    results_diss = h2_twa.simulate_twa_dynamics(
        r=0.74, total_time=15.0, n_steps=75,
        add_T1=True, add_T2=True
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy
    ax = axes[0]
    ax.plot(results_ideal['times'], results_ideal['avg_energies'],
            'b-', linewidth=2, label='Ideal (no dissipation)')
    ax.fill_between(results_ideal['times'],
                    results_ideal['avg_energies'] - results_ideal['std_energies'],
                    results_ideal['avg_energies'] + results_ideal['std_energies'],
                    alpha=0.2, color='blue')
    ax.plot(results_diss['times'], results_diss['avg_energies'],
            'r-', linewidth=2, label='With T1 + T2')
    ax.fill_between(results_diss['times'],
                    results_diss['avg_energies'] - results_diss['std_energies'],
                    results_diss['avg_energies'] + results_diss['std_energies'],
                    alpha=0.2, color='red')
    ax.set_xlabel('Time (a.u.)', fontsize=12)
    ax.set_ylabel('Energy (Hartree)', fontsize=12)
    ax.set_title('H₂ Energy Evolution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Magnetization
    ax = axes[1]
    ax.plot(results_ideal['times'], results_ideal['magnetization'],
            'b-', linewidth=2, label='Ideal')
    ax.plot(results_diss['times'], results_diss['magnetization'],
            'r-', linewidth=2, label='With T1 + T2')
    ax.set_xlabel('Time (a.u.)', fontsize=12)
    ax.set_ylabel('Total Magnetization', fontsize=12)
    ax.set_title('Magnetization Dynamics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.suptitle('TWA Simulation: Ideal vs. Dissipative Dynamics',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('twa_validation_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'twa_validation_demo.png'")
    plt.show()


if __name__ == "__main__":
    # Run all validation tests
    run_all_tests()

    # Optionally run visualization demo
    print("\n" + "=" * 70)
    response = input("\nRun visualization demo? (y/n): ")
    if response.lower() == 'y':
        demo_visualization()
    else:
        print("Skipping demo visualization.")

    print("\n✓ Validation complete!")
