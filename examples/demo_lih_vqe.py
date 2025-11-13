#!/usr/bin/env python3
"""
Quick Start Demo: LiH VQE Simulation

This script demonstrates the basic usage of the LiH VQE simulator,
including:
1. Basic VQE optimization
2. Potential energy surface scanning
3. Dissipative dynamics with TWA
4. GPU acceleration (if available)

Run with: python examples/demo_lih_vqe.py
"""

import numpy as np
import matplotlib.pyplot as plt
from rich_sim_lih import TrappedIonSimulator, LiHSimulator

print("=" * 70)
print("LiH VQE QUICK START DEMO")
print("=" * 70)
print()

# ============================================================================
# PART 1: Basic VQE at Equilibrium Geometry
# ============================================================================

print("PART 1: Basic VQE Optimization")
print("-" * 70)

# Create trapped-ion system
ion_system = TrappedIonSimulator(N=12, geometry='1D', anharmonic=False)

# Create LiH simulator (4 qubits, minimal active space)
lih_sim = LiHSimulator(
    ion_system,
    use_hardware_gates=True,  # Use hardware-realistic gates
    n_qubits=4
)

# Run VQE at equilibrium bond length
r_eq = 1.5949  # Angstroms (LiH equilibrium)

print(f"\nRunning VQE at R = {r_eq:.4f} Å")
print("This may take 5-10 minutes...")

vqe_result = lih_sim.vqe_optimization(
    r=r_eq,
    n_layers=4,         # 4 layers of gates
    max_iter=800,       # 800 iterations max
    method='COBYLA',    # COBYLA optimizer
    n_trials=3          # 3 random initializations
)

print("\n" + "=" * 70)
print("VQE RESULTS")
print("=" * 70)
print(f"VQE energy:         {vqe_result['vqe_energy']:.8f} Hartree")
print(f"Exact energy:       {vqe_result['exact_energy']:.8f} Hartree")
print(f"Absolute error:     {vqe_result['error']:.6e} Hartree")
print(f"Ground state overlap: {vqe_result['overlap']:.6f}")
print(f"Iterations:         {vqe_result['n_iterations']}")

# ============================================================================
# PART 2: Potential Energy Surface Scan
# ============================================================================

print("\n\nPART 2: Potential Energy Surface Scan")
print("-" * 70)

# Define bond lengths to scan
r_values = np.linspace(1.2, 2.4, 7)  # 7 points from 1.2 to 2.4 Å

print(f"\nScanning {len(r_values)} bond lengths...")
print("This may take 15-20 minutes...")

scan_results = lih_sim.scan_bond_length(
    r_values,
    n_layers=4,
    max_iter=500  # Fewer iterations per point
)

# ============================================================================
# PART 3: Visualize Results
# ============================================================================

print("\n\nPART 3: Creating Visualizations")
print("-" * 70)

fig = plt.figure(figsize=(16, 10))

# Plot 1: VQE convergence (from Part 1)
ax1 = plt.subplot(2, 3, 1)
energy_history = vqe_result['energy_history']
exact_energy = vqe_result['exact_energy']

ax1.plot(energy_history, 'b-', linewidth=2, label='VQE')
ax1.axhline(y=exact_energy, color='r', linestyle='--', linewidth=2, label='Exact')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Energy (Hartree)')
ax1.set_title(f'VQE Convergence (R = {r_eq:.4f} Å)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: VQE error (log scale)
ax2 = plt.subplot(2, 3, 2)
errors = np.abs(np.array(energy_history) - exact_energy)
ax2.semilogy(errors, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('|E - E₀| (Hartree)')
ax2.set_title('VQE Error Convergence')
ax2.grid(True, alpha=0.3)

# Plot 3: Final energy comparison
ax3 = plt.subplot(2, 3, 3)
methods = ['Exact', 'VQE']
energies = [exact_energy, vqe_result['vqe_energy']]
colors = ['red', 'blue']
bars = ax3.bar(methods, energies, color=colors, alpha=0.7)
ax3.set_ylabel('Energy (Hartree)')
ax3.set_title('Ground State Energy')
ax3.grid(True, alpha=0.3, axis='y')

for bar, energy in zip(bars, energies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{energy:.6f}',
            ha='center', va='bottom', fontsize=9)

# Plot 4: Potential energy surface
ax4 = plt.subplot(2, 3, 4)
ax4.plot(scan_results['bond_lengths'], scan_results['exact_energies'],
         'r-', linewidth=2, marker='o', markersize=8, label='Exact')
ax4.plot(scan_results['bond_lengths'], scan_results['vqe_energies'],
         'b--', linewidth=2, marker='s', markersize=8, label='VQE')
ax4.axvline(x=r_eq, color='gray', linestyle=':', alpha=0.5, label='Equilibrium')
ax4.set_xlabel('Li-H Bond Length (Å)')
ax4.set_ylabel('Energy (Hartree)')
ax4.set_title('LiH Potential Energy Surface')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: VQE error vs bond length
ax5 = plt.subplot(2, 3, 5)
ax5.semilogy(scan_results['bond_lengths'], np.abs(scan_results['errors']),
             'b-', linewidth=2, marker='o', markersize=8)
ax5.set_xlabel('Li-H Bond Length (Å)')
ax5.set_ylabel('|VQE Error| (Hartree)')
ax5.set_title('VQE Accuracy vs Bond Length')
ax5.grid(True, alpha=0.3)

# Plot 6: Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = f"""
LiH VQE SUMMARY
{'=' * 40}

System Properties:
  • Qubits: {lih_sim.n_qubits}
  • Ansatz layers: 4
  • Parameters: 44
  • Hardware gates: {'Yes' if lih_sim.use_hardware_gates else 'No'}

VQE Results (R = {r_eq:.4f} Å):
  • VQE energy: {vqe_result['vqe_energy']:.6f} H
  • Exact energy: {vqe_result['exact_energy']:.6f} H
  • Error: {abs(vqe_result['error']):.2e} H
  • Overlap: {vqe_result['overlap']:.4f}

PES Scan:
  • Points: {len(r_values)}
  • Range: {r_values[0]:.2f} - {r_values[-1]:.2f} Å
  • Avg error: {np.mean(np.abs(scan_results['errors'])):.2e} H
  • Max error: {np.max(np.abs(scan_results['errors'])):.2e} H

Performance:
  • Iterations: {vqe_result['n_iterations']}
  • Trials: 3
"""

ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
         verticalalignment='center', transform=ax6.transAxes)

plt.suptitle('LiH VQE Simulation Results', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = 'lih_vqe_demo.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_file}")

plt.show()

# ============================================================================
# PART 4: (Optional) TWA Dissipative Dynamics
# ============================================================================

print("\n\nPART 4: TWA Dissipative Dynamics (Optional)")
print("-" * 70)
print("This demonstrates realistic decoherence effects.")
print("Skip this section if you want faster execution.")

user_input = input("\nRun TWA simulation? (y/n): ").lower()

if user_input == 'y':
    try:
        from rich_sim_lih_twa import LiH_TWA_Simulator, plot_comparison_results

        print("\nInitializing TWA simulator...")
        lih_twa = LiH_TWA_Simulator(n_trajectories=300)  # Fewer for demo

        print("Running TWA comparison (this may take 5-10 minutes)...")
        twa_results = lih_twa.compare_with_ideal(
            r=r_eq,
            total_time=10.0,  # Shorter for demo
            dt=0.01
        )

        # Summary
        print("\n" + "=" * 70)
        print("TWA RESULTS")
        print("=" * 70)
        print(f"\nEnergy shifts from dissipation:")
        ideal_E = twa_results['ideal']['avg_energies'][-1]
        print(f"  T1 effect:  {twa_results['t1_only']['avg_energies'][-1] - ideal_E:.6e} H")
        print(f"  T2 effect:  {twa_results['t2_only']['avg_energies'][-1] - ideal_E:.6e} H")
        print(f"  Total:      {twa_results['full']['avg_energies'][-1] - ideal_E:.6e} H")

        # Plot TWA results
        fig_twa = plot_comparison_results(twa_results)
        fig_twa.savefig('lih_twa_demo.png', dpi=300, bbox_inches='tight')
        print(f"\nTWA plot saved to: lih_twa_demo.png")
        plt.show()

    except ImportError as e:
        print(f"\nWarning: TWA framework not available ({e})")
        print("Install with: pip install -r requirements.txt")

else:
    print("\nSkipping TWA simulation.")

# ============================================================================
# PART 5: (Optional) GPU Acceleration
# ============================================================================

print("\n\nPART 5: GPU Acceleration (Optional)")
print("-" * 70)
print("This demonstrates 10-100x speedup using CuPy.")

user_input = input("\nRun GPU simulation? (requires CUDA and CuPy) (y/n): ").lower()

if user_input == 'y':
    try:
        from cudaq_rich_sim_lih_twa import CUDAQ_LiH_TWA_Simulator
        import time

        print("\nInitializing GPU simulator...")
        lih_gpu = CUDAQ_LiH_TWA_Simulator(n_trajectories=1000, use_gpu=True)

        print("Running GPU-accelerated simulation...")
        start = time.time()
        gpu_results = lih_gpu.simulate_dynamics_gpu(
            r=r_eq,
            total_time=10.0,
            dt=0.01,
            add_t1=True,
            add_t2=True
        )
        elapsed = time.time() - start

        print("\n" + "=" * 70)
        print("GPU PERFORMANCE")
        print("=" * 70)
        print(f"Computation time: {elapsed:.2f} seconds")
        print(f"Trajectories: {lih_gpu.n_trajectories}")
        print(f"Time steps: {len(gpu_results['times'])}")
        print(f"Throughput: {lih_gpu.n_trajectories * len(gpu_results['times']) / elapsed:.0f} traj-steps/sec")

    except ImportError:
        print("\nWarning: CuPy not available.")
        print("Install with: pip install cupy-cuda12x (or cupy-cuda11x)")
    except Exception as e:
        print(f"\nError running GPU simulation: {e}")

else:
    print("\nSkipping GPU simulation.")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 70)
print("DEMO COMPLETED!")
print("=" * 70)
print("\nFiles created:")
print("  • lih_vqe_demo.png - VQE results and PES")
if user_input == 'y':
    print("  • lih_twa_demo.png - TWA dissipative dynamics")
print("\nFor more examples, see:")
print("  • docs/LIH_IMPLEMENTATION.md - Comprehensive documentation")
print("  • tests/test_lih_simulator.py - Test suite")
print("=" * 70)
