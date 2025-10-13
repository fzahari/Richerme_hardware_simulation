"""
CUDA-Q Accelerated H2 Molecule Simulation with TWA

GPU-accelerated version of H2 TWA simulation using CUDA-Q and CuPy.
All trajectories are evolved in parallel on the GPU for massive speedup.

Based on: "User-Friendly Truncated Wigner Approximation for Dissipative Spin Dynamics"
Hosseinabadi, Chelpanova, and Marino, PRX Quantum 6, 030344 (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from cudaq_twa_framework import CUDAQTWASpinSimulator, IonTrapDissipationRates

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False


class CUDAQ_H2_TWA_Simulator:
    """
    GPU-accelerated H2 molecule simulator with TWA.

    Speedup over CPU version: ~10-100x depending on trajectory count
    Recommended: n_trajectories >= 1000 for best GPU utilization
    """

    def __init__(self, n_trajectories: int = 1000, use_gpu: bool = True):
        """
        Initialize CUDA-Q H2 TWA simulator.

        Args:
            n_trajectories: Number of stochastic trajectories
                           (increase for better GPU utilization)
            use_gpu: Use GPU acceleration if available
        """
        self.n_qubits = 4
        self.n_trajectories = n_trajectories
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        # Initialize CUDA-Q TWA simulator
        self.twa = CUDAQTWASpinSimulator(self.n_qubits, n_trajectories, use_gpu=use_gpu)
        self.xp = self.twa.xp  # cp or np

        # Hardware parameters
        self.hardware = IonTrapDissipationRates()

        print(f"\n{'='*70}")
        print(f"CUDA-Q H2 TWA Simulator Initialized")
        print(f"{'='*70}")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  Backend: {'GPU (CuPy)' if self.use_gpu else 'CPU (NumPy)'}")
        print(f"  T1 = {self.hardware.T1_SI} s")
        print(f"  T2 = {self.hardware.T2_SI} s")
        print(f"{'='*70}")

    def build_h2_classical_hamiltonian_vectorized(self, r: float, s_all: cp.ndarray) -> cp.ndarray:
        """
        Build H2 Hamiltonian for ALL trajectories simultaneously (vectorized).

        Args:
            r: Bond distance in Angstroms
            s_all: Spin configurations, shape (n_trajectories, 4, 3)

        Returns:
            H_all: Hamiltonian values, shape (n_trajectories,)
        """
        xp = self.xp

        # Empirical parameters
        e_nuc = 1.0 / r
        t = 0.52917 * xp.exp(-1.5 * (r - 0.741))
        mu = -1.1256 + 0.2 * r
        u = 0.6744 / (1 + 0.1 * r)
        v = 0.1815 * xp.exp(-0.5 * r)

        # Initialize Hamiltonian (broadcast scalar to all trajectories)
        H_all = xp.full(s_all.shape[0], e_nuc)

        # Single-qubit Z terms (vectorized)
        H_all += (mu / 2) * s_all[:, 0, 2]  # Z on qubit 0
        H_all += (mu / 2) * s_all[:, 1, 2]  # Z on qubit 1
        H_all += (mu / 4) * s_all[:, 2, 2]  # Z on qubit 2
        H_all += (mu / 4) * s_all[:, 3, 2]  # Z on qubit 3

        # Hopping terms: X_i X_j and Y_i Y_j (vectorized)
        H_all += (-t / 2) * s_all[:, 0, 0] * s_all[:, 2, 0]  # X0 X2
        H_all += (-t / 2) * s_all[:, 0, 1] * s_all[:, 2, 1]  # Y0 Y2
        H_all += (-t / 2) * s_all[:, 1, 0] * s_all[:, 3, 0]  # X1 X3
        H_all += (-t / 2) * s_all[:, 1, 1] * s_all[:, 3, 1]  # Y1 Y3

        # Two-qubit ZZ interactions (vectorized)
        H_all += (u / 4) * s_all[:, 0, 2] * s_all[:, 1, 2]   # Z0 Z1
        H_all += (u / 4) * s_all[:, 2, 2] * s_all[:, 3, 2]   # Z2 Z3
        H_all += (v / 4) * s_all[:, 0, 2] * s_all[:, 2, 2]   # Z0 Z2
        H_all += (v / 4) * s_all[:, 1, 2] * s_all[:, 3, 2]   # Z1 Z3
        H_all += (v / 8) * s_all[:, 0, 2] * s_all[:, 3, 2]   # Z0 Z3
        H_all += (v / 8) * s_all[:, 1, 2] * s_all[:, 2, 2]   # Z1 Z2

        return H_all

    def hamiltonian_gradient_vectorized(self, s_all: cp.ndarray, r: float) -> cp.ndarray:
        """
        Compute gradient ∂H/∂s for ALL trajectories simultaneously (vectorized).

        Args:
            s_all: Spin configurations (n_trajectories, 4, 3)
            r: Bond distance

        Returns:
            grad_H_all: Gradients, shape (n_trajectories, 4, 3)
        """
        xp = self.xp
        n_traj = s_all.shape[0]

        # Parameters
        t = 0.52917 * xp.exp(-1.5 * (r - 0.741))
        mu = -1.1256 + 0.2 * r
        u = 0.6744 / (1 + 0.1 * r)
        v = 0.1815 * xp.exp(-0.5 * r)

        grad_all = xp.zeros((n_traj, 4, 3))

        # ∂H/∂s^x (vectorized across trajectories)
        grad_all[:, 0, 0] = (-t / 2) * s_all[:, 2, 0]  # from X0 X2
        grad_all[:, 1, 0] = (-t / 2) * s_all[:, 3, 0]  # from X1 X3
        grad_all[:, 2, 0] = (-t / 2) * s_all[:, 0, 0]  # from X0 X2
        grad_all[:, 3, 0] = (-t / 2) * s_all[:, 1, 0]  # from X1 X3

        # ∂H/∂s^y (vectorized)
        grad_all[:, 0, 1] = (-t / 2) * s_all[:, 2, 1]  # from Y0 Y2
        grad_all[:, 1, 1] = (-t / 2) * s_all[:, 3, 1]  # from Y1 Y3
        grad_all[:, 2, 1] = (-t / 2) * s_all[:, 0, 1]  # from Y0 Y2
        grad_all[:, 3, 1] = (-t / 2) * s_all[:, 1, 1]  # from Y1 Y3

        # ∂H/∂s^z (all Z terms, vectorized)
        grad_all[:, 0, 2] = (mu / 2) + (u / 4) * s_all[:, 1, 2] + (v / 4) * s_all[:, 2, 2] + (v / 8) * s_all[:, 3, 2]
        grad_all[:, 1, 2] = (mu / 2) + (u / 4) * s_all[:, 0, 2] + (v / 4) * s_all[:, 3, 2] + (v / 8) * s_all[:, 2, 2]
        grad_all[:, 2, 2] = (mu / 4) + (u / 4) * s_all[:, 3, 2] + (v / 4) * s_all[:, 0, 2] + (v / 8) * s_all[:, 1, 2]
        grad_all[:, 3, 2] = (mu / 4) + (u / 4) * s_all[:, 2, 2] + (v / 4) * s_all[:, 1, 2] + (v / 8) * s_all[:, 0, 2]

        return grad_all

    def simulate_twa_dynamics(self, r: float, total_time: float, n_steps: int = 100,
                             add_T1: bool = True, add_T2: bool = True) -> Dict:
        """
        Simulate H2 dynamics with GPU-accelerated TWA.

        Args:
            r: Bond distance in Angstroms
            total_time: Total simulation time (a.u.)
            n_steps: Number of time steps
            add_T1: Include T1 energy relaxation
            add_T2: Include T2 dephasing

        Returns:
            Dictionary with simulation results (on CPU for plotting)
        """
        xp = self.xp
        dt = total_time / n_steps
        times = xp.linspace(0, total_time, n_steps)

        print(f"\nRunning CUDA-Q TWA simulation:")
        print(f"  Bond distance: R = {r:.3f} Å")
        print(f"  Time steps: {n_steps}, dt = {dt:.4f}")
        print(f"  Trajectories: {self.n_trajectories} ({'GPU' if self.use_gpu else 'CPU'})")
        print(f"  T1 decay: {'ON' if add_T1 else 'OFF'}")
        print(f"  T2 dephasing: {'ON' if add_T2 else 'OFF'}")

        # Set up dissipation channels
        self.twa.dissipation_channels = []  # Reset
        if add_T1:
            self.twa.add_dissipation('decay', self.hardware.gamma_decay, list(range(self.n_qubits)))
        if add_T2:
            self.twa.add_dissipation('dephasing', self.hardware.kappa_dephasing, list(range(self.n_qubits)))

        # Initialize ALL trajectories at once on GPU
        # Hartree-Fock: |0011⟩ - qubits 0,1 in |↓⟩, qubits 2,3 in |↑⟩
        s_all = xp.zeros((self.n_trajectories, self.n_qubits, 3))
        sx = xp.random.choice([-1, 1], size=(self.n_trajectories, self.n_qubits))
        sy = xp.random.choice([-1, 1], size=(self.n_trajectories, self.n_qubits))
        sz = xp.ones((self.n_trajectories, self.n_qubits))
        sz[:, 0] = -1  # Qubit 0: |↓⟩
        sz[:, 1] = -1  # Qubit 1: |↓⟩
        # Qubits 2,3 stay at +1: |↑⟩

        s_all[:, :, 0] = sx
        s_all[:, :, 1] = sy
        s_all[:, :, 2] = sz

        # Storage for trajectory evolution
        all_spins = xp.zeros((n_steps, self.n_trajectories, self.n_qubits, 3))
        all_energies = xp.zeros((n_steps, self.n_trajectories))

        # Time evolution (all trajectories evolve in parallel on GPU)
        import time
        start_time = time.time()

        for step in range(n_steps):
            # Store current state
            all_spins[step] = s_all
            E_all = self.build_h2_classical_hamiltonian_vectorized(r, s_all)
            all_energies[step] = E_all

            # Generate noise for ALL trajectories
            noise = self.twa.generate_noise_vectorized(dt)

            # RK4 integration (GPU-parallelized across trajectories)
            s_all = self.twa.rk4_step_vectorized(
                float(times[step]), s_all, dt,
                self.hamiltonian_gradient_vectorized,
                noise, r
            )

            if (step + 1) % (n_steps // 10) == 0:
                print(f"  Progress: {step+1}/{n_steps} steps ({100*(step+1)/n_steps:.0f}%)")

        elapsed = time.time() - start_time
        print(f"✓ Simulation complete in {elapsed:.2f} seconds")
        print(f"  ({elapsed/n_steps:.4f} s/step, {elapsed/self.n_trajectories:.4f} s/trajectory)")

        # Transfer to CPU for analysis
        if self.use_gpu:
            all_spins = cp.asnumpy(all_spins)
            all_energies = cp.asnumpy(all_energies)
            times = cp.asnumpy(times)

        # Compute expectation values (transpose to match original format)
        all_spins = np.transpose(all_spins, (1, 0, 2, 3))  # (n_traj, n_steps, n_qubits, 3)
        all_energies = np.transpose(all_energies)  # (n_traj, n_steps)

        avg_spins = np.mean(all_spins, axis=0)
        avg_energies = np.mean(all_energies, axis=0)
        std_energies = np.std(all_energies, axis=0)
        magnetization = np.mean(np.sum(all_spins[:, :, :, 2], axis=2), axis=0)

        return {
            'times': times,
            'avg_spins': avg_spins,
            'avg_energies': avg_energies,
            'std_energies': std_energies,
            'magnetization': magnetization,
            'all_spins': all_spins,
            'all_energies': all_energies,
        }

    def compare_with_ideal(self, r: float = 0.74, total_time: float = 10.0, n_steps: int = 100):
        """
        Compare TWA dissipative dynamics with ideal case (GPU-accelerated).

        Args:
            r: Bond distance
            total_time: Simulation time
            n_steps: Number of time steps
        """
        print("\n" + "=" * 70)
        print("COMPARING IDEAL VS. DISSIPATIVE DYNAMICS (GPU-ACCELERATED)")
        print("=" * 70)

        # Run ideal (no dissipation)
        print("\n[1] Running ideal dynamics (no T1/T2)...")
        results_ideal = self.simulate_twa_dynamics(r, total_time, n_steps=n_steps,
                                                   add_T1=False, add_T2=False)

        # Run with only T2 dephasing
        print("\n[2] Running with T2 dephasing only...")
        results_T2 = self.simulate_twa_dynamics(r, total_time, n_steps=n_steps,
                                                add_T1=False, add_T2=True)

        # Run with T1 and T2
        print("\n[3] Running with T1 decay + T2 dephasing...")
        results_full = self.simulate_twa_dynamics(r, total_time, n_steps=n_steps,
                                                  add_T1=True, add_T2=True)

        # Visualization (same as CPU version)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Energy evolution
        ax = axes[0, 0]
        ax.plot(results_ideal['times'], results_ideal['avg_energies'],
                'b-', linewidth=2, label='Ideal (no dissipation)')
        ax.plot(results_T2['times'], results_T2['avg_energies'],
                'g--', linewidth=2, label='T2 dephasing only')
        ax.plot(results_full['times'], results_full['avg_energies'],
                'r:', linewidth=2, label='T1 + T2')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title(f'H₂ Energy Evolution (R = {r:.2f} Å) [GPU-Accelerated]')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Magnetization
        ax = axes[0, 1]
        ax.plot(results_ideal['times'], results_ideal['magnetization'],
                'b-', linewidth=2, label='Ideal')
        ax.plot(results_T2['times'], results_T2['magnetization'],
                'g--', linewidth=2, label='T2 only')
        ax.plot(results_full['times'], results_full['magnetization'],
                'r:', linewidth=2, label='T1 + T2')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Total Magnetization ⟨∑ᵢ σᶻᵢ⟩')
        ax.set_title('Magnetization Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Energy uncertainty
        ax = axes[1, 0]
        ax.fill_between(results_ideal['times'],
                        results_ideal['avg_energies'] - results_ideal['std_energies'],
                        results_ideal['avg_energies'] + results_ideal['std_energies'],
                        alpha=0.3, color='blue', label='Ideal')
        ax.fill_between(results_full['times'],
                        results_full['avg_energies'] - results_full['std_energies'],
                        results_full['avg_energies'] + results_full['std_energies'],
                        alpha=0.3, color='red', label='T1 + T2')
        ax.plot(results_ideal['times'], results_ideal['avg_energies'], 'b-', linewidth=2)
        ax.plot(results_full['times'], results_full['avg_energies'], 'r-', linewidth=2)
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title('Energy with Statistical Uncertainty')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Individual spin trajectories (qubit 0)
        ax = axes[1, 1]
        n_show = min(20, self.n_trajectories)  # Show up to 20 trajectories
        for traj in range(n_show):
            ax.plot(results_full['times'],
                   results_full['all_spins'][traj, :, 0, 2],
                   alpha=0.3, linewidth=0.5, color='gray')
        ax.plot(results_full['times'], results_full['avg_spins'][:, 0, 2],
                'r-', linewidth=3, label='Average ⟨σᶻ₀⟩')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('σᶻ₀')
        ax.set_title(f'Sample Trajectories (Qubit 0, showing {n_show})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle(f'H₂ Molecule: GPU-Accelerated TWA ({self.n_trajectories} trajectories)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        print(f"\nFinal energies:")
        print(f"  Ideal:      {results_ideal['avg_energies'][-1]:.6f} ± {results_ideal['std_energies'][-1]:.6f} H")
        print(f"  T2 only:    {results_T2['avg_energies'][-1]:.6f} ± {results_T2['std_energies'][-1]:.6f} H")
        print(f"  T1 + T2:    {results_full['avg_energies'][-1]:.6f} ± {results_full['std_energies'][-1]:.6f} H")

        return {
            'ideal': results_ideal,
            'T2_only': results_T2,
            'full': results_full
        }


if __name__ == "__main__":
    print("=" * 70)
    print("CUDA-Q H₂ MOLECULE SIMULATION WITH TWA")
    print("=" * 70)
    print()
    print("GPU-accelerated Truncated Wigner Approximation (TWA)")
    print("for dissipative quantum dynamics of the H₂ molecule.")
    print()
    print("Features:")
    print("  - 4-qubit quantum chemistry Hamiltonian")
    print("  - All trajectories evolved in parallel on GPU")
    print("  - 10-100x speedup over CPU version")
    print("  - Hardware-realistic T1/T2 decoherence (171Yb+)")
    print()
    print("=" * 70)
    print()

    # Create GPU-accelerated simulator
    # Use more trajectories than CPU version for better GPU utilization
    h2_twa = CUDAQ_H2_TWA_Simulator(n_trajectories=2000, use_gpu=True)

    # Run comparison
    results = h2_twa.compare_with_ideal(r=0.74, total_time=20.0, n_steps=100)

    print("\n✓ GPU-accelerated simulation complete! Check the plots.")
