"""
H2 Molecule Simulation with Truncated Wigner Approximation (TWA)

This module extends the original H2 simulation (rich_sim_h2.py) with dissipative
dynamics using the TWA method. It models realistic decoherence effects from:
- T1 energy relaxation (spin decay)
- T2 dephasing (phase coherence loss)

Based on: "User-Friendly Truncated Wigner Approximation for Dissipative Spin Dynamics"
Hosseinabadi, Chelpanova, and Marino, PRX Quantum 6, 030344 (2025)
"""

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from twa_framework import TWASpinSimulator, IonTrapDissipationRates


class H2_TWA_Simulator:
    """
    H2 molecule simulator with TWA for dissipative dynamics.

    Combines quantum chemistry Hamiltonian with realistic decoherence:
    - 4 qubits representing H2 molecular orbitals
    - Pauli string decomposition of electronic Hamiltonian
    - TWA for T1/T2 dissipation
    - Hardware-realistic 171Yb+ parameters
    """

    def __init__(self, n_trajectories: int = 500):
        """
        Initialize H2 TWA simulator.

        Args:
            n_trajectories: Number of stochastic trajectories for TWA
        """
        self.n_qubits = 4
        self.n_trajectories = n_trajectories

        # Initialize TWA simulator
        self.twa = TWASpinSimulator(self.n_qubits, n_trajectories)

        # Hardware parameters
        self.hardware = IonTrapDissipationRates()

        # Pauli matrices for building quantum Hamiltonian (reference)
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

        print(f"Initialized H2 TWA simulator:")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1 = {self.hardware.T1} s")
        print(f"  T2 = {self.hardware.T2} s")

    def build_h2_classical_hamiltonian(self, r: float, s: np.ndarray) -> float:
        """
        Build classical H2 Hamiltonian evaluated at spin configuration s.

        Maps Pauli string operators to classical spin products:
        σ̂ᶻ_i → s^z_i, σ̂ˣ_i → s^x_i, σ̂ʸ_i → s^y_i

        Args:
            r: Bond distance in Angstroms
            s: Classical spin configuration, shape (4, 3)

        Returns:
            H(s): Classical Hamiltonian value
        """
        # Empirical parameters for H2 at distance r
        e_nuc = 1.0 / r
        t = 0.52917 * np.exp(-1.5 * (r - 0.741))
        mu = -1.1256 + 0.2 * r
        u = 0.6744 / (1 + 0.1 * r)
        v = 0.1815 * np.exp(-0.5 * r)

        H = 0.0

        # Identity term
        H += e_nuc

        # Single-qubit Z terms
        H += (mu / 2) * s[0, 2]  # Z on qubit 0
        H += (mu / 2) * s[1, 2]  # Z on qubit 1
        H += (mu / 4) * s[2, 2]  # Z on qubit 2
        H += (mu / 4) * s[3, 2]  # Z on qubit 3

        # Hopping terms: X_i X_j and Y_i Y_j
        H += (-t / 2) * s[0, 0] * s[2, 0]  # X0 X2
        H += (-t / 2) * s[0, 1] * s[2, 1]  # Y0 Y2
        H += (-t / 2) * s[1, 0] * s[3, 0]  # X1 X3
        H += (-t / 2) * s[1, 1] * s[3, 1]  # Y1 Y3

        # Two-qubit ZZ interaction terms
        H += (u / 4) * s[0, 2] * s[1, 2]   # Z0 Z1
        H += (u / 4) * s[2, 2] * s[3, 2]   # Z2 Z3
        H += (v / 4) * s[0, 2] * s[2, 2]   # Z0 Z2
        H += (v / 4) * s[1, 2] * s[3, 2]   # Z1 Z3
        H += (v / 8) * s[0, 2] * s[3, 2]   # Z0 Z3
        H += (v / 8) * s[1, 2] * s[2, 2]   # Z1 Z2

        return H

    def hamiltonian_gradient(self, r: float, s: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∂H/∂s for equations of motion.

        Args:
            r: Bond distance
            s: Spin configuration (4, 3)

        Returns:
            grad_H: Gradient array of shape (4, 3)
        """
        # Parameters
        t = 0.52917 * np.exp(-1.5 * (r - 0.741))
        mu = -1.1256 + 0.2 * r
        u = 0.6744 / (1 + 0.1 * r)
        v = 0.1815 * np.exp(-0.5 * r)

        grad = np.zeros((4, 3))

        # Gradient with respect to each spin component
        # ∂H/∂s^x
        grad[0, 0] = (-t / 2) * s[2, 0]  # from X0 X2
        grad[1, 0] = (-t / 2) * s[3, 0]  # from X1 X3
        grad[2, 0] = (-t / 2) * s[0, 0]  # from X0 X2
        grad[3, 0] = (-t / 2) * s[1, 0]  # from X1 X3

        # ∂H/∂s^y
        grad[0, 1] = (-t / 2) * s[2, 1]  # from Y0 Y2
        grad[1, 1] = (-t / 2) * s[3, 1]  # from Y1 Y3
        grad[2, 1] = (-t / 2) * s[0, 1]  # from Y0 Y2
        grad[3, 1] = (-t / 2) * s[1, 1]  # from Y1 Y3

        # ∂H/∂s^z (all Z terms)
        grad[0, 2] = (mu / 2) + (u / 4) * s[1, 2] + (v / 4) * s[2, 2] + (v / 8) * s[3, 2]
        grad[1, 2] = (mu / 2) + (u / 4) * s[0, 2] + (v / 4) * s[3, 2] + (v / 8) * s[2, 2]
        grad[2, 2] = (mu / 4) + (u / 4) * s[3, 2] + (v / 4) * s[0, 2] + (v / 8) * s[1, 2]
        grad[3, 2] = (mu / 4) + (u / 4) * s[2, 2] + (v / 4) * s[1, 2] + (v / 8) * s[0, 2]

        return grad

    def equations_of_motion_twa(self, t: float, s: np.ndarray, r: float,
                                noise: Dict[str, np.ndarray]) -> np.ndarray:
        """
        TWA equations of motion for H2 system with dissipation.

        Combines:
        1. Coherent Hamiltonian evolution: 2(s × ∇H)
        2. T1 decay terms
        3. T2 dephasing terms
        4. Stochastic noise

        Args:
            t: Current time
            s: Spin configuration (4, 3)
            r: Bond distance
            noise: Noise realizations

        Returns:
            ds/dt: Time derivatives (4, 3)
        """
        dsdt = np.zeros_like(s)

        # Coherent part: {s, H}_p = 2(s × ∇H)
        grad_H = self.hamiltonian_gradient(r, s)
        coherent = 2.0 * np.cross(s, grad_H)
        dsdt += coherent

        # Dissipative part (T1 decay)
        if 'decay_x' in noise:
            gamma = self.hardware.gamma_decay
            for k in range(self.n_qubits):
                xi_x = noise['decay_x'][k]
                xi_y = noise['decay_y'][k]

                # From TWA paper Table I (spin loss channel)
                dsdt[k, 0] += (gamma / 2) * s[k, 0] * s[k, 2] + xi_x * s[k, 2]
                dsdt[k, 1] += (gamma / 2) * s[k, 1] * s[k, 2] + xi_y * s[k, 2]
                dsdt[k, 2] += -(gamma / 2) * (s[k, 0]**2 + s[k, 1]**2) - (xi_x * s[k, 0] + xi_y * s[k, 1])

        # Dephasing part (T2)
        if 'dephasing' in noise:
            kappa = self.hardware.kappa_dephasing
            for k in range(self.n_qubits):
                eta = noise['dephasing'][k]

                # From TWA paper Table I (dephasing channel)
                dsdt[k, 0] += 2 * eta * s[k, 1]
                dsdt[k, 1] += -2 * eta * s[k, 0]
                # ds^z/dt unchanged by pure dephasing

        return dsdt

    def simulate_twa_dynamics(self, r: float, total_time: float, n_steps: int = 100,
                             add_T1: bool = True, add_T2: bool = True) -> Dict:
        """
        Simulate H2 dynamics with TWA including dissipation.

        Args:
            r: Bond distance in Angstroms
            total_time: Total simulation time (a.u.)
            n_steps: Number of time steps
            add_T1: Include T1 energy relaxation
            add_T2: Include T2 dephasing

        Returns:
            Dictionary with simulation results
        """
        dt = total_time / n_steps
        times = np.linspace(0, total_time, n_steps)

        print(f"\nRunning TWA simulation:")
        print(f"  Bond distance: R = {r:.3f} Å")
        print(f"  Time steps: {n_steps}, dt = {dt:.4f}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1 decay: {'ON' if add_T1 else 'OFF'}")
        print(f"  T2 dephasing: {'ON' if add_T2 else 'OFF'}")

        # Set up dissipation channels
        if add_T1:
            self.twa.add_dissipation('decay', self.hardware.gamma_decay, list(range(self.n_qubits)))
        if add_T2:
            self.twa.add_dissipation('dephasing', self.hardware.kappa_dephasing, list(range(self.n_qubits)))

        # Storage for all trajectories
        all_spins = np.zeros((self.n_trajectories, n_steps, self.n_qubits, 3))
        all_energies = np.zeros((self.n_trajectories, n_steps))

        # Run multiple trajectories
        for traj in range(self.n_trajectories):
            # Initialize state (Hartree-Fock: |0011⟩)
            # Qubits 0,1 in |↓⟩, qubits 2,3 in |↑⟩
            s = np.zeros((self.n_qubits, 3))
            for k in range(self.n_qubits):
                sx = np.random.choice([-1, 1])
                sy = np.random.choice([-1, 1])
                sz = 1 if k >= 2 else -1  # |0011⟩ state
                s[k] = [sx, sy, sz]

            # Time evolution
            for step in range(n_steps):
                # Store current state
                all_spins[traj, step] = s.copy()
                E = self.build_h2_classical_hamiltonian(r, s)
                all_energies[traj, step] = E

                # Generate noise for this step
                noise = self.twa.generate_noise(dt)

                # RK4 integration
                k1 = self.equations_of_motion_twa(times[step], s, r, noise)
                k2 = self.equations_of_motion_twa(times[step] + dt/2, s + dt*k1/2, r, noise)
                k3 = self.equations_of_motion_twa(times[step] + dt/2, s + dt*k2/2, r, noise)
                k4 = self.equations_of_motion_twa(times[step] + dt, s + dt*k3, r, noise)

                s = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

            if (traj + 1) % 100 == 0:
                print(f"  Completed trajectory {traj + 1}/{self.n_trajectories}")

        # Compute expectation values
        avg_spins = np.mean(all_spins, axis=0)  # Average over trajectories
        avg_energies = np.mean(all_energies, axis=0)
        std_energies = np.std(all_energies, axis=0)

        # Compute magnetization
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

    def compare_with_ideal(self, r: float = 0.74, total_time: float = 10.0):
        """
        Compare TWA dissipative dynamics with ideal (no dissipation) case.

        Args:
            r: Bond distance
            total_time: Simulation time
        """
        print("\n" + "=" * 70)
        print("COMPARING IDEAL VS. DISSIPATIVE DYNAMICS")
        print("=" * 70)

        # Run ideal (no dissipation)
        print("\n[1] Running ideal dynamics (no T1/T2)...")
        results_ideal = self.simulate_twa_dynamics(r, total_time, n_steps=100,
                                                   add_T1=False, add_T2=False)

        # Run with only T2 dephasing
        print("\n[2] Running with T2 dephasing only...")
        self.twa.dissipation_channels = []  # Reset
        results_T2 = self.simulate_twa_dynamics(r, total_time, n_steps=100,
                                                add_T1=False, add_T2=True)

        # Run with T1 and T2
        print("\n[3] Running with T1 decay + T2 dephasing...")
        self.twa.dissipation_channels = []  # Reset
        results_full = self.simulate_twa_dynamics(r, total_time, n_steps=100,
                                                  add_T1=True, add_T2=True)

        # Visualization
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
        ax.set_title(f'H₂ Energy Evolution (R = {r:.2f} Å)')
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
        n_show = 10  # Show first 10 trajectories
        for traj in range(n_show):
            ax.plot(results_full['times'],
                   results_full['all_spins'][traj, :, 0, 2],
                   alpha=0.3, linewidth=0.5, color='gray')
        ax.plot(results_full['times'], results_full['avg_spins'][:, 0, 2],
                'r-', linewidth=3, label='Average ⟨σᶻ₀⟩')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('σᶻ₀')
        ax.set_title('Sample Trajectories (Qubit 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('H₂ Molecule: Ideal vs. Dissipative Dynamics (TWA)',
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
    print("H₂ MOLECULE SIMULATION WITH TWA DISSIPATION")
    print("=" * 70)
    print()
    print("This simulation uses the Truncated Wigner Approximation (TWA)")
    print("to model dissipative quantum dynamics of the H₂ molecule.")
    print()
    print("Features:")
    print("  - 4-qubit quantum chemistry Hamiltonian")
    print("  - Hardware-realistic T1/T2 decoherence (171Yb+)")
    print("  - Stochastic trajectory averaging")
    print("  - Comparison with ideal (no dissipation) case")
    print()
    print("=" * 70)
    print()

    # Create simulator
    h2_twa = H2_TWA_Simulator(n_trajectories=500)

    # Run comparison
    results = h2_twa.compare_with_ideal(r=0.74, total_time=20.0)

    print("\n✓ Simulation complete! Check the plots to see dissipative effects.")
