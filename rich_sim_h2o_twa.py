"""
H2O Molecule Simulation with Truncated Wigner Approximation (TWA)

This module extends the original H2O simulation (rich_sim_h2o.py) with dissipative
dynamics using the TWA method. It models realistic decoherence effects for a
10-qubit quantum chemistry simulation.

Based on: "User-Friendly Truncated Wigner Approximation for Dissipative Spin Dynamics"
Hosseinabadi, Chelpanova, and Marino, PRX Quantum 6, 030344 (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from twa_framework import TWASpinSimulator, IonTrapDissipationRates


class H2O_TWA_Simulator:
    """
    H2O molecule simulator with TWA for dissipative dynamics.

    Features:
    - 10 qubits (compressed from 14 via QEE)
    - Smart term grouping (73 terms → 3 operations)
    - TWA for T1/T2 dissipation
    - Hardware-realistic 171Yb+ parameters
    """

    def __init__(self, n_trajectories: int = 300, energy_scale: float = 1e15):
        """
        Initialize H2O TWA simulator.

        Args:
            n_trajectories: Number of stochastic trajectories for TWA
                           (reduced from H2 due to larger system size)
            energy_scale: Scaling factor to match dissipation rates to Hamiltonian
                         (default: 1e15 gives reasonable dissipation for model)
        """
        self.n_qubits = 10
        self.n_trajectories = n_trajectories

        # Initialize TWA simulator
        self.twa = TWASpinSimulator(self.n_qubits, n_trajectories)

        # Hardware parameters (with scaled rates)
        self.hardware = IonTrapDissipationRates(energy_scale=energy_scale)

        # Generate Hamiltonian terms
        self.hamiltonian_terms = self._generate_h2o_hamiltonian()

        print(f"\nInitialized H2O TWA simulator:")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Hamiltonian terms: {len(self.hamiltonian_terms)}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1 = {self.hardware.T1_SI} s (SI)")
        print(f"  T2 = {self.hardware.T2_SI} s (SI)")

    def _generate_h2o_hamiltonian(self) -> List[Tuple[str, List[int], float]]:
        """
        Generate H2O Hamiltonian terms (same as original rich_sim_h2o.py).

        Returns:
            List of (pauli_string, qubits, coefficient) tuples
        """
        terms = []

        # Diagonal Z terms (one-body energies)
        for i in range(self.n_qubits):
            coeff = -10.0 + 2.0 * i
            terms.append(('Z', [i], coeff))

        # Two-qubit ZZ terms (Coulomb repulsion)
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                coeff = 0.5 * np.exp(-0.3 * abs(i - j))
                terms.append(('ZZ', [i, j], coeff))

        # Hopping terms (XX and YY)
        for i in range(self.n_qubits - 1):
            coeff = -1.5 * np.exp(-0.1 * i)
            terms.append(('XX', [i, i + 1], coeff))
            terms.append(('YY', [i, i + 1], coeff))

        return terms

    def build_h2o_classical_hamiltonian(self, s: np.ndarray) -> float:
        """
        Build classical H2O Hamiltonian evaluated at spin configuration s.

        Args:
            s: Classical spin configuration, shape (10, 3)

        Returns:
            H(s): Classical Hamiltonian value
        """
        H = 0.0

        for pauli_string, qubits, coeff in self.hamiltonian_terms:
            if pauli_string == 'Z':
                # Single Z term
                q = qubits[0]
                H += coeff * s[q, 2]

            elif pauli_string == 'ZZ':
                # Two-qubit ZZ term
                q1, q2 = qubits[0], qubits[1]
                H += coeff * s[q1, 2] * s[q2, 2]

            elif pauli_string == 'XX':
                # XX hopping term
                q1, q2 = qubits[0], qubits[1]
                H += coeff * s[q1, 0] * s[q2, 0]

            elif pauli_string == 'YY':
                # YY hopping term
                q1, q2 = qubits[0], qubits[1]
                H += coeff * s[q1, 1] * s[q2, 1]

        return H

    def hamiltonian_gradient(self, s: np.ndarray) -> np.ndarray:
        """
        Compute gradient ∂H/∂s for equations of motion.

        Args:
            s: Spin configuration (10, 3)

        Returns:
            grad_H: Gradient array of shape (10, 3)
        """
        grad = np.zeros((self.n_qubits, 3))

        for pauli_string, qubits, coeff in self.hamiltonian_terms:
            if pauli_string == 'Z':
                # ∂(c·s^z_i)/∂s^z_i = c
                q = qubits[0]
                grad[q, 2] += coeff

            elif pauli_string == 'ZZ':
                # ∂(c·s^z_i·s^z_j)/∂s^z_i = c·s^z_j
                q1, q2 = qubits[0], qubits[1]
                grad[q1, 2] += coeff * s[q2, 2]
                grad[q2, 2] += coeff * s[q1, 2]

            elif pauli_string == 'XX':
                # ∂(c·s^x_i·s^x_j)/∂s^x_i = c·s^x_j
                q1, q2 = qubits[0], qubits[1]
                grad[q1, 0] += coeff * s[q2, 0]
                grad[q2, 0] += coeff * s[q1, 0]

            elif pauli_string == 'YY':
                # ∂(c·s^y_i·s^y_j)/∂s^y_i = c·s^y_j
                q1, q2 = qubits[0], qubits[1]
                grad[q1, 1] += coeff * s[q2, 1]
                grad[q2, 1] += coeff * s[q1, 1]

        return grad

    def equations_of_motion_twa(self, t: float, s: np.ndarray,
                                noise: Dict[str, np.ndarray]) -> np.ndarray:
        """
        TWA equations of motion for H2O system with dissipation.

        Args:
            t: Current time
            s: Spin configuration (10, 3)
            noise: Noise realizations

        Returns:
            ds/dt: Time derivatives (10, 3)
        """
        dsdt = np.zeros_like(s)

        # Coherent part: {s, H}_p = 2(s × ∇H)
        grad_H = self.hamiltonian_gradient(s)
        coherent = 2.0 * np.cross(s, grad_H)
        dsdt += coherent

        # Dissipative part (T1 decay)
        if 'decay_x' in noise:
            gamma = self.hardware.gamma_decay
            for k in range(self.n_qubits):
                xi_x = noise['decay_x'][k]
                xi_y = noise['decay_y'][k]

                dsdt[k, 0] += (gamma / 2) * s[k, 0] * s[k, 2] + xi_x * s[k, 2]
                dsdt[k, 1] += (gamma / 2) * s[k, 1] * s[k, 2] + xi_y * s[k, 2]
                dsdt[k, 2] += -(gamma / 2) * (s[k, 0]**2 + s[k, 1]**2) - (xi_x * s[k, 0] + xi_y * s[k, 1])

        # Dephasing part (T2)
        if 'dephasing' in noise:
            kappa = self.hardware.kappa_dephasing
            for k in range(self.n_qubits):
                eta = noise['dephasing'][k]

                dsdt[k, 0] += 2 * eta * s[k, 1]
                dsdt[k, 1] += -2 * eta * s[k, 0]

        return dsdt

    def simulate_twa_dynamics(self, total_time: float, n_steps: int = 200,
                             add_T1: bool = True, add_T2: bool = True,
                             renormalize_spins: bool = True) -> Dict:
        """
        Simulate H2O dynamics with TWA including dissipation.

        Note: Increased n_steps default to prevent numerical instability.

        Args:
            total_time: Total simulation time (a.u.)
            n_steps: Number of time steps (increased from 50 to 200 for stability)
            add_T1: Include T1 energy relaxation
            add_T2: Include T2 dephasing
            renormalize_spins: Renormalize spin vectors to prevent numerical drift

        Returns:
            Dictionary with simulation results
        """
        dt = total_time / n_steps
        times = np.linspace(0, total_time, n_steps)

        print(f"\nRunning H2O TWA simulation:")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Time steps: {n_steps}, dt = {dt:.4f}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1 decay: {'ON' if add_T1 else 'OFF'}")
        print(f"  T2 dephasing: {'ON' if add_T2 else 'OFF'}")
        print(f"  Spin renormalization: {'ON' if renormalize_spins else 'OFF'}")

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
            # Initialize state (Hartree-Fock: first 5 qubits excited)
            s = np.zeros((self.n_qubits, 3))
            for k in range(self.n_qubits):
                sx = np.random.choice([-1, 1])
                sy = np.random.choice([-1, 1])
                sz = 1 if k < 5 else -1  # |1111100000⟩ state
                s[k] = [sx, sy, sz]

            # Time evolution
            for step in range(n_steps):
                # Store current state
                all_spins[traj, step] = s.copy()
                E = self.build_h2o_classical_hamiltonian(s)
                all_energies[traj, step] = E

                # Check for numerical instability
                if step > 0 and (np.isnan(E) or np.isinf(E) or abs(E) > 1e6):
                    print(f"  WARNING: Trajectory {traj} became unstable at step {step}")
                    print(f"    Energy = {E}, |s|² = {np.sum(s**2, axis=1)[:3]}")
                    # Fill remaining steps with NaN
                    all_spins[traj, step:] = np.nan
                    all_energies[traj, step:] = np.nan
                    break

                # Generate noise for this step
                noise = self.twa.generate_noise(dt)

                # RK4 integration
                k1 = self.equations_of_motion_twa(times[step], s, noise)
                k2 = self.equations_of_motion_twa(times[step] + dt/2, s + dt*k1/2, noise)
                k3 = self.equations_of_motion_twa(times[step] + dt/2, s + dt*k2/2, noise)
                k4 = self.equations_of_motion_twa(times[step] + dt, s + dt*k3, noise)

                s = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

                # Renormalize spins to prevent numerical drift
                if renormalize_spins and (step + 1) % 10 == 0:
                    self.twa.check_spin_conservation(s, renormalize=True)

            if (traj + 1) % 50 == 0:
                print(f"  Completed trajectory {traj + 1}/{self.n_trajectories}")

        # Compute expectation values (ignoring NaN trajectories)
        avg_spins = np.nanmean(all_spins, axis=0)
        avg_energies = np.nanmean(all_energies, axis=0)
        std_energies = np.nanstd(all_energies, axis=0)

        # Compute magnetization
        magnetization = np.nanmean(np.sum(all_spins[:, :, :, 2], axis=2), axis=0)

        # Count successful trajectories
        n_successful = np.sum(~np.isnan(all_energies[:, -1]))
        if n_successful < self.n_trajectories:
            print(f"  WARNING: {self.n_trajectories - n_successful} trajectories failed")
        print(f"  Successful trajectories: {n_successful}/{self.n_trajectories}")

        return {
            'times': times,
            'avg_spins': avg_spins,
            'avg_energies': avg_energies,
            'std_energies': std_energies,
            'magnetization': magnetization,
            'all_spins': all_spins,
            'all_energies': all_energies,
        }

    def compare_dissipation_effects(self, total_time: float = 5.0, n_steps: int = 200):
        """
        Compare H2O dynamics with different dissipation channels.

        Args:
            total_time: Simulation time (reduced from 10.0 to 5.0 for stability)
            n_steps: Number of time steps (increased to 200 for stability)
        """
        print("\n" + "=" * 70)
        print("H2O: COMPARING DISSIPATION EFFECTS")
        print("=" * 70)

        # Run ideal (no dissipation)
        print("\n[1] Running ideal dynamics (no T1/T2)...")
        results_ideal = self.simulate_twa_dynamics(total_time, n_steps=n_steps,
                                                   add_T1=False, add_T2=False)

        # Run with T1 and T2
        print("\n[2] Running with T1 decay + T2 dephasing...")
        self.twa.dissipation_channels = []  # Reset
        results_full = self.simulate_twa_dynamics(total_time, n_steps=n_steps,
                                                  add_T1=True, add_T2=True)

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Energy evolution
        ax = axes[0, 0]
        ax.plot(results_ideal['times'], results_ideal['avg_energies'],
                'b-', linewidth=2, label='Ideal (no dissipation)')
        ax.fill_between(results_ideal['times'],
                        results_ideal['avg_energies'] - results_ideal['std_energies'],
                        results_ideal['avg_energies'] + results_ideal['std_energies'],
                        alpha=0.2, color='blue')
        ax.plot(results_full['times'], results_full['avg_energies'],
                'r-', linewidth=2, label='T1 + T2')
        ax.fill_between(results_full['times'],
                        results_full['avg_energies'] - results_full['std_energies'],
                        results_full['avg_energies'] + results_full['std_energies'],
                        alpha=0.2, color='red')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Energy (Hartree)')
        ax.set_title('H₂O Energy Evolution (10 qubits)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Magnetization
        ax = axes[0, 1]
        ax.plot(results_ideal['times'], results_ideal['magnetization'],
                'b-', linewidth=2, label='Ideal')
        ax.plot(results_full['times'], results_full['magnetization'],
                'r-', linewidth=2, label='T1 + T2')
        ax.set_xlabel('Time (a.u.)')
        ax.set_ylabel('Total Magnetization ⟨∑ᵢ σᶻᵢ⟩')
        ax.set_title('Magnetization Dynamics')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Per-qubit magnetization (final time)
        ax = axes[1, 0]
        final_mags_ideal = results_ideal['avg_spins'][-1, :, 2]
        final_mags_full = results_full['avg_spins'][-1, :, 2]
        x = np.arange(self.n_qubits)
        width = 0.35
        ax.bar(x - width/2, final_mags_ideal, width, label='Ideal', alpha=0.7, color='blue')
        ax.bar(x + width/2, final_mags_full, width, label='T1 + T2', alpha=0.7, color='red')
        ax.set_xlabel('Qubit Index')
        ax.set_ylabel('⟨σᶻᵢ⟩')
        ax.set_title(f'Final Magnetization per Qubit (t = {total_time:.1f})')
        ax.set_xticks(x)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 4: Energy distribution histogram (final time)
        ax = axes[1, 1]
        ax.hist(results_ideal['all_energies'][:, -1], bins=30, alpha=0.5,
                label='Ideal', color='blue', density=True)
        ax.hist(results_full['all_energies'][:, -1], bins=30, alpha=0.5,
                label='T1 + T2', color='red', density=True)
        ax.set_xlabel('Energy (Hartree)')
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Energy Distribution (t = {total_time:.1f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('H₂O Molecule: Dissipative Dynamics (TWA)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print("\n" + "=" * 70)
        print("SIMULATION COMPLETE")
        print("=" * 70)
        print(f"\nFinal energies:")
        print(f"  Ideal:   {results_ideal['avg_energies'][-1]:.4f} ± {results_ideal['std_energies'][-1]:.4f} H")
        print(f"  T1 + T2: {results_full['avg_energies'][-1]:.4f} ± {results_full['std_energies'][-1]:.4f} H")
        print(f"\nFinal magnetization:")
        print(f"  Ideal:   {results_ideal['magnetization'][-1]:.4f}")
        print(f"  T1 + T2: {results_full['magnetization'][-1]:.4f}")

        return {
            'ideal': results_ideal,
            'full': results_full
        }


if __name__ == "__main__":
    print("=" * 70)
    print("H₂O MOLECULE SIMULATION WITH TWA DISSIPATION")
    print("=" * 70)
    print()
    print("This simulation uses the Truncated Wigner Approximation (TWA)")
    print("to model dissipative quantum dynamics of the H₂O molecule.")
    print()
    print("Features:")
    print("  - 10-qubit quantum chemistry Hamiltonian (QEE compressed)")
    print("  - Hardware-realistic T1/T2 decoherence (171Yb+)")
    print("  - Stochastic trajectory averaging")
    print("  - Comparison with ideal (no dissipation) case")
    print()
    print("Note: This is computationally more intensive than H2 (10 vs 4 qubits)")
    print("=" * 70)
    print()

    # Create simulator with properly scaled dissipation
    # energy_scale=1e15 gives reasonable dissipation for this model
    h2o_twa = H2O_TWA_Simulator(n_trajectories=300, energy_scale=1e15)

    # Run comparison with shorter time and more steps for stability
    results = h2o_twa.compare_dissipation_effects(total_time=5.0, n_steps=200)

    print("\n✓ Simulation complete! Check the plots to see dissipative effects.")
