"""
LiH Molecule Simulation with Truncated Wigner Approximation (TWA)

This module extends the LiH VQE simulation (rich_sim_lih.py) with dissipative
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


class LiH_TWA_Simulator:
    """
    LiH molecule simulator with TWA for dissipative dynamics.

    Combines quantum chemistry Hamiltonian with realistic decoherence:
    - 4 qubits representing LiH molecular orbitals (minimal active space)
    - Pauli string decomposition of electronic Hamiltonian
    - TWA for T1/T2 dissipation
    - Hardware-realistic 171Yb+ parameters
    """

    def __init__(self, n_trajectories: int = 500):
        """
        Initialize LiH TWA simulator.

        Args:
            n_trajectories: Number of stochastic trajectories for TWA
        """
        self.n_qubits = 4  # Minimal active space
        self.n_trajectories = n_trajectories

        # Initialize TWA simulator
        self.twa = TWASpinSimulator(self.n_qubits, n_trajectories)

        # Hardware parameters
        self.hardware = IonTrapDissipationRates()

        # Pauli matrices for reference
        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

        print(f"Initialized LiH TWA simulator:")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1 = {self.hardware.T1_SI} s")
        print(f"  T2 = {self.hardware.T2_SI} s")

    def build_lih_classical_hamiltonian(self, r: float, s: np.ndarray) -> float:
        """
        Build classical LiH Hamiltonian evaluated at spin configuration s.

        Maps Pauli string operators to classical spin products:
        σ̂ᶻ_i → s^z_i, σ̂ˣ_i → s^x_i, σ̂ʸ_i → s^y_i

        Args:
            r: Li-H bond distance in Angstroms
            s: Classical spin configuration, shape (4, 3)

        Returns:
            H(s): Classical Hamiltonian value
        """
        # EXACT HAMILTONIAN COEFFICIENTS (Generated from PySCF)
        # Ground state energy: -9.20404581 H at r=1.5949 Å
        # Active space: (2 electrons, 2 orbitals), 1 frozen core
        # Basis: STO-3G
        H = 0.0

        # Identity term (includes core energy + nuclear repulsion)
        H += -6.80295271

        # Single-qubit Z terms
        H += +0.74730805 * s[0, 2]  # Z on qubit 0
        H += +0.74730805 * s[1, 2]  # Z on qubit 1
        H += +0.56294509 * s[2, 2]  # Z on qubit 2
        H += +0.56294509 * s[3, 2]  # Z on qubit 3

        # Two-qubit ZZ terms
        H += +0.12191619 * s[0, 2] * s[1, 2]   # Z0 Z1
        H += +0.08448401 * s[2, 2] * s[3, 2]   # Z2 Z3
        H += +0.00325324 * s[0, 2] * s[2, 2]   # Z0 Z2
        H += +0.00325324 * s[0, 2] * s[3, 2]   # Z0 Z3
        H += +0.00325324 * s[1, 2] * s[2, 2]   # Z1 Z2
        H += +0.00325324 * s[1, 2] * s[3, 2]   # Z1 Z3

        # Hopping terms: XX + YY
        H += +0.03303591 * s[0, 0] * s[2, 0]   # X0 X2
        H += +0.03303591 * s[0, 1] * s[2, 1]   # Y0 Y2
        H += +0.03303591 * s[1, 0] * s[3, 0]   # X1 X3
        H += +0.03303591 * s[1, 1] * s[3, 1]   # Y1 Y3

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
        grad = np.zeros((4, 3))

        # Gradient with respect to s^x (from XX terms) - EXACT coefficients
        grad[0, 0] = +0.03303591 * s[2, 0]  # from X0X2
        grad[1, 0] = +0.03303591 * s[3, 0]  # from X1X3
        grad[2, 0] = +0.03303591 * s[0, 0]  # from X0X2
        grad[3, 0] = +0.03303591 * s[1, 0]  # from X1X3

        # Gradient with respect to s^y (from YY terms) - EXACT coefficients
        grad[0, 1] = +0.03303591 * s[2, 1]  # from Y0Y2
        grad[1, 1] = +0.03303591 * s[3, 1]  # from Y1Y3
        grad[2, 1] = +0.03303591 * s[0, 1]  # from Y0Y2
        grad[3, 1] = +0.03303591 * s[1, 1]  # from Y1Y3

        # Gradient with respect to s^z (from all Z terms) - EXACT coefficients
        grad[0, 2] = +0.74730805 + 0.12191619 * s[1, 2] + 0.00325324 * s[2, 2] + 0.00325324 * s[3, 2]
        grad[1, 2] = +0.74730805 + 0.12191619 * s[0, 2] + 0.00325324 * s[2, 2] + 0.00325324 * s[3, 2]
        grad[2, 2] = +0.56294509 + 0.08448401 * s[3, 2] + 0.00325324 * s[0, 2] + 0.00325324 * s[1, 2]
        grad[3, 2] = +0.56294509 + 0.08448401 * s[2, 2] + 0.00325324 * s[0, 2] + 0.00325324 * s[1, 2]

        return grad

    def equations_of_motion_twa(self, t: float, s: np.ndarray, r: float,
                                noise: Dict[str, np.ndarray]) -> np.ndarray:
        """
        TWA equations of motion for LiH system with dissipation.

        Combines coherent evolution (Hamiltonian gradient) with dissipative terms.

        Args:
            t: Current time
            s: Spin configuration (4, 3)
            r: Bond distance
            noise: Noise realizations

        Returns:
            ds/dt: Time derivatives
        """
        # Create gradient function wrapper
        def H_grad_func(spins):
            return self.hamiltonian_gradient(r, spins)

        # Use TWA framework's equations of motion
        return self.twa.equations_of_motion_decay(t, s, H_grad_func, noise)

    def simulate_dynamics(self, r: float, total_time: float, dt: float = 0.01,
                         add_t1: bool = True, add_t2: bool = True,
                         initial_state: str = 'ground') -> Dict:
        """
        Simulate LiH dynamics with dissipation using TWA.

        Args:
            r: Bond distance in Angstroms
            total_time: Total simulation time
            dt: Time step size
            add_t1: Include T1 (spin decay) dissipation
            add_t2: Include T2 (dephasing) dissipation
            initial_state: Initial state ('ground', 'excited', 'superposition')

        Returns:
            Dictionary with simulation results
        """
        n_steps = int(total_time / dt)
        times = np.linspace(0, total_time, n_steps)

        # Add dissipation channels
        self.twa.dissipation_channels = []
        if add_t1:
            gamma = self.hardware.gamma_decay
            self.twa.add_dissipation('decay', gamma, list(range(self.n_qubits)))
        if add_t2:
            kappa = self.hardware.kappa_dephasing
            self.twa.add_dissipation('dephasing', kappa, list(range(self.n_qubits)))

        # Storage for all trajectories
        all_spins = np.zeros((n_steps, self.n_trajectories, self.n_qubits, 3))
        all_energies = np.zeros((n_steps, self.n_trajectories))

        # Evolve each trajectory
        for traj in range(self.n_trajectories):
            # Initialize spin configuration
            s = self.twa.discrete_sample_initial_state(initial_state)

            for step, time in enumerate(times):
                # Store current state
                all_spins[step, traj] = s.copy()
                all_energies[step, traj] = self.build_lih_classical_hamiltonian(r, s)

                if step < n_steps - 1:
                    # Generate noise
                    noise = self.twa.generate_noise(dt)

                    # RK4 integration step
                    s = self.twa.rk4_step(time, s, dt,
                                         lambda spins: self.hamiltonian_gradient(r, spins),
                                         noise)

                    # Optional: renormalize spins for stability
                    self.twa.check_spin_conservation(s, renormalize=True)

            if (traj + 1) % 100 == 0:
                print(f"  Completed trajectory {traj + 1}/{self.n_trajectories}")

        # Compute expectation values
        avg_spins = np.mean(all_spins, axis=1)  # Average over trajectories
        avg_energies = np.mean(all_energies, axis=1)

        # Compute magnetization
        magnetization = np.sum(avg_spins[:, :, 2], axis=1)  # Sum of <sz> over qubits

        return {
            'times': times,
            'avg_spins': avg_spins,
            'avg_energies': avg_energies,
            'magnetization': magnetization,
            'all_trajectories': all_spins,
            'all_energies': all_energies,
            'r': r,
            'dt': dt,
            'n_trajectories': self.n_trajectories
        }

    def compare_with_ideal(self, r: float = 1.5949, total_time: float = 20.0,
                          dt: float = 0.01) -> Dict:
        """
        Compare ideal (no dissipation) vs dissipative dynamics.

        Args:
            r: Bond distance
            total_time: Simulation time
            dt: Time step

        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 70)
        print("COMPARING IDEAL VS DISSIPATIVE DYNAMICS FOR LiH")
        print("=" * 70)
        print(f"Bond distance: R = {r:.4f} Å")
        print(f"Simulation time: {total_time} a.u.")
        print(f"Time step: {dt} a.u.")
        print(f"Trajectories: {self.n_trajectories}")
        print()

        # Ideal dynamics (no dissipation)
        print("Simulating ideal dynamics (no T1/T2)...")
        ideal_results = self.simulate_dynamics(
            r, total_time, dt,
            add_t1=False, add_t2=False,
            initial_state='ground'
        )

        # Only T1
        print("\nSimulating with T1 decay only...")
        t1_results = self.simulate_dynamics(
            r, total_time, dt,
            add_t1=True, add_t2=False,
            initial_state='ground'
        )

        # Only T2
        print("\nSimulating with T2 dephasing only...")
        t2_results = self.simulate_dynamics(
            r, total_time, dt,
            add_t1=False, add_t2=True,
            initial_state='ground'
        )

        # Full dissipation (T1 + T2)
        print("\nSimulating with full dissipation (T1 + T2)...")
        full_results = self.simulate_dynamics(
            r, total_time, dt,
            add_t1=True, add_t2=True,
            initial_state='ground'
        )

        return {
            'ideal': ideal_results,
            't1_only': t1_results,
            't2_only': t2_results,
            'full': full_results,
            'r': r
        }


def plot_comparison_results(comparison: Dict):
    """Plot comparison of ideal vs dissipative dynamics."""
    times = comparison['ideal']['times']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Energy evolution
    ax1 = axes[0, 0]
    ax1.plot(times, comparison['ideal']['avg_energies'], 'b-', linewidth=2, label='Ideal')
    ax1.plot(times, comparison['t1_only']['avg_energies'], 'g--', linewidth=2, label='T1 only')
    ax1.plot(times, comparison['t2_only']['avg_energies'], 'orange', linestyle='-.', linewidth=2, label='T2 only')
    ax1.plot(times, comparison['full']['avg_energies'], 'r:', linewidth=2, label='T1 + T2')
    ax1.set_xlabel('Time (a.u.)')
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title('Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Magnetization
    ax2 = axes[0, 1]
    ax2.plot(times, comparison['ideal']['magnetization'], 'b-', linewidth=2, label='Ideal')
    ax2.plot(times, comparison['t1_only']['magnetization'], 'g--', linewidth=2, label='T1 only')
    ax2.plot(times, comparison['t2_only']['magnetization'], 'orange', linestyle='-.', linewidth=2, label='T2 only')
    ax2.plot(times, comparison['full']['magnetization'], 'r:', linewidth=2, label='T1 + T2')
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('Total Magnetization')
    ax2.set_title('Magnetization (Sum of ⟨σᶻ⟩)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy difference from ideal
    ax3 = axes[1, 0]
    ideal_E = comparison['ideal']['avg_energies']
    ax3.plot(times, comparison['t1_only']['avg_energies'] - ideal_E, 'g-', linewidth=2, label='T1 only')
    ax3.plot(times, comparison['t2_only']['avg_energies'] - ideal_E, 'orange', linewidth=2, label='T2 only')
    ax3.plot(times, comparison['full']['avg_energies'] - ideal_E, 'r-', linewidth=2, label='T1 + T2')
    ax3.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='Ideal')
    ax3.set_xlabel('Time (a.u.)')
    ax3.set_ylabel('ΔE from Ideal (Hartree)')
    ax3.set_title('Dissipation-Induced Energy Shift')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Individual spin components
    ax4 = axes[1, 1]
    # Plot <sz> for each qubit in full dissipation case
    for i in range(4):
        ax4.plot(times, comparison['full']['avg_spins'][:, i, 2],
                linewidth=2, label=f'Qubit {i}')
    ax4.set_xlabel('Time (a.u.)')
    ax4.set_ylabel('⟨σᶻ⟩')
    ax4.set_title('Individual Qubit Expectation (Full Dissipation)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f"LiH Dissipative Dynamics (R = {comparison['r']:.4f} Å)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    print("=" * 70)
    print("LiH MOLECULE TWA SIMULATION WITH DISSIPATION")
    print("=" * 70)
    print()

    # Create simulator
    lih_twa = LiH_TWA_Simulator(n_trajectories=500)

    # Run comparison
    r_eq = 1.5949  # Equilibrium bond length
    comparison = lih_twa.compare_with_ideal(
        r=r_eq,
        total_time=20.0,
        dt=0.01
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nInitial energy (ideal):    {comparison['ideal']['avg_energies'][0]:.6f} H")
    print(f"Final energy (ideal):      {comparison['ideal']['avg_energies'][-1]:.6f} H")
    print(f"Final energy (T1 only):    {comparison['t1_only']['avg_energies'][-1]:.6f} H")
    print(f"Final energy (T2 only):    {comparison['t2_only']['avg_energies'][-1]:.6f} H")
    print(f"Final energy (T1 + T2):    {comparison['full']['avg_energies'][-1]:.6f} H")
    print()
    print(f"Energy shift from T1:      {comparison['t1_only']['avg_energies'][-1] - comparison['ideal']['avg_energies'][-1]:.6e} H")
    print(f"Energy shift from T2:      {comparison['t2_only']['avg_energies'][-1] - comparison['ideal']['avg_energies'][-1]:.6e} H")
    print(f"Total dissipation effect:  {comparison['full']['avg_energies'][-1] - comparison['ideal']['avg_energies'][-1]:.6e} H")

    # Plot results
    fig = plot_comparison_results(comparison)
    plt.savefig('lih_twa_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: lih_twa_comparison.png")
    plt.show()

    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70)
