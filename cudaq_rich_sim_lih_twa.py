"""
GPU-Accelerated LiH Molecule TWA Simulation

This module provides GPU-accelerated simulation of LiH with dissipative dynamics
using CuPy for 10-100x speedup compared to CPU implementation.

Requires:
- CuPy (pip install cupy-cuda12x or cupy-cuda11x)
- CUDA-capable GPU

Based on cudaq_rich_sim_h2_twa.py GPU implementation.
"""

import numpy as np
import time
from typing import Dict

# Try to import CuPy
try:
    import cupy as cp
    from cupy import ndarray as cp_array
    GPU_AVAILABLE = True
except ImportError:
    print("Warning: CuPy not available. GPU acceleration disabled.")
    GPU_AVAILABLE = False
    cp = np


class CUDAQ_LiH_TWA_Simulator:
    """
    GPU-accelerated LiH TWA simulator using CuPy.

    Provides 10-100x speedup over CPU implementation for large
    numbers of trajectories.
    """

    def __init__(self, n_trajectories: int = 2000, use_gpu: bool = True):
        """
        Initialize GPU-accelerated LiH TWA simulator.

        Args:
            n_trajectories: Number of stochastic trajectories
            use_gpu: Use GPU if available
        """
        self.n_qubits = 4
        self.n_trajectories = n_trajectories
        self.use_gpu = use_gpu and GPU_AVAILABLE

        if self.use_gpu:
            print(f"✓ GPU acceleration enabled (CuPy)")
            print(f"  Device: {cp.cuda.Device().name}")
            self.xp = cp
        else:
            print("Running on CPU (NumPy)")
            self.xp = np

        # Hardware parameters (in atomic units, scaled)
        self.T1_SI = 1000.0  # seconds
        self.T2_SI = 1.0     # seconds
        self.AU_TIME = 2.4189e-17  # seconds per a.u.

        # Dissipation rates (scaled to match Hamiltonian units)
        energy_scale = 1.0  # Adjust based on typical H energies
        self.gamma_decay = (1.0 / self.T1_SI) * self.AU_TIME * energy_scale
        self.kappa_dephasing = (1.0 / self.T2_SI) * self.AU_TIME * energy_scale

        print(f"\nInitialized GPU LiH TWA simulator:")
        print(f"  Qubits: {self.n_qubits}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1 = {self.T1_SI} s → γ = {self.gamma_decay:.2e}")
        print(f"  T2 = {self.T2_SI} s → κ = {self.kappa_dephasing:.2e}")

    def to_gpu(self, array: np.ndarray):
        """Transfer array to GPU if using GPU."""
        if self.use_gpu:
            return cp.asarray(array)
        return array

    def to_cpu(self, array):
        """Transfer array to CPU."""
        if self.use_gpu and isinstance(array, cp_array):
            return cp.asnumpy(array)
        return array

    def build_lih_classical_hamiltonian_batch(self, r: float, s: cp_array) -> cp_array:
        """
        Build classical LiH Hamiltonian for batched spin configurations.

        Args:
            r: Bond distance
            s: Spin configurations, shape (n_traj, 4, 3)

        Returns:
            H(s): Energies for each trajectory, shape (n_traj,)
        """
        xp = self.xp

        # EXACT HAMILTONIAN COEFFICIENTS (Generated from PySCF)
        # Ground state energy: -9.20404581 H at r=1.5949 Å
        # Active space: (2 electrons, 2 orbitals), 1 frozen core
        # Basis: STO-3G

        # Initialize energy array
        H = xp.zeros(s.shape[0])

        # Identity term (includes core energy + nuclear repulsion)
        H += -6.80295271

        # Single-qubit Z terms
        H += +0.74730805 * s[:, 0, 2]  # Z0
        H += +0.74730805 * s[:, 1, 2]  # Z1
        H += +0.56294509 * s[:, 2, 2]  # Z2
        H += +0.56294509 * s[:, 3, 2]  # Z3

        # Two-qubit ZZ terms
        H += +0.12191619 * s[:, 0, 2] * s[:, 1, 2]   # Z0 Z1
        H += +0.08448401 * s[:, 2, 2] * s[:, 3, 2]   # Z2 Z3
        H += +0.00325324 * s[:, 0, 2] * s[:, 2, 2]   # Z0 Z2
        H += +0.00325324 * s[:, 0, 2] * s[:, 3, 2]   # Z0 Z3
        H += +0.00325324 * s[:, 1, 2] * s[:, 2, 2]   # Z1 Z2
        H += +0.00325324 * s[:, 1, 2] * s[:, 3, 2]   # Z1 Z3

        # Hopping terms: XX + YY
        H += +0.03303591 * s[:, 0, 0] * s[:, 2, 0]   # X0 X2
        H += +0.03303591 * s[:, 0, 1] * s[:, 2, 1]   # Y0 Y2
        H += +0.03303591 * s[:, 1, 0] * s[:, 3, 0]   # X1 X3
        H += +0.03303591 * s[:, 1, 1] * s[:, 3, 1]   # Y1 Y3

        return H

    def hamiltonian_gradient_batch(self, r: float, s: cp_array) -> cp_array:
        """
        Compute Hamiltonian gradient for batched configurations.

        Args:
            r: Bond distance
            s: Spin configurations (n_traj, 4, 3)

        Returns:
            grad_H: Gradients (n_traj, 4, 3)
        """
        xp = self.xp
        n_traj = s.shape[0]
        grad = xp.zeros((n_traj, 4, 3))

        # Gradient with respect to s^x - EXACT coefficients
        grad[:, 0, 0] = +0.03303591 * s[:, 2, 0]  # from X0X2
        grad[:, 1, 0] = +0.03303591 * s[:, 3, 0]  # from X1X3
        grad[:, 2, 0] = +0.03303591 * s[:, 0, 0]  # from X0X2
        grad[:, 3, 0] = +0.03303591 * s[:, 1, 0]  # from X1X3

        # Gradient with respect to s^y - EXACT coefficients
        grad[:, 0, 1] = +0.03303591 * s[:, 2, 1]  # from Y0Y2
        grad[:, 1, 1] = +0.03303591 * s[:, 3, 1]  # from Y1Y3
        grad[:, 2, 1] = +0.03303591 * s[:, 0, 1]  # from Y0Y2
        grad[:, 3, 1] = +0.03303591 * s[:, 1, 1]  # from Y1Y3

        # Gradient with respect to s^z - EXACT coefficients
        grad[:, 0, 2] = +0.74730805 + 0.12191619 * s[:, 1, 2] + 0.00325324 * s[:, 2, 2] + 0.00325324 * s[:, 3, 2]
        grad[:, 1, 2] = +0.74730805 + 0.12191619 * s[:, 0, 2] + 0.00325324 * s[:, 2, 2] + 0.00325324 * s[:, 3, 2]
        grad[:, 2, 2] = +0.56294509 + 0.08448401 * s[:, 3, 2] + 0.00325324 * s[:, 0, 2] + 0.00325324 * s[:, 1, 2]
        grad[:, 3, 2] = +0.56294509 + 0.08448401 * s[:, 2, 2] + 0.00325324 * s[:, 0, 2] + 0.00325324 * s[:, 1, 2]

        return grad

    def equations_of_motion_batch(self, s: cp_array, r: float,
                                  noise_decay_x: cp_array, noise_decay_y: cp_array,
                                  noise_dephasing: cp_array,
                                  add_t1: bool, add_t2: bool) -> cp_array:
        """
        Batched TWA equations of motion with dissipation.

        Args:
            s: Spin configurations (n_traj, 4, 3)
            r: Bond distance
            noise_*: Noise arrays
            add_t1, add_t2: Include dissipation

        Returns:
            ds/dt: Time derivatives (n_traj, 4, 3)
        """
        xp = self.xp

        # Coherent evolution: ds/dt = 2 * s × ∇H
        H_grad = self.hamiltonian_gradient_batch(r, s)
        coherent = 2.0 * xp.cross(s, H_grad)

        dsdt = coherent.copy()

        # T1 decay dissipation
        if add_t1:
            gamma = self.gamma_decay
            dsdt[:, :, 0] += (gamma / 2) * s[:, :, 0] * s[:, :, 2] + noise_decay_x * s[:, :, 2]
            dsdt[:, :, 1] += (gamma / 2) * s[:, :, 1] * s[:, :, 2] + noise_decay_y * s[:, :, 2]
            dsdt[:, :, 2] += -(gamma / 2) * (s[:, :, 0]**2 + s[:, :, 1]**2) - \
                            (noise_decay_x * s[:, :, 0] + noise_decay_y * s[:, :, 1])

        # T2 dephasing dissipation
        if add_t2:
            dsdt[:, :, 0] += 2 * noise_dephasing * s[:, :, 1]
            dsdt[:, :, 1] += -2 * noise_dephasing * s[:, :, 0]

        return dsdt

    def discrete_sample_initial_batch(self, initial_state: str = 'ground') -> cp_array:
        """
        Sample initial spin configurations (batched).

        Args:
            initial_state: 'ground', 'excited', or 'superposition'

        Returns:
            Spin configurations (n_traj, 4, 3)
        """
        xp = self.xp
        spins = xp.zeros((self.n_trajectories, self.n_qubits, 3))

        # Random x and y components
        spins[:, :, 0] = xp.random.choice([-1, 1], size=(self.n_trajectories, self.n_qubits))
        spins[:, :, 1] = xp.random.choice([-1, 1], size=(self.n_trajectories, self.n_qubits))

        # Z component based on initial state
        if initial_state == 'ground':
            spins[:, :, 2] = -1
        elif initial_state == 'excited':
            spins[:, :, 2] = 1
        elif initial_state == 'superposition':
            spins[:, :, 2] = xp.random.choice([-1, 1], size=(self.n_trajectories, self.n_qubits))

        return spins

    def rk4_step_batch(self, s: cp_array, dt: float, r: float,
                      noise_decay_x: cp_array, noise_decay_y: cp_array,
                      noise_dephasing: cp_array,
                      add_t1: bool, add_t2: bool) -> cp_array:
        """Batched RK4 integration step."""
        k1 = self.equations_of_motion_batch(s, r, noise_decay_x, noise_decay_y,
                                           noise_dephasing, add_t1, add_t2)
        k2 = self.equations_of_motion_batch(s + dt*k1/2, r, noise_decay_x, noise_decay_y,
                                           noise_dephasing, add_t1, add_t2)
        k3 = self.equations_of_motion_batch(s + dt*k2/2, r, noise_decay_x, noise_decay_y,
                                           noise_dephasing, add_t1, add_t2)
        k4 = self.equations_of_motion_batch(s + dt*k3, r, noise_decay_x, noise_decay_y,
                                           noise_dephasing, add_t1, add_t2)

        s_new = s + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return s_new

    def simulate_dynamics_gpu(self, r: float, total_time: float, dt: float = 0.01,
                             add_t1: bool = True, add_t2: bool = True,
                             initial_state: str = 'ground') -> Dict:
        """
        GPU-accelerated TWA dynamics simulation.

        Args:
            r: Bond distance
            total_time: Total simulation time
            dt: Time step
            add_t1, add_t2: Include dissipation
            initial_state: Initial state

        Returns:
            Dictionary with results
        """
        xp = self.xp
        n_steps = int(total_time / dt)
        times = np.linspace(0, total_time, n_steps)

        print(f"\nSimulating LiH dynamics (GPU)...")
        print(f"  Steps: {n_steps}")
        print(f"  Trajectories: {self.n_trajectories}")
        print(f"  T1: {add_t1}, T2: {add_t2}")

        start_time = time.time()

        # Initialize all trajectories on GPU
        s = self.discrete_sample_initial_batch(initial_state)

        # Storage (keep on GPU until end)
        avg_energies = xp.zeros(n_steps)
        avg_spins = xp.zeros((n_steps, self.n_qubits, 3))

        # Noise variance
        if add_t1 or add_t2:
            sigma_decay = xp.sqrt(max(self.gamma_decay / dt, 1e-20))
            sigma_dephasing = xp.sqrt(max(self.kappa_dephasing / dt, 1e-20))
        else:
            sigma_decay = 0.0
            sigma_dephasing = 0.0

        for step in range(n_steps):
            # Compute observables
            energies = self.build_lih_classical_hamiltonian_batch(r, s)
            avg_energies[step] = xp.mean(energies)
            avg_spins[step] = xp.mean(s, axis=0)

            if step < n_steps - 1:
                # Generate noise
                if add_t1:
                    noise_decay_x = xp.random.normal(0, sigma_decay,
                                                     (self.n_trajectories, self.n_qubits))
                    noise_decay_y = xp.random.normal(0, sigma_decay,
                                                     (self.n_trajectories, self.n_qubits))
                else:
                    noise_decay_x = xp.zeros((self.n_trajectories, self.n_qubits))
                    noise_decay_y = xp.zeros((self.n_trajectories, self.n_qubits))

                if add_t2:
                    noise_dephasing = xp.random.normal(0, sigma_dephasing,
                                                       (self.n_trajectories, self.n_qubits))
                else:
                    noise_dephasing = xp.zeros((self.n_trajectories, self.n_qubits))

                # RK4 step (all trajectories in parallel)
                s = self.rk4_step_batch(s, dt, r, noise_decay_x, noise_decay_y,
                                       noise_dephasing, add_t1, add_t2)

            if (step + 1) % 500 == 0:
                print(f"  Step {step + 1}/{n_steps}")

        end_time = time.time()
        elapsed = end_time - start_time

        print(f"  Completed in {elapsed:.2f} seconds")
        print(f"  Speed: {n_steps * self.n_trajectories / elapsed:.0f} traj-steps/sec")

        # Transfer results to CPU
        avg_energies_cpu = self.to_cpu(avg_energies)
        avg_spins_cpu = self.to_cpu(avg_spins)
        magnetization = np.sum(avg_spins_cpu[:, :, 2], axis=1)

        return {
            'times': times,
            'avg_spins': avg_spins_cpu,
            'avg_energies': avg_energies_cpu,
            'magnetization': magnetization,
            'r': r,
            'dt': dt,
            'n_trajectories': self.n_trajectories,
            'elapsed_time': elapsed
        }

    def compare_with_ideal(self, r: float = 1.5949, total_time: float = 20.0,
                          dt: float = 0.01) -> Dict:
        """Compare ideal vs dissipative dynamics."""
        print("\n" + "=" * 70)
        print("GPU-ACCELERATED LiH TWA COMPARISON")
        print("=" * 70)

        results = {}

        # Ideal
        print("\n1. Ideal (no dissipation)")
        results['ideal'] = self.simulate_dynamics_gpu(r, total_time, dt, False, False)

        # T1 only
        print("\n2. T1 decay only")
        results['t1_only'] = self.simulate_dynamics_gpu(r, total_time, dt, True, False)

        # T2 only
        print("\n3. T2 dephasing only")
        results['t2_only'] = self.simulate_dynamics_gpu(r, total_time, dt, False, True)

        # Full
        print("\n4. Full dissipation (T1 + T2)")
        results['full'] = self.simulate_dynamics_gpu(r, total_time, dt, True, True)

        results['r'] = r
        return results


if __name__ == "__main__":
    print("=" * 70)
    print("GPU-ACCELERATED LiH TWA SIMULATION")
    print("=" * 70)

    if not GPU_AVAILABLE:
        print("\nERROR: CuPy not available. Install with:")
        print("  pip install cupy-cuda12x  # for CUDA 12.x")
        print("  pip install cupy-cuda11x  # for CUDA 11.x")
        exit(1)

    # Create simulator
    lih_gpu = CUDAQ_LiH_TWA_Simulator(n_trajectories=2000, use_gpu=True)

    # Run comparison
    r_eq = 1.5949
    comparison = lih_gpu.compare_with_ideal(r=r_eq, total_time=20.0, dt=0.01)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    total_time = sum(r['elapsed_time'] for r in
                    [comparison['ideal'], comparison['t1_only'],
                     comparison['t2_only'], comparison['full']])

    print(f"\nTotal computation time: {total_time:.2f} seconds")
    print(f"Average time per simulation: {total_time / 4:.2f} seconds")
    print()
    print(f"Final energies:")
    print(f"  Ideal:      {comparison['ideal']['avg_energies'][-1]:.6f} H")
    print(f"  T1 only:    {comparison['t1_only']['avg_energies'][-1]:.6f} H")
    print(f"  T2 only:    {comparison['t2_only']['avg_energies'][-1]:.6f} H")
    print(f"  T1 + T2:    {comparison['full']['avg_energies'][-1]:.6f} H")
    print()
    print(f"Dissipation effects:")
    ideal_E = comparison['ideal']['avg_energies'][-1]
    print(f"  ΔE (T1):    {comparison['t1_only']['avg_energies'][-1] - ideal_E:.6e} H")
    print(f"  ΔE (T2):    {comparison['t2_only']['avg_energies'][-1] - ideal_E:.6e} H")
    print(f"  ΔE (Total): {comparison['full']['avg_energies'][-1] - ideal_E:.6e} H")

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    times = comparison['ideal']['times']

    # Energy
    ax1 = axes[0, 0]
    ax1.plot(times, comparison['ideal']['avg_energies'], 'b-', lw=2, label='Ideal')
    ax1.plot(times, comparison['t1_only']['avg_energies'], 'g--', lw=2, label='T1')
    ax1.plot(times, comparison['t2_only']['avg_energies'], 'orange', lw=2, label='T2')
    ax1.plot(times, comparison['full']['avg_energies'], 'r:', lw=2, label='T1+T2')
    ax1.set_xlabel('Time (a.u.)')
    ax1.set_ylabel('Energy (H)')
    ax1.set_title('Energy Evolution (GPU)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Magnetization
    ax2 = axes[0, 1]
    ax2.plot(times, comparison['ideal']['magnetization'], 'b-', lw=2, label='Ideal')
    ax2.plot(times, comparison['full']['magnetization'], 'r:', lw=2, label='T1+T2')
    ax2.set_xlabel('Time (a.u.)')
    ax2.set_ylabel('Total Magnetization')
    ax2.set_title('Magnetization')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Energy difference
    ax3 = axes[1, 0]
    ideal_E = comparison['ideal']['avg_energies']
    ax3.plot(times, comparison['full']['avg_energies'] - ideal_E, 'r-', lw=2)
    ax3.set_xlabel('Time (a.u.)')
    ax3.set_ylabel('ΔE from Ideal (H)')
    ax3.set_title('Dissipation Energy Shift')
    ax3.grid(True, alpha=0.3)

    # Individual qubits
    ax4 = axes[1, 1]
    for i in range(4):
        ax4.plot(times, comparison['full']['avg_spins'][:, i, 2], lw=2, label=f'Q{i}')
    ax4.set_xlabel('Time (a.u.)')
    ax4.set_ylabel('⟨σᶻ⟩')
    ax4.set_title('Individual Qubits (Full Diss.)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'GPU LiH TWA (R={r_eq:.4f} Å, {lih_gpu.n_trajectories} traj)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lih_gpu_twa.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: lih_gpu_twa.png")
    plt.show()

    print("\n" + "=" * 70)
    print("GPU simulation completed!")
    print("=" * 70)
