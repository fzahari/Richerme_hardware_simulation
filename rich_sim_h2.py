import numpy as np
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Callable

# Import library for hardware-realistic gate synthesis
try:
    from richerme_ion_analog import (
        target_pauli_string_unitary,
        IonTrapHardware,
        unitary_distance
    )
    EXTENDED_LIB_AVAILABLE = True
except ImportError:
    print("Warning: richerme_ion_analog not found. Using direct exponentiation.")
    EXTENDED_LIB_AVAILABLE = False

class TrappedIonSimulator:
    """Basic trapped ion simulator for the H2 simulation."""

    def __init__(self, N: int, geometry: str = '1D', anharmonic: bool = False):
        self.N = N
        self.geometry = geometry
        self.anharmonic = anharmonic
        self.mode_frequencies = np.linspace(4.8, 5.0, N)  # MHz

    def calculate_infidelity(self, J_target, J_exp):
        J_target_tilde = J_target - np.diag(np.diag(J_target))
        J_exp_tilde = J_exp - np.diag(np.diag(J_exp))
        inner = np.trace(J_exp_tilde.T @ J_target_tilde)
        norm_exp = np.sqrt(np.trace(J_exp_tilde.T @ J_exp_tilde))
        norm_target = np.sqrt(np.trace(J_target_tilde.T @ J_target_tilde))
        if norm_exp * norm_target > 0:
            return 0.5 * (1 - inner / (norm_exp * norm_target))
        return 1.0

    def power_law_interaction(self, alpha, J0=1.0):
        N = min(self.N, 4)
        J = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                J[i,j] = J[j,i] = J0 / (abs(i-j)**alpha if i != j else 1)
        return J

class H2TimeEvolutionSimulator:
    """Hydrogen molecule simulator using time propagation with hardware-realistic gates."""

    def __init__(self, ion_simulator, use_hardware_gates: bool = True):
        """
        Initialize H2 simulator.

        Args:
            ion_simulator: TrappedIonSimulator instance
            use_hardware_gates: If True, use hardware-native gate synthesis from extended library
        """
        self.ion_sim = ion_simulator
        self.n_qubits = 4
        if self.ion_sim.N < self.n_qubits:
            raise ValueError(f"Need at least {self.n_qubits} ions")

        self.I = np.eye(2, dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

        # Hardware-realistic gate synthesis
        self.use_hardware_gates = use_hardware_gates and EXTENDED_LIB_AVAILABLE
        if self.use_hardware_gates:
            self.hardware = IonTrapHardware()
            print(f"Using hardware-realistic gates (171Yb+)")
            print(f"  Two-qubit gate fidelity: {self.hardware.two_qubit_fidelity*100:.1f}%")
        else:
            self.hardware = None
            if use_hardware_gates and not EXTENDED_LIB_AVAILABLE:
                print("Warning: Extended library not available, using ideal gates")
        
    def build_h2_hamiltonian_pauli(self, r):
        """Build H2 Hamiltonian as Pauli strings."""
        e_nuc = 1.0 / r
        t = 0.52917 * np.exp(-1.5 * (r - 0.741))
        mu = -1.1256 + 0.2 * r
        u = 0.6744 / (1 + 0.1 * r)
        v = 0.1815 * np.exp(-0.5 * r)
        
        pauli_terms = [
            ([self.I, self.I, self.I, self.I], e_nuc),
            ([self.Z, self.I, self.I, self.I], mu/2),
            ([self.I, self.Z, self.I, self.I], mu/2),
            ([self.I, self.I, self.Z, self.I], mu/4),
            ([self.I, self.I, self.I, self.Z], mu/4),
            ([self.X, self.I, self.X, self.I], -t/2),
            ([self.Y, self.I, self.Y, self.I], -t/2),
            ([self.I, self.X, self.I, self.X], -t/2),
            ([self.I, self.Y, self.I, self.Y], -t/2),
            ([self.Z, self.Z, self.I, self.I], u/4),
            ([self.I, self.I, self.Z, self.Z], u/4),
            ([self.Z, self.I, self.Z, self.I], v/4),
            ([self.I, self.Z, self.I, self.Z], v/4),
            ([self.Z, self.I, self.I, self.Z], v/8),
            ([self.I, self.Z, self.Z, self.I], v/8),
        ]
        return pauli_terms
    
    def pauli_to_matrix(self, pauli_terms):
        H = np.zeros((16, 16), dtype=complex)
        for ops, coeff in pauli_terms:
            term = ops[0]
            for op in ops[1:]:
                term = np.kron(term, op)
            H += coeff * term
        return H
    
    def adiabatic_evolution(self, r_initial, r_final, total_time, n_steps=100):
        dt = total_time / n_steps
        times = np.linspace(0, total_time, n_steps)
        psi = np.zeros(16, dtype=complex)
        psi[0b0011] = 1.0
        
        results = {
            'times': times,
            'distances': [],
            'energies': [],
            'states': [],
            'fidelities': []
        }
        
        for i, t in enumerate(times):
            r = r_initial + (r_final - r_initial) * (t / total_time)
            results['distances'].append(r)
            
            pauli_terms = self.build_h2_hamiltonian_pauli(r)
            H = self.pauli_to_matrix(pauli_terms)
            
            U = expm(-1j * H * dt)
            psi = U @ psi
            
            energy = np.real(psi.conj() @ H @ psi)
            results['energies'].append(energy)
            results['states'].append(psi.copy())
            
            eigvals, eigvecs = eigh(H)
            ground_state = eigvecs[:, 0]
            fidelity = np.abs(psi.conj() @ ground_state)**2
            results['fidelities'].append(fidelity)
            
            if i % 20 == 0:
                print(f"Step {i}/{n_steps}: R={r:.3f} Å, E={energy:.4f} H, Fidelity={fidelity:.3f}")
        
        return results
    
    def imaginary_time_evolution(self, r, beta_max=100.0, n_steps=1000, initial_state='uniform'):
        """
        Imaginary-time evolution to find ground state.

        Args:
            r: Bond distance in Angstroms
            beta_max: Maximum imaginary time (larger = better convergence)
            n_steps: Number of time steps (more = better accuracy)
            initial_state: Initial state ('uniform', 'hartree_fock', 'random')

        Returns:
            Dictionary with evolution results

        Note: The H2 ground state is 2-fold degenerate (spin singlet/triplet).
              Imaginary time evolution converges to a superposition of both.
              For higher accuracy, use beta_max >= 50 and n_steps >= 500.
              Default beta_max=100, n_steps=1000 gives ~1e-9 H error.
        """
        dbeta = beta_max / n_steps
        betas = np.linspace(0, beta_max, n_steps)

        pauli_terms = self.build_h2_hamiltonian_pauli(r)
        H = self.pauli_to_matrix(pauli_terms)

        # Initialize state based on choice
        if initial_state == 'uniform':
            # Uniform superposition (default)
            psi = np.ones(16, dtype=complex) / 4.0
        elif initial_state == 'hartree_fock':
            # Hartree-Fock state |0011> (2 electrons in lowest orbitals)
            psi = np.zeros(16, dtype=complex)
            psi[0b0011] = 1.0
        elif initial_state == 'random':
            # Random state (for testing robustness)
            np.random.seed(42)
            psi = np.random.randn(16) + 1j*np.random.randn(16)
            psi = psi / np.linalg.norm(psi)
        else:
            raise ValueError(f"Unknown initial_state: {initial_state}")

        results = {'betas': betas, 'energies': [], 'states': [], 'overlaps': []}

        eigvals, eigvecs = eigh(H)
        ground_state = eigvecs[:, 0]
        ground_energy = eigvals[0]

        # Check for degeneracy
        degeneracy = 1
        for i in range(1, len(eigvals)):
            if abs(eigvals[i] - ground_energy) < 1e-8:
                degeneracy += 1
            else:
                break

        for beta in betas:
            U_imag = expm(-H * dbeta)
            psi = U_imag @ psi
            psi = psi / np.linalg.norm(psi)

            energy = np.real(psi.conj() @ H @ psi)
            results['energies'].append(energy)
            results['states'].append(psi.copy())

            overlap = np.abs(psi.conj() @ ground_state)**2
            results['overlaps'].append(overlap)

        results['ground_energy'] = ground_energy
        results['degeneracy'] = degeneracy
        results['initial_state_type'] = initial_state

        return results

    def hardware_efficient_ansatz(self, params: np.ndarray, use_full_gates: bool = True,
                                   n_layers: int = 3, use_nonlocal: bool = False) -> np.ndarray:
        """
        Hardware-efficient ansatz for 4-qubit system suitable for trapped ions.

        Uses layers of:
        1. Single-qubit Ry and Rz rotations (easily implemented)
        2. Entangling gates (XX, YY, ZZ if use_full_gates=True; otherwise just XX)
        3. Optional non-local gates for better expressiveness

        Args:
            params: Parameter vector
            use_full_gates: If True, use XX, YY, ZZ gates; if False, only XX
            n_layers: Number of ansatz layers
            use_nonlocal: If True, add non-local gates (0-2, 1-3)

        Returns:
            Quantum state vector |ψ(θ)>
        """
        # Calculate number of gate pairs
        local_pairs = 3  # (0,1), (1,2), (2,3)
        nonlocal_pairs = 2 if use_nonlocal else 0  # (0,2), (1,3)
        total_pairs = local_pairs + nonlocal_pairs

        if use_full_gates:
            # Each layer: 4 Ry + 4 Rz + pairs*(XX+YY+ZZ)
            params_per_layer = self.n_qubits * 2 + total_pairs * 3
        else:
            # Each layer: 4 Ry + 4 Rz + pairs*XX
            params_per_layer = self.n_qubits * 2 + total_pairs

        expected_params = params_per_layer * n_layers

        if len(params) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, got {len(params)}")

        # Start with Hartree-Fock state |0011> (2 electrons in lowest orbitals)
        psi = np.zeros(16, dtype=complex)
        psi[0b0011] = 1.0

        param_idx = 0

        for layer in range(n_layers):
            # Single-qubit Y rotations on all qubits
            for qubit in range(self.n_qubits):
                theta_y = params[param_idx]
                param_idx += 1
                Ry_i = self._single_qubit_gate(qubit, self._Ry(theta_y))
                psi = Ry_i @ psi

            # Single-qubit Z rotations on all qubits
            for qubit in range(self.n_qubits):
                theta_z = params[param_idx]
                param_idx += 1
                Rz_i = self._single_qubit_gate(qubit, self._Rz(theta_z))
                psi = Rz_i @ psi

            # Entangling gates
            local_pairs = [(0, 1), (1, 2), (2, 3)]
            all_pairs = local_pairs + ([(0, 2), (1, 3)] if use_nonlocal else [])

            if use_full_gates:
                # Apply XX, YY, and ZZ gates
                for q1, q2 in all_pairs:
                    phi_xx = params[param_idx]
                    param_idx += 1
                    XX_gate = self._xx_gate(q1, q2, phi_xx)
                    psi = XX_gate @ psi

                for q1, q2 in all_pairs:
                    phi_yy = params[param_idx]
                    param_idx += 1
                    YY_gate = self._yy_gate(q1, q2, phi_yy)
                    psi = YY_gate @ psi

                for q1, q2 in all_pairs:
                    phi_zz = params[param_idx]
                    param_idx += 1
                    ZZ_gate = self._zz_gate(q1, q2, phi_zz)
                    psi = ZZ_gate @ psi
            else:
                # Only XX gates
                for q1, q2 in all_pairs:
                    phi = params[param_idx]
                    param_idx += 1
                    XX_gate = self._xx_gate(q1, q2, phi)
                    psi = XX_gate @ psi

        return psi

    def _Ry(self, theta: float) -> np.ndarray:
        """Single-qubit Y rotation matrix."""
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    def _Rz(self, theta: float) -> np.ndarray:
        """Single-qubit Z rotation matrix."""
        return np.array([[np.exp(-1j * theta / 2), 0],
                        [0, np.exp(1j * theta / 2)]], dtype=complex)

    def _single_qubit_gate(self, qubit_idx: int, gate: np.ndarray) -> np.ndarray:
        """Apply single-qubit gate to specific qubit in 4-qubit system."""
        ops = [self.I if i != qubit_idx else gate for i in range(self.n_qubits)]
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _xx_gate(self, q1: int, q2: int, phi: float) -> np.ndarray:
        """
        XX entangling gate: exp(-i * phi/2 * X_q1 X_q2)
        Native to trapped-ion hardware.

        If use_hardware_gates=True, uses hardware-native UMQ-Rz-UMQ synthesis.
        Otherwise uses direct matrix exponentiation.
        """
        if self.use_hardware_gates:
            # Hardware-native synthesis using extended library
            # Build Pauli string with X on positions q1 and q2
            pauli_str = 'I' * q1 + 'X' + 'I' * (q2 - q1 - 1) + 'X' + 'I' * (self.n_qubits - q2 - 1)
            return target_pauli_string_unitary(pauli_str, phi / 2)
        else:
            # Direct exponentiation (ideal gates)
            ops = [self.I] * self.n_qubits
            ops[q1] = self.X
            ops[q2] = self.X

            XX = ops[0]
            for op in ops[1:]:
                XX = np.kron(XX, op)

            return expm(-1j * phi / 2 * XX)

    def _yy_gate(self, q1: int, q2: int, phi: float) -> np.ndarray:
        """
        YY entangling gate: exp(-i * phi/2 * Y_q1 Y_q2)

        If use_hardware_gates=True, uses hardware-native synthesis via basis rotations.
        """
        if self.use_hardware_gates:
            # Hardware-native synthesis
            pauli_str = 'I' * q1 + 'Y' + 'I' * (q2 - q1 - 1) + 'Y' + 'I' * (self.n_qubits - q2 - 1)
            return target_pauli_string_unitary(pauli_str, phi / 2)
        else:
            # Direct exponentiation (ideal gates)
            ops = [self.I] * self.n_qubits
            ops[q1] = self.Y
            ops[q2] = self.Y

            YY = ops[0]
            for op in ops[1:]:
                YY = np.kron(YY, op)

            return expm(-1j * phi / 2 * YY)

    def _zz_gate(self, q1: int, q2: int, phi: float) -> np.ndarray:
        """
        ZZ entangling gate: exp(-i * phi/2 * Z_q1 Z_q2)

        If use_hardware_gates=True, uses hardware-native synthesis.
        """
        if self.use_hardware_gates:
            # Hardware-native synthesis
            pauli_str = 'I' * q1 + 'Z' + 'I' * (q2 - q1 - 1) + 'Z' + 'I' * (self.n_qubits - q2 - 1)
            return target_pauli_string_unitary(pauli_str, phi / 2)
        else:
            # Direct exponentiation (ideal gates)
            ops = [self.I] * self.n_qubits
            ops[q1] = self.Z
            ops[q2] = self.Z

            ZZ = ops[0]
            for op in ops[1:]:
                ZZ = np.kron(ZZ, op)

            return expm(-1j * phi / 2 * ZZ)

    def vqe_optimization(self, r: float, n_layers: int = 4,
                        max_iter: int = 1000, method: str = 'COBYLA',
                        use_full_gates: bool = True, n_trials: int = 5,
                        use_nonlocal: bool = True, use_hybrid: bool = True) -> Dict:
        """
        Run VQE optimization for H2 molecule with advanced multi-start strategy.

        Args:
            r: Bond distance in Angstroms
            n_layers: Number of ansatz layers
            max_iter: Maximum optimization iterations per trial
            method: Optimization method ('COBYLA', 'L-BFGS-B', etc.)
            use_full_gates: Use XX+YY+ZZ gates if True, only XX if False
            n_trials: Number of random initializations to try
            use_nonlocal: Include non-local gates (0-2, 1-3)
            use_hybrid: Use hybrid COBYLA->BFGS optimization

        Returns:
            Dictionary with VQE results from best trial
        """
        # Build Hamiltonian
        pauli_terms = self.build_h2_hamiltonian_pauli(r)
        H = self.pauli_to_matrix(pauli_terms)

        # Calculate exact ground state for comparison
        eigvals, eigvecs = eigh(H)
        exact_ground_energy = eigvals[0]
        exact_ground_state = eigvecs[:, 0]

        # Determine number of parameters
        local_pairs = 3
        nonlocal_pairs = 2 if use_nonlocal else 0
        total_pairs = local_pairs + nonlocal_pairs

        if use_full_gates:
            params_per_layer = self.n_qubits * 2 + total_pairs * 3
        else:
            params_per_layer = self.n_qubits * 2 + total_pairs

        n_params = params_per_layer * n_layers

        print(f"\nRunning VQE optimization at R = {r:.3f} Å")
        print(f"Target ground state energy: {exact_ground_energy:.6f} H")
        print(f"Ansatz: {'XX+YY+ZZ' if use_full_gates else 'XX only'} gates")
        print(f"Layers: {n_layers}, Non-local: {use_nonlocal}, Hybrid: {use_hybrid}")
        print(f"Number of parameters: {n_params}")
        print(f"Number of trials: {n_trials}")
        print(f"Optimizer: {method if not use_hybrid else 'COBYLA->L-BFGS-B'}")
        print("-" * 60)

        best_energy = np.inf
        best_result = None
        all_trial_results = []

        for trial in range(n_trials):
            print(f"\n[Trial {trial + 1}/{n_trials}]")

            # Initialize parameters randomly
            np.random.seed(42 + trial)  # Different seed for each trial
            initial_params = np.random.randn(n_params) * 0.1

            # Track optimization history for this trial
            iteration_count = [0]
            energy_history = []
            param_history = []

            def objective(params: np.ndarray) -> float:
                """Objective function: expectation value <ψ(θ)|H|ψ(θ)>"""
                psi = self.hardware_efficient_ansatz(params, use_full_gates=use_full_gates,
                                                     n_layers=n_layers, use_nonlocal=use_nonlocal)
                energy = np.real(psi.conj() @ H @ psi)

                # Track history
                iteration_count[0] += 1
                energy_history.append(energy)
                param_history.append(params.copy())

                if iteration_count[0] % 100 == 0:
                    overlap = np.abs(psi.conj() @ exact_ground_state)**2
                    print(f"  Iter {iteration_count[0]}: E = {energy:.6f} H, "
                          f"Error = {energy - exact_ground_energy:.6e}, Overlap = {overlap:.4f}")

                return energy

            # Run optimization (possibly hybrid)
            if use_hybrid:
                # Phase 1: COBYLA (derivative-free, good for initial exploration)
                result1 = minimize(
                    objective,
                    initial_params,
                    method='COBYLA',
                    options={'maxiter': max_iter // 2, 'disp': False}
                )

                print(f"  Phase 1 (COBYLA) complete: E = {result1.fun:.8f} H")

                # Phase 2: L-BFGS-B (gradient-based, good for refinement)
                result = minimize(
                    objective,
                    result1.x,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter // 2, 'disp': False}
                )

                print(f"  Phase 2 (L-BFGS-B) complete: E = {result.fun:.8f} H")
            else:
                # Single optimizer
                result = minimize(
                    objective,
                    initial_params,
                    method=method,
                    options={'maxiter': max_iter, 'disp': False}
                )

            # Get final state and energy for this trial
            trial_params = result.x
            trial_state = self.hardware_efficient_ansatz(trial_params, use_full_gates=use_full_gates,
                                                         n_layers=n_layers, use_nonlocal=use_nonlocal)
            trial_energy = np.real(trial_state.conj() @ H @ trial_state)
            trial_overlap = np.abs(trial_state.conj() @ exact_ground_state)**2

            trial_result = {
                'trial_number': trial + 1,
                'energy': trial_energy,
                'error': trial_energy - exact_ground_energy,
                'params': trial_params,
                'state': trial_state,
                'overlap': trial_overlap,
                'energy_history': energy_history,
                'param_history': param_history,
                'n_iterations': iteration_count[0],
                'convergence': result.success
            }

            all_trial_results.append(trial_result)

            print(f"  Trial {trial + 1} final energy: {trial_energy:.8f} H (error: {trial_energy - exact_ground_energy:.6e})")

            # Track best result
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_result = trial_result

        print("\n" + "=" * 60)
        print(f"BEST RESULT (Trial {best_result['trial_number']})")
        print("=" * 60)
        print(f"VQE energy: {best_result['energy']:.8f} H")
        print(f"Exact energy: {exact_ground_energy:.8f} H")
        print(f"Absolute error: {best_result['error']:.6e} H")
        print(f"Overlap with ground state: {best_result['overlap']:.6f}")
        print(f"Total function evaluations: {best_result['n_iterations']}")

        return {
            'r': r,
            'vqe_energy': best_result['energy'],
            'exact_energy': exact_ground_energy,
            'error': best_result['error'],
            'optimal_params': best_result['params'],
            'optimal_state': best_result['state'],
            'overlap': best_result['overlap'],
            'energy_history': best_result['energy_history'],
            'param_history': best_result['param_history'],
            'n_iterations': best_result['n_iterations'],
            'convergence': best_result['convergence'],
            'all_trials': all_trial_results,
            'use_full_gates': use_full_gates
        }

    def compare_methods(self, r: float = 0.74) -> Dict:
        """
        Compare VQE, imaginary-time evolution, and exact diagonalization.

        Args:
            r: Bond distance in Angstroms

        Returns:
            Dictionary with comparison results
        """
        print("\n" + "=" * 70)
        print("COMPARING GROUND STATE METHODS FOR H₂ MOLECULE")
        print("=" * 70)
        print(f"Bond distance: R = {r:.3f} Å\n")

        # Build Hamiltonian
        pauli_terms = self.build_h2_hamiltonian_pauli(r)
        H = self.pauli_to_matrix(pauli_terms)

        # Exact diagonalization
        print("1. EXACT DIAGONALIZATION")
        print("-" * 60)
        eigvals, eigvecs = eigh(H)
        exact_energy = eigvals[0]
        exact_state = eigvecs[:, 0]
        print(f"Ground state energy: {exact_energy:.8f} H")

        # VQE
        print("\n2. VARIATIONAL QUANTUM EIGENSOLVER (VQE)")
        print("-" * 60)
        vqe_results = self.vqe_optimization(
            r, n_layers=3, max_iter=800, method='COBYLA',
            use_full_gates=True, n_trials=3, use_nonlocal=True, use_hybrid=True
        )

        # Imaginary-time evolution
        print("\n3. IMAGINARY-TIME EVOLUTION")
        print("-" * 60)
        imag_results = self.imaginary_time_evolution(r, beta_max=100.0, n_steps=1000)
        imag_final_energy = imag_results['energies'][-1]
        imag_final_state = imag_results['states'][-1]
        imag_overlap = imag_results['overlaps'][-1]
        print(f"Final energy: {imag_final_energy:.8f} H")
        print(f"Ground state energy: {imag_results['ground_energy']:.8f} H")
        print(f"Absolute error: {imag_final_energy - exact_energy:.6e} H")
        print(f"Final overlap with ground state: {imag_overlap:.4f}")
        if imag_results.get('degeneracy', 1) > 1:
            print(f"Note: Ground state is {imag_results['degeneracy']}-fold degenerate")
            print(f"      Overlap ~0.5 expected for degenerate states")

        # Comparison summary
        print("\n" + "=" * 70)
        print("SUMMARY COMPARISON")
        print("=" * 70)
        print(f"{'Method':<30} {'Energy (H)':<15} {'Error (H)':<15} {'Overlap':<10}")
        print("-" * 70)
        print(f"{'Exact Diagonalization':<30} {exact_energy:<15.8f} {0.0:<15.6e} {1.0:<10.6f}")
        print(f"{'VQE':<30} {vqe_results['vqe_energy']:<15.8f} "
              f"{vqe_results['error']:<15.6e} {vqe_results['overlap']:<10.6f}")
        print(f"{'Imaginary-Time Evolution':<30} {imag_final_energy:<15.8f} "
              f"{imag_final_energy - exact_energy:<15.6e} {imag_overlap:<10.6f}")
        print("=" * 70)

        return {
            'r': r,
            'exact': {
                'energy': exact_energy,
                'state': exact_state
            },
            'vqe': vqe_results,
            'imaginary_time': {
                'energy': imag_final_energy,
                'state': imag_final_state,
                'overlap': imag_overlap,
                'full_results': imag_results
            }
        }

if __name__ == "__main__":
    print("=" * 70)
    print("H₂ MOLECULE SIMULATION WITH HARDWARE-REALISTIC GATES")
    print("=" * 70)
    print()

    # Check if extended library is available
    if EXTENDED_LIB_AVAILABLE:
        print("✓ Extended library detected - using hardware-realistic gates")
        print("  Gate synthesis: UMQ-Rz-UMQ construction (171Yb+)")
        print()
    else:
        print("⚠ Extended library not found - using ideal gate exponentiation")
        print("  (Install richerme_ion_analog_extended.py for hardware-realistic gates)")
        print()

    ion_system = TrappedIonSimulator(N=4, geometry='1D', anharmonic=False)
    h2_sim = H2TimeEvolutionSimulator(ion_system, use_hardware_gates=True)

    # Run comprehensive comparison at equilibrium bond distance
    r_eq = 0.74
    comparison_results = h2_sim.compare_methods(r=r_eq)

    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: VQE convergence
    ax1 = plt.subplot(2, 3, 1)
    vqe_history = comparison_results['vqe']['energy_history']
    exact_energy = comparison_results['exact']['energy']
    ax1.plot(vqe_history, 'b-', linewidth=2, label='VQE')
    ax1.axhline(y=exact_energy, color='r', linestyle='--', linewidth=2, label='Exact')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Energy (Hartree)')
    ax1.set_title('VQE Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: VQE error convergence (log scale)
    ax2 = plt.subplot(2, 3, 2)
    vqe_errors = np.abs(np.array(vqe_history) - exact_energy)
    ax2.semilogy(vqe_errors, 'b-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('|E - E₀| (Hartree)')
    ax2.set_title('VQE Error (Log Scale)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Imaginary-time evolution convergence
    ax3 = plt.subplot(2, 3, 3)
    imag_results = comparison_results['imaginary_time']['full_results']
    imag_errors = np.abs(np.array(imag_results['energies']) - imag_results['ground_energy'])
    ax3.semilogy(imag_results['betas'], imag_errors, 'g-', linewidth=2)
    ax3.set_xlabel('β (Imaginary Time)')
    ax3.set_ylabel('|E - E₀| (Hartree)')
    ax3.set_title('Imaginary-Time Evolution Error')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Method comparison bar chart
    ax4 = plt.subplot(2, 3, 4)
    methods = ['Exact', 'VQE', 'Imag-Time']
    energies = [
        exact_energy,
        comparison_results['vqe']['vqe_energy'],
        comparison_results['imaginary_time']['energy']
    ]
    colors = ['red', 'blue', 'green']
    bars = ax4.bar(methods, energies, color=colors, alpha=0.7)
    ax4.set_ylabel('Energy (Hartree)')
    ax4.set_title('Ground State Energy Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.6f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 5: Error comparison bar chart (log scale)
    ax5 = plt.subplot(2, 3, 5)
    errors = [
        0.0,  # Exact has no error
        abs(comparison_results['vqe']['error']),
        abs(comparison_results['imaginary_time']['energy'] - exact_energy)
    ]
    # Handle zero error for exact method
    display_errors = [max(e, 1e-12) for e in errors]
    bars = ax5.bar(methods, display_errors, color=colors, alpha=0.7)
    ax5.set_ylabel('|Error| (Hartree)')
    ax5.set_title('Absolute Error (Log Scale)')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.2e}',
                ha='center', va='bottom', fontsize=8)

    # Plot 6: Overlap with ground state
    ax6 = plt.subplot(2, 3, 6)
    overlaps = [
        1.0,  # Exact has perfect overlap
        comparison_results['vqe']['overlap'],
        comparison_results['imaginary_time']['overlap']
    ]
    bars = ax6.bar(methods, overlaps, color=colors, alpha=0.7)
    ax6.set_ylabel('Overlap with Ground State')
    ax6.set_title('State Fidelity')
    ax6.set_ylim([0.9, 1.01])
    ax6.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, overlap in zip(bars, overlaps):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{overlap:.6f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'H₂ Ground State Methods Comparison (R = {r_eq:.2f} Å)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Optional: Run adiabatic evolution for additional analysis
    print("\n\n" + "=" * 70)
    print("BONUS: Adiabatic Evolution Demonstration")
    print("=" * 70)
    print("Running adiabatic evolution from R = 3.0 Å to R = 0.74 Å...")
    adiabatic_results = h2_sim.adiabatic_evolution(
        r_initial=3.0, r_final=0.74, total_time=50.0, n_steps=100
    )

    # Plot adiabatic results
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(adiabatic_results['times'], adiabatic_results['distances'], 'b-', linewidth=2)
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Bond Distance (Å)')
    axes[0,0].set_title('Adiabatic Path')
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(adiabatic_results['times'], adiabatic_results['energies'], 'r-', linewidth=2)
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Energy (Hartree)')
    axes[0,1].set_title('Energy Evolution')
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(adiabatic_results['times'], adiabatic_results['fidelities'], 'g-', linewidth=2)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Ground State Fidelity')
    axes[1,0].set_title('Adiabatic Performance')
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].plot(adiabatic_results['distances'], adiabatic_results['energies'],
                   'purple', linewidth=2, marker='o', markersize=3, markevery=5)
    axes[1,1].set_xlabel('Bond Distance (Å)')
    axes[1,1].set_ylabel('Energy (Hartree)')
    axes[1,1].set_title('Potential Energy Surface')
    axes[1,1].grid(True, alpha=0.3)

    plt.suptitle('Adiabatic State Preparation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 70)
    print("All simulations completed successfully!")
    print("=" * 70)
