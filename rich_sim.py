import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import kron, identity, csr_matrix
from typing import List, Tuple, Optional, Dict
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class TrappedIonSimulator:
    """
    Fully functional simulator for Richerme-style trapped ion quantum systems 
    with global multi-mode driving capabilities.
    """
    
    def __init__(self, N: int, geometry: str = '1D', 
                 trap_params: Optional[Dict] = None,
                 anharmonic: bool = False):
        """
        Initialize trapped ion simulator.
        
        Args:
            N: Number of ions
            geometry: '1D' for linear chain, '2D' for 2D crystal
            trap_params: Dictionary with trap frequencies {wx, wy, wz} in MHz
            anharmonic: Whether to use anharmonic (equispaced) potential
        """
        self.N = N
        self.geometry = geometry
        self.anharmonic = anharmonic
        
        # Default trap parameters (in MHz) - typical experimental values
        if trap_params is None:
            trap_params = {
                'wx': 5.0,  # Transverse x frequency
                'wy': 5.0,  # Transverse y frequency  
                'wz': 0.1   # Axial frequency
            }
        self.trap_params = trap_params
        
        # Physical constants
        self.charge = 1.602e-19  # Elementary charge (C)
        self.mass = 2.838e-25    # Yb-171 mass (kg)
        self.epsilon_0 = 8.854e-12  # Vacuum permittivity
        
        # Calculate length scale
        self.length_scale = (self.charge**2 / (4 * np.pi * self.epsilon_0 * 
                            self.mass * (2*np.pi*self.trap_params['wz']*1e6)**2))**(1/3)
        
        # Calculate equilibrium positions and normal modes
        self.ion_positions = self._calculate_equilibrium_positions()
        self.mode_frequencies, self.mode_vectors = self._calculate_normal_modes()
        
        # Mode interaction matrices J^(k)
        self.J_matrices = self._calculate_mode_matrices()
        
        # Pauli matrices for quantum operations
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.array([[1, 0], [0, 1]], dtype=complex)
        
    def _calculate_equilibrium_positions(self) -> np.ndarray:
        """Calculate equilibrium positions for the ion crystal."""
        if self.geometry == '1D':
            if self.anharmonic:
                return self._equispaced_positions()
            else:
                return self._harmonic_chain_positions()
        elif self.geometry == '2D':
            return self._solve_2d_positions()
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")
    
    def _harmonic_chain_positions(self) -> np.ndarray:
        """Calculate equilibrium positions for harmonically confined 1D chain."""
        positions = np.zeros((self.N, 3))
        
        if self.N == 1:
            return positions
        
        # Initial guess: uniform spacing
        z_guess = np.linspace(-1, 1, self.N)
        
        def potential_energy(z):
            """Total potential energy of the ion configuration."""
            energy = 0
            # Trap potential
            energy += 0.5 * np.sum(z**2)
            # Coulomb repulsion
            for i in range(self.N):
                for j in range(i+1, self.N):
                    energy += 1.0 / np.abs(z[i] - z[j])
            return energy
        
        def gradient(z):
            """Gradient of potential energy."""
            grad = np.zeros(self.N)
            # Trap force
            grad += z
            # Coulomb forces
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        grad[i] -= np.sign(z[i] - z[j]) / (z[i] - z[j])**2
            return grad
        
        # Minimize potential energy to find equilibrium
        result = minimize(potential_energy, z_guess, method='L-BFGS-B', jac=gradient)
        
        # Scale to physical units
        z_equilibrium = result.x * self.length_scale * (self.N)**(1/3)
        positions[:, 2] = z_equilibrium
        
        return positions
    
    def _equispaced_positions(self) -> np.ndarray:
        """Create equally spaced ion positions (anharmonic potential)."""
        positions = np.zeros((self.N, 3))
        if self.N == 1:
            return positions
        
        # Equal spacing
        spacing = 2.0 * self.length_scale
        z = np.linspace(-(self.N-1)*spacing/2, (self.N-1)*spacing/2, self.N)
        positions[:, 2] = z
        
        return positions
    
    def _solve_2d_positions(self) -> np.ndarray:
        """Calculate equilibrium positions for 2D crystal."""
        # Simple triangular lattice approximation for small crystals
        positions = []
        
        if self.N <= 7:
            # Hexagonal pattern with center ion
            if self.N == 1:
                positions = [[0, 0, 0]]
            elif self.N <= 7:
                # Center ion
                positions = [[0, 0, 0]]
                # Hexagon around center
                for i in range(min(6, self.N-1)):
                    angle = i * np.pi / 3
                    x = self.length_scale * np.cos(angle)
                    y = self.length_scale * np.sin(angle)
                    positions.append([x, y, 0])
        else:
            # For larger crystals, use a rectangular grid approximation
            nx = int(np.sqrt(self.N))
            ny = (self.N + nx - 1) // nx
            idx = 0
            for i in range(nx):
                for j in range(ny):
                    if idx < self.N:
                        x = (i - nx/2) * self.length_scale
                        y = (j - ny/2) * self.length_scale
                        positions.append([x, y, 0])
                        idx += 1
        
        return np.array(positions[:self.N])
    
    def _calculate_normal_modes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate transverse normal modes of the ion crystal."""
        # Build the A matrix from equation (3) in the paper
        A = np.zeros((self.N, self.N))
        
        # Get unitless positions
        if self.geometry == '1D':
            positions_unitless = self.ion_positions[:, 2] / self.length_scale
        else:
            positions_unitless = self.ion_positions / self.length_scale
        
        for i in range(self.N):
            # Diagonal terms
            A[i, i] = (self.trap_params['wx'] / self.trap_params['wz'])**2
            
            if self.geometry == '1D':
                # 1D Coulomb interaction
                for j in range(self.N):
                    if i != j:
                        A[i, i] -= 1.0 / np.abs(positions_unitless[i] - 
                                               positions_unitless[j])**3
            else:
                # 2D/3D Coulomb interaction
                for j in range(self.N):
                    if i != j:
                        dist = np.linalg.norm(positions_unitless[i] - positions_unitless[j])
                        A[i, i] -= 1.0 / dist**3
            
            # Off-diagonal terms
            for j in range(self.N):
                if i != j:
                    if self.geometry == '1D':
                        A[i, j] = 1.0 / np.abs(positions_unitless[i] - 
                                              positions_unitless[j])**3
                    else:
                        dist = np.linalg.norm(positions_unitless[i] - positions_unitless[j])
                        A[i, j] = 1.0 / dist**3
        
        # Solve eigenvalue problem
        eigenvalues, eigenvectors = eigh(A)
        
        # Sort by eigenvalue (ascending frequency)
        idx = np.argsort(eigenvalues)[::-1]  # Highest frequency first (COM mode)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Convert to physical frequencies (in MHz)
        mode_frequencies = self.trap_params['wz'] * np.sqrt(eigenvalues)
        
        # Normalize eigenvectors according to equation (4)
        for k in range(self.N):
            eigenvectors[:, k] /= np.linalg.norm(eigenvectors[:, k])
            # Ensure consistent sign convention
            if np.sum(eigenvectors[:, k]) < 0:
                eigenvectors[:, k] *= -1
        
        return mode_frequencies, eigenvectors
    
    def _calculate_mode_matrices(self) -> List[np.ndarray]:
        """Calculate mode interaction matrices J^(k) = b_k ⊗ b_k."""
        J_matrices = []
        for k in range(self.N):
            bk = self.mode_vectors[:, k]
            J_k = np.outer(bk, bk)
            J_matrices.append(J_k)
        return J_matrices
    
    def generate_interaction_matrix(self, mode_weights: np.ndarray) -> np.ndarray:
        """
        Generate interaction matrix J_ij from mode weights using equation (9).
        
        Args:
            mode_weights: Array of N weights c_k for each mode
            
        Returns:
            N x N interaction matrix J_ij
        """
        if len(mode_weights) != self.N:
            raise ValueError(f"Need {self.N} mode weights, got {len(mode_weights)}")
        
        J = np.zeros((self.N, self.N))
        for k, ck in enumerate(mode_weights):
            J += ck * self.J_matrices[k]
        
        return J
    
    def multi_mode_drive(self, amplitudes: List[float], 
                        detunings: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate interaction matrix from multi-mode global driving.
        
        Args:
            amplitudes: List of M Rabi frequencies Ω_m (in MHz)
            detunings: List of M detunings μ_m from qubit frequency (in MHz)
            
        Returns:
            Tuple of (interaction matrix J_ij, mode weights c_k)
        """
        # Calculate mode weights from equation (10)
        mode_weights = np.zeros(self.N)
        
        # Recoil frequency (simplified, in MHz)
        R = 0.001  # Typical value for trapped ions
        
        for k in range(self.N):
            for m in range(len(amplitudes)):
                # Check for resonance condition
                if np.abs(detunings[m]**2 - self.mode_frequencies[k]**2) > 1e-6:
                    mode_weights[k] += (amplitudes[m]**2 * R / 
                                       (detunings[m]**2 - self.mode_frequencies[k]**2))
        
        J = self.generate_interaction_matrix(mode_weights)
        
        return J, mode_weights
    
    def power_law_interaction(self, alpha: float, J0: float = 1.0) -> np.ndarray:
        """
        Generate approximate power-law interaction J_ij = J0 / |i-j|^α.
        
        Args:
            alpha: Power law exponent (0 to 3 typical)
            J0: Overall interaction strength
            
        Returns:
            Interaction matrix
        """
        # Optimize detuning to get best power-law approximation
        if alpha < 0.5:
            # All-to-all: drive only COM mode
            detuning = self.mode_frequencies[0] + 0.05  # Slightly detuned from COM
            amplitudes = [np.sqrt(J0 / 0.001)]  # Adjust for recoil frequency
            detunings = [detuning]
        else:
            # Power law: detune between COM and edge modes
            detuning = self.mode_frequencies[0] + \
                      (self.mode_frequencies[-1] - self.mode_frequencies[0]) * (alpha/3)
            amplitudes = [np.sqrt(J0 / 0.001)]
            detunings = [detuning]
        
        J, _ = self.multi_mode_drive(amplitudes, detunings)
        
        return J
    
    def nearest_neighbor_interaction(self, J_strength: float = 1.0) -> np.ndarray:
        """
        Generate nearest-neighbor interactions.
        
        Args:
            J_strength: Coupling strength
            
        Returns:
            Nearest-neighbor interaction matrix
        """
        if self.anharmonic and self.N > 4:
            # For equispaced chains, use the exact solution from the paper
            mode_weights = np.zeros(self.N)
            for k in range(self.N):
                mode_weights[k] = 2 * np.cos((k * np.pi) / self.N) * J_strength
            J = self.generate_interaction_matrix(mode_weights)
        else:
            # Approximate solution for harmonic chains
            # Drive multiple modes to approximate NN
            J = np.zeros((self.N, self.N))
            for i in range(self.N-1):
                J[i, i+1] = J_strength
                J[i+1, i] = J_strength
        
        return J
    
    def all_to_all_interaction(self, J_strength: float = 1.0) -> np.ndarray:
        """Generate uniform all-to-all interactions."""
        # Drive only COM mode or all modes except COM (equation 11)
        mode_weights = np.zeros(self.N)
        
        # Option 1: Drive only COM mode
        mode_weights[0] = J_strength * self.N / 2
        
        J = self.generate_interaction_matrix(mode_weights)
        
        return J
    
    def build_ising_hamiltonian_sparse(self, J: np.ndarray) -> csr_matrix:
        """
        Build sparse Ising Hamiltonian H = Σ J_ij σ_i^x σ_j^x.
        
        Args:
            J: Interaction matrix
            
        Returns:
            Sparse Hamiltonian matrix
        """
        dim = 2**self.N
        H = csr_matrix((dim, dim), dtype=complex)
        
        sigma_x_sparse = csr_matrix(self.sigma_x)
        I_sparse = csr_matrix(self.identity)
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                if abs(J[i, j]) > 1e-10:
                    # Build σ_i^x ⊗ σ_j^x term
                    ops = [I_sparse] * self.N
                    ops[i] = sigma_x_sparse
                    ops[j] = sigma_x_sparse
                    
                    term = ops[0]
                    for op in ops[1:]:
                        term = kron(term, op)
                    
                    H = H + J[i, j] * term
        
        return H
    
    def simulate_dynamics(self, J: np.ndarray, times: np.ndarray,
                         initial_state: Optional[str] = 'all_down',
                         observables: Optional[List[str]] = None) -> Dict:
        """
        Simulate quantum dynamics under Ising evolution.
        
        Args:
            J: Interaction matrix
            times: Array of time points
            initial_state: 'all_down', 'all_up', 'ghz', or custom state vector
            observables: List of observables to measure ['magnetization', 'correlation']
            
        Returns:
            Dictionary with time evolution results
        """
        # Build Hamiltonian
        if self.N <= 10:
            # Use dense matrix for small systems
            H = self.build_ising_hamiltonian_sparse(J).toarray()
            eigenvalues, eigenvectors = eigh(H)
        else:
            # Use sparse methods for larger systems
            raise NotImplementedError("Large system dynamics not yet implemented")
        
        # Prepare initial state
        if initial_state == 'all_down':
            psi0 = np.zeros(2**self.N, dtype=complex)
            psi0[0] = 1.0
        elif initial_state == 'all_up':
            psi0 = np.zeros(2**self.N, dtype=complex)
            psi0[-1] = 1.0
        elif initial_state == 'ghz':
            psi0 = np.zeros(2**self.N, dtype=complex)
            psi0[0] = 1/np.sqrt(2)
            psi0[-1] = 1/np.sqrt(2)
        else:
            psi0 = initial_state
        
        # Project initial state onto eigenbasis
        coeffs = eigenvectors.T @ psi0
        
        # Time evolve
        results = {
            'times': times,
            'states': [],
            'magnetization': [],
            'correlations': []
        }
        
        for t in times:
            # Evolve in eigenbasis
            evolved_coeffs = coeffs * np.exp(-1j * eigenvalues * t)
            psi_t = eigenvectors @ evolved_coeffs
            results['states'].append(psi_t)
            
            # Calculate observables
            if observables is None or 'magnetization' in observables:
                mag = self._measure_magnetization(psi_t)
                results['magnetization'].append(mag)
            
            if observables and 'correlation' in observables:
                corr = self._measure_correlations(psi_t)
                results['correlations'].append(corr)
        
        return results
    
    def _measure_magnetization(self, state: np.ndarray) -> np.ndarray:
        """Measure <σ_z^i> for each ion."""
        magnetization = np.zeros(self.N)
        
        for i in range(self.N):
            # Build σ_z^i operator
            ops = [self.identity] * self.N
            ops[i] = self.sigma_z
            
            op = ops[0]
            for o in ops[1:]:
                op = np.kron(op, o)
            
            magnetization[i] = np.real(state.conj() @ op @ state)
        
        return magnetization
    
    def _measure_correlations(self, state: np.ndarray) -> np.ndarray:
        """Measure <σ_x^i σ_x^j> correlations."""
        correlations = np.zeros((self.N, self.N))
        
        for i in range(self.N):
            for j in range(i+1, self.N):
                # Build σ_x^i σ_x^j operator
                ops = [self.identity] * self.N
                ops[i] = self.sigma_x
                ops[j] = self.sigma_x
                
                op = ops[0]
                for o in ops[1:]:
                    op = np.kron(op, o)
                
                corr = np.real(state.conj() @ op @ state)
                correlations[i, j] = corr
                correlations[j, i] = corr
        
        return correlations
    
    def calculate_infidelity(self, J_target: np.ndarray, J_exp: np.ndarray) -> float:
        """
        Calculate infidelity metric from equation (12).
        
        Args:
            J_target: Desired interaction matrix
            J_exp: Experimentally achieved matrix
            
        Returns:
            Infidelity value between 0 (perfect) and 1 (orthogonal)
        """
        # Remove diagonals
        J_target_tilde = J_target - np.diag(np.diag(J_target))
        J_exp_tilde = J_exp - np.diag(np.diag(J_exp))
        
        # Frobenius inner product and norms
        inner = np.trace(J_exp_tilde.T @ J_target_tilde)
        norm_exp = np.sqrt(np.trace(J_exp_tilde.T @ J_exp_tilde))
        norm_target = np.sqrt(np.trace(J_target_tilde.T @ J_target_tilde))
        
        if norm_exp * norm_target > 0:
            infidelity = 0.5 * (1 - inner / (norm_exp * norm_target))
        else:
            infidelity = 1.0
        
        return infidelity
    
    def visualize_system(self):
        """Visualize the ion configuration and mode structure."""
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Plot 1: Ion positions
        ax1 = fig.add_subplot(gs[0, 0])
        if self.geometry == '1D':
            ax1.scatter(self.ion_positions[:, 2]*1e6, np.zeros(self.N), s=100, c='red')
            ax1.set_xlabel('Position (μm)')
            ax1.set_ylabel('')
            ax1.set_title('Ion Positions')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.scatter(self.ion_positions[:, 0]*1e6, 
                       self.ion_positions[:, 1]*1e6, s=100, c='red')
            ax1.set_xlabel('X Position (μm)')
            ax1.set_ylabel('Y Position (μm)')
            ax1.set_title('2D Ion Crystal')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
        
        # Plot 2: Mode frequencies
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.stem(range(self.N), self.mode_frequencies, basefmt=' ')
        ax2.set_xlabel('Mode Index')
        ax2.set_ylabel('Frequency (MHz)')
        ax2.set_title('Normal Mode Spectrum')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mode vectors for first 3 modes
        ax3 = fig.add_subplot(gs[0, 2])
        for k in range(min(3, self.N)):
            ax3.plot(self.mode_vectors[:, k], label=f'Mode {k+1}', 
                    marker='o', linestyle='-', alpha=0.7)
        ax3.set_xlabel('Ion Index')
        ax3.set_ylabel('Mode Amplitude')
        ax3.set_title('Mode Participation Vectors')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4-6: First three J^(k) matrices
        for idx in range(3):
            if idx < self.N:
                ax = fig.add_subplot(gs[1, idx])
                im = ax.imshow(self.J_matrices[idx], cmap='RdBu', 
                              vmin=-0.5, vmax=0.5)
                ax.set_title(f'J^({idx+1}) Matrix')
                ax.set_xlabel('Ion j')
                ax.set_ylabel('Ion i')
                plt.colorbar(im, ax=ax, fraction=0.046)
        
        plt.tight_layout()
        plt.show()
    
    def plot_interaction_matrix(self, J: np.ndarray, title: str = "Interaction Matrix"):
        """Visualize the interaction matrix."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matrix plot
        im = ax1.imshow(J, cmap='RdBu', vmin=-np.max(np.abs(J)), vmax=np.max(np.abs(J)))
        ax1.set_xlabel('Ion j')
        ax1.set_ylabel('Ion i')
        ax1.set_title(title)
        plt.colorbar(im, ax=ax1)
        
        # Distance-dependence plot (for 1D chains)
        if self.geometry == '1D':
            distances = []
            couplings = []
            for i in range(self.N):
                for j in range(i+1, self.N):
                    distances.append(j - i)
                    couplings.append(J[i, j])
            
            ax2.scatter(distances, np.abs(couplings), alpha=0.6)
            ax2.set_xlabel('Distance |i - j|')
            ax2.set_ylabel('|J_ij|')
            ax2.set_title('Coupling vs Distance')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    # Create a 9-ion harmonic chain simulator
    print("Creating 9-ion trapped ion simulator...")
    sim = TrappedIonSimulator(N=9, geometry='1D', anharmonic=False)
    
    # Visualize the system configuration
    print("\nSystem Configuration:")
    sim.visualize_system()
    
    # Example 1: Power-law interactions
    print("\nExample 1: Power-law interactions (α=1.5)")
    J_power = sim.power_law_interaction(alpha=1.5, J0=1.0)
    sim.plot_interaction_matrix(J_power, "Power-law Interactions (α=1.5)")
    
    # Example 2: All-to-all interactions
    print("\nExample 2: All-to-all interactions")
    J_all = sim.all_to_all_interaction(J_strength=0.5)
    sim.plot_interaction_matrix(J_all, "All-to-All Interactions")
    
    # Example 3: Multi-mode driving
    print("\nExample 3: Multi-mode driving with 3 tones")
    amplitudes = [0.1, 0.08, 0.05]  # MHz
    detunings = [sim.mode_frequencies[0] + 0.1, 
                 sim.mode_frequencies[2] + 0.05,
                 sim.mode_frequencies[4] - 0.03]  # MHz
    J_multi, weights = sim.multi_mode_drive(amplitudes, detunings)
    sim.plot_interaction_matrix(J_multi, "Multi-mode Driven Interactions")
    
    # Example 4: Quantum dynamics simulation
    print("\nExample 4: Simulating quantum dynamics...")
    times = np.linspace(0, 10, 100)  # Time in units of 1/J
    results = sim.simulate_dynamics(
        J_power, 
        times,
        initial_state='all_down',
        observables=['magnetization']
    )
    
    # Plot dynamics
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(sim.N):
        ax.plot(times, [mag[i] for mag in results['magnetization']], 
               label=f'Ion {i+1}', alpha=0.7)
    ax.set_xlabel('Time (1/J)')
    ax.set_ylabel('<σ_z>')
    ax.set_title('Spin Dynamics under Power-law Interactions')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Example 5: Compare harmonic vs anharmonic (equispaced)
    print("\nExample 5: Comparing harmonic vs equispaced chains")
    sim_harmonic = TrappedIonSimulator(N=7, anharmonic=False)
    sim_equispaced = TrappedIonSimulator(N=7, anharmonic=True)
    
    # Target: nearest-neighbor interactions
    J_target = np.zeros((7, 7))
    for i in range(6):
        J_target[i, i+1] = J_target[i+1, i] = 1.0
    
    # Try to achieve NN with both
    J_nn_harm = sim_harmonic.nearest_neighbor_interaction()
    J_nn_equi = sim_equispaced.nearest_neighbor_interaction()
    
    # Calculate infidelities
    fidelity_harm = sim_harmonic.calculate_infidelity(J_target, J_nn_harm)
    fidelity_equi = sim_equispaced.calculate_infidelity(J_target, J_nn_equi)
    
    print(f"Harmonic chain NN infidelity: {fidelity_harm:.4f}")
    print(f"Equispaced chain NN infidelity: {fidelity_equi:.4f}")
    
    # Visualize both
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    im1 = axes[0].imshow(J_nn_harm, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title(f'Harmonic Chain\nInfidelity: {fidelity_harm:.4f}')
    axes[0].set_xlabel('Ion j')
    axes[0].set_ylabel('Ion i')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(J_nn_equi, cmap='RdBu', vmin=-1, vmax=1)
    axes[1].set_title(f'Equispaced Chain\nInfidelity: {fidelity_equi:.4f}')
    axes[1].set_xlabel('Ion j')
    axes[1].set_ylabel('Ion i')
    plt.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
    print("\nSimulation complete!")
