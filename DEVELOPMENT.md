# Development Guide

This file provides comprehensive development guidelines, coding standards, and technical documentation for contributors.

## Project Overview

This is a Python library for synthesizing quantum gates on trapped-ion analog hardware using the Richerme protocol. It implements quantum gate synthesis techniques for trapped-ion analog quantum computers, specifically focusing on efficient construction of multi-qubit Pauli string unitaries using global Mølmer-Sørensen (MS) operations combined with single-qubit rotations.

## Development Environment

**Recommended Environment**: qiskit-fresh conda environment

```bash
# Create and activate the recommended environment
conda create -n qiskit-fresh python=3.9 -y
conda activate qiskit-fresh

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Common Commands

### Testing
```bash
# Run the comprehensive demonstration
python demo_zxx.py

# Run full test suite with verbose output
python -m pytest test_richerme_ion_analog.py -v

# Run tests quietly (minimal output)
pytest -q

# Run specific test class
python -m pytest test_richerme_ion_analog.py::TestGateSynthesis -v

# Run single test
python -m pytest test_richerme_ion_analog.py::TestGateSynthesis::test_Z1X2X3_synthesis_accuracy -v
```

### Running Simulations
```bash
# Basic demonstration
python demo_zxx.py

# Full trapped-ion simulator
python rich_sim.py

# H2 molecule simulation
python rich_sim_h2.py

# H2O molecule simulation (if available)
python rich_sim_h2o.py
```

## Code Architecture

### Core Module: richerme_ion_analog.py

The library implements a **UMQ–Rz–UMQ construction pattern** for synthesizing arbitrary n-body Pauli string operations:

```
U(P₁⊗P₂⊗...⊗Pₙ, t) = R_post · UMQ(-π/2) · Rz(±2t) · UMQ(+π/2) · R_pre
```

**Key Components:**

1. **Linear Algebra Utilities** (lines 6-27)
   - `_kronN()`: Efficient Kronecker product for multiple matrices
   - `_pauli()`: Single-qubit Pauli operators embedded in n-qubit space
   - `_sum_pauli()`: Sum of Pauli operators across all qubits
   - `_expm()`: Matrix exponentiation via eigendecomposition (numerically stable)

2. **Single-Qubit Rotations** (lines 29-36)
   - `Rx()`, `Ry()`, `Rz()`: Rotation gates around X, Y, Z axes
   - Applied to individual qubits in n-qubit systems

3. **Global Entangling Operation** (lines 39-47)
   - `UMQ(n, chi)`: Universal Multi-Qubit operation implementing exp(-i(χ/4)(∑Xᵢ)²)
   - Global MS-like gate that entangles all qubits simultaneously
   - Hardware-native for trapped-ion systems

4. **Gate Synthesis Engine** (lines 59-114)
   - `_build_core_string()`: Core synthesis routine implementing the UMQ-Rz-UMQ pattern
     - `internal_choice='X'`: Builds XXX... operations
     - `internal_choice='Y'`: Builds YXX... operations
   - `n_body_string()`: Main public interface for arbitrary Pauli strings
     - **Current limitation**: Only supports patterns like [P, X, X, ...] where first qubit can be X/Y/Z and all others must be X
     - Uses basis rotation techniques to convert Y/Z operations
   - `Z1X2X3()`: Convenience function for 3-qubit Z⊗X⊗X operation

5. **Hardware Coupling Calculation** (lines 117-142)
   - `Jij_from_multimode()`: Calculates realistic Ising couplings from multi-tone driving parameters
   - Implements: J_ij = ∑ₖ [(∑ₘ Ωₘ²R/(μₘ² - ωₖ²)) · Bᵢₖ · Bⱼₖ]
   - Used for hardware-realistic simulations

6. **Validation Utilities** (lines 145-151)
   - `target_pauli_string_unitary()`: Generates ideal target unitaries for comparison
   - `unitary_distance()`: Phase-invariant spectral distance metric

### Trapped-Ion Simulators

**rich_sim.py** - Full-featured trapped-ion simulator (684 lines):
- `TrappedIonSimulator` class implements complete physics model
- Key capabilities:
  - Calculates equilibrium ion positions (harmonic and anharmonic traps)
  - Computes normal modes and mode frequencies
  - Generates interaction matrices from mode participation
  - Simulates quantum dynamics under Ising evolution
  - Supports 1D and 2D ion crystals
- Interaction types: power-law, nearest-neighbor, all-to-all
- Includes visualization methods for ion configurations and mode structures

**rich_sim_h2.py** - Hydrogen molecule simulation (203 lines):
- `H2TimeEvolutionSimulator` class for molecular quantum chemistry
- Implements H₂ Hamiltonian as Pauli string decomposition
- Two evolution methods:
  - Adiabatic evolution: Gradually changes bond length
  - Imaginary time evolution: Finds ground state via cooling
- Used for demonstrating quantum chemistry applications

### Testing Philosophy

The test suite (`test_richerme_ion_analog.py`, 406 lines) achieves **63 comprehensive tests** organized by component:
- Tests validate numerical accuracy to machine precision (~1e-15)
- Structured by functionality: constants, linear algebra, rotations, UMQ, targets, synthesis, coupling, utilities
- Uses parametrized tests for parameter sweeps
- Two precision thresholds:
  - `NUMERICAL_PRECISION = 1e-14`: For linear algebra operations
  - `GATE_SYNTHESIS_PRECISION = 1e-12`: For full gate synthesis

## Important Constraints

### Current Limitations of n_body_string()

The `n_body_string()` function expects **all qubits except the first to have X Pauli operators**:
- Valid: `['Z', 'X', 'X']`, `['Y', 'X', 'X']`, `['X', 'X', 'X']`
- Invalid: `['Z', 'Y', 'Z']`, `['Z', 'Z', 'X']`

This is not a fundamental limitation of the physics but a current implementation constraint. To synthesize arbitrary Pauli strings, additional basis rotations would need to be implemented.

### Numerical Considerations

- **Memory scaling**: O(2ⁿ) for n-qubit systems (exponential quantum state space)
- **Computational complexity**: O(8ⁿ) dominated by matrix exponentiation
- **Precision**: Uses `np.linalg.eigh()` for stable eigenvalue decomposition
- **Systems tested**: Up to 4 qubits thoroughly tested; larger systems possible but slower

## Key Scientific Concepts

1. **UMQ Construction**: The library synthesizes arbitrary multi-qubit Pauli operations using only:
   - Global entangling gates (UMQ) that act on all qubits simultaneously
   - Local single-qubit rotations
   - This matches trapped-ion hardware capabilities

2. **Hardware-Native Operations**:
   - Trapped ions naturally implement global XX-type interactions
   - Single-qubit rotations are fast and accurate
   - The synthesis decomposes arbitrary operations into these native gates

3. **Multi-Mode Coupling**:
   - Trapped ions have multiple vibrational modes
   - Multi-tone driving can selectively couple to different modes
   - Different mode weights produce different interaction patterns (power-law, nearest-neighbor, all-to-all)

## File Organization

- `richerme_ion_analog.py` (152 lines): Core synthesis library - use this for gate synthesis
- `demo_zxx.py` (235 lines): Comprehensive demonstrations and examples
- `test_richerme_ion_analog.py` (406 lines): Full test suite with 63 tests
- `rich_sim.py` (684 lines): Advanced trapped-ion physics simulator
- `rich_sim_h2.py` (203 lines): Hydrogen molecule quantum chemistry
- `rich_sim_h2o.py`: Water molecule simulation (not yet examined)
- `setup.py`: Package configuration with entry points
- `requirements.txt`: Dependencies (numpy, scipy, pytest, matplotlib, jupyter, qiskit)

## Expected Test Output

Running `python -m pytest test_richerme_ion_analog.py -v` should show:
```
================================== 63 passed in X.XXs ==================================
```

Running `python demo_zxx.py` should show synthesis fidelity errors around `~1e-15`, demonstrating machine precision accuracy.

## Richerme Hardware Specifications (171Yb+)

### Physical System
This project simulates **trapped 171Yb+ ions** in linear Paul traps, based on experimental systems described in the hardware documentation.

### Hardware Parameters

#### Ion Species: 171Yb+ (Ytterbium-171)
- **Qubit encoding**: Hyperfine ground states in 2S₁/₂ manifold
  - |0⟩ = |F=0, mF=0⟩
  - |1⟩ = |F=1, mF=0⟩
- **Hyperfine splitting**: ω₀/(2π) = 12.6 GHz
- **Transition wavelength**: 369.5 nm (for cooling and manipulation)
- **Ion mass**: 171 amu = 2.838×10⁻²⁵ kg

#### Coherence Properties
- **T₁ (energy relaxation)**: Effectively infinite (>> 1000 s)
- **T₂ (dephasing time)**: > 1 second
- **Single-qubit gate fidelity**: 99.8%
- **Two-qubit gate fidelity**: 97.0%

#### Trap Parameters (Typical Values)
- **Axial frequency**: ωz/(2π) ≈ 0.1 MHz (100 kHz)
- **Radial frequencies**: ωx,y/(2π) ≈ 5 MHz
- **Ion separation**: ~5 μm (equilibrium spacing)
- **Trap depth**: ~100 K equivalent

#### Laser Parameters
- **Cooling wavelength**: 369.5 nm (2S₁/₂ → 2P₁/₂)
- **Rabi frequency**: Ω/(2π) ~ 10-100 kHz (typical)
- **Recoil frequency**: R/(2π) = ℏk²/(2m·2π) ≈ 8.55 kHz
- **Detuning range**: μ/(2π) ~ 0.1-10 MHz from motional modes

#### System Scale
- **Current systems**: Up to 30 ions in 1D chains
- **Planned systems**: 50-100 ions
- **2D crystals**: Up to ~50-100 ions in planar arrangements

### Gate Implementation

#### Native Gate Set
1. **Single-qubit rotations**: Rₓ(θ), Ry(θ), Rz(θ)
   - Implementation: Resonant laser pulses on hyperfine transition
   - Speed: ~1-10 μs per gate
   - Fidelity: 99.8%

2. **Global Mølmer-Sørensen (MS) gate**: UMQ(χ)
   - Mathematical form: exp(-i(χ/4)(∑ᵢ Xᵢ)²)
   - Implementation: Bichromatic laser beams detuned from vibrational modes
   - Acts on all ions simultaneously
   - Speed: ~100 μs - 1 ms
   - Fidelity: 97%

#### Multi-Mode Driving
The key to flexible interactions is **multi-tone global driving**:

```
Hamiltonian: H = ∑ₘ [Ωₘ cos(μₘt) · ∑ᵢ σᵢˣ · (aₖ + aₖ†)]
```

Where:
- `Ωₘ`: Rabi frequency of tone m
- `μₘ`: Detuning of tone m from qubit transition
- `aₖ, aₖ†`: Creation/annihilation operators for mode k

This produces effective Ising couplings:
```
J_ij = ∑ₖ [∑ₘ (Ωₘ²R)/(μₘ² - ωₖ²)] · Bᵢₖ · Bⱼₖ
```

**Key insight**: By choosing different sets of {Ωₘ, μₘ}, different interaction graphs can be realized:
- **All-to-all**: Drive only COM mode (k=0)
- **Nearest-neighbor**: Drive high-k modes (with equispaced ions)
- **Power-law**: Drive intermediate modes

### Normal Modes

#### Harmonic Chains
For ions in harmonic confinement, normal modes must be computed numerically by solving the eigenvalue problem for the dynamical matrix.

**Mode properties**:
- N ions → N transverse modes
- COM mode (k=0): All ions oscillate in phase
- Zig-zag mode (k=N-1): Alternating phase oscillation
- Frequency range: ωz < ωk < ωx for all transverse modes

#### Equispaced Chains (Anharmonic Potential)
For specially engineered anharmonic potentials, ions can be equally spaced. In this case, modes become **sinusoidal** (Equation 18, Kyprianidis 2024):

```python
Bᵢₖ ≈ √((2 - δₖ,₁)/N) · cos((2i-1)(k-1)π / (2N))
```

**Advantages**:
- Analytical mode structure
- Better nearest-neighbor interactions
- Infidelity → 0 as N → ∞ for nearest-neighbor graphs

### Accessibility Criterion

Not all interaction graphs can be realized with global beams. The **accessibility criterion** (Equation 14, Kyprianidis 2024) states:

```
J_desired is accessible ⟺ B^T · J_desired · B is diagonal
```

Where:
- `B`: N×N matrix of normal mode eigenvectors (columns are modes)
- `J_desired`: Target interaction matrix

**Physical interpretation**: A graph is accessible if and only if it can be decomposed as:
```
J = ∑ₖ cₖ J⁽ᵏ⁾  where  J⁽ᵏ⁾ = b⃗ₖ ⊗ b⃗ₖ
```

For inaccessible graphs, find best approximation by optimizing {cₖ} to minimize infidelity.

### Extended Library Features

The `richerme_ion_analog_extended.py` module provides:

1. **`IonTrapHardware` dataclass**: All 171Yb+ specifications
   ```python
   hw = IonTrapHardware()
   hw.hyperfine_splitting  # 12.6e9 Hz
   hw.T2_coherence        # 1.0 s
   hw.two_qubit_fidelity  # 0.97
   hw.recoil_frequency()  # 8.55e3 Hz
   ```

2. **`n_body_string_arbitrary()`**: Arbitrary Pauli string synthesis
   - Removes original library's limitation
   - Can synthesize ZYZ, XYX, etc. (any pattern)
   - Uses UMQ-Rz-UMQ with basis rotations

3. **`is_accessible()`**: Check if interaction graph is realizable
   - Implements Equation 14 criterion
   - Returns True if B^T·J·B is diagonal

4. **`optimize_mode_weights()`**: Find best approximation
   - For inaccessible graphs
   - Minimizes infidelity via least-squares
   - Returns {cₖ} weights and achieved J matrix

5. **`compute_sinusoidal_modes()`**: Analytical modes for equispaced ions
   - Implements Equation 18 (Kyprianidis 2024)
   - Enables better nearest-neighbor interactions

### Simulation Guidelines

#### When to Use Which Simulator

**`richerme_ion_analog.py`** (Original library):
- Pure gate synthesis (no physics simulation)
- Small systems (≤4 qubits)
- When you need exact UMQ-Rz-UMQ decomposition
- Pedagogical purposes

**`richerme_ion_analog_extended.py`** (Extended library):
- Arbitrary Pauli strings (not just first-qubit-variable)
- Accessibility checking
- Mode weight optimization
- Hardware parameter integration

**`rich_sim.py`** (Full physics simulator):
- Complete ion trap physics
- Any number of ions (tested up to ~30)
- Calculate equilibrium positions and normal modes
- Study interaction graphs
- Visualize configurations

**`rich_sim_h2.py`** (Hydrogen molecule):
- Quantum chemistry with 4 qubits
- VQE implementation
- Hardware-realistic gates (updated)
- Adiabatic and imaginary-time evolution

**`rich_sim_h2o.py`** (Water molecule):
- Quantum chemistry with 10 qubits
- QEE compression (14→10 qubits)
- Hardware-realistic gates (updated)
- Smart term grouping (73→3 operations)

#### Performance Scaling
- **Gate synthesis**: O(8ⁿ) - dominated by 2ⁿ×2ⁿ matrix operations
- **Physics simulation**: O(N²) for ion positions, O(N³) for modes
- **Quantum dynamics**: O(8ⁿ) for n-qubit Hamiltonian evolution

**Practical limits**:
- Gate synthesis: ~10 qubits (1024×1024 matrices)
- Ion physics: ~50-100 ions
- Quantum simulation: ~12-14 qubits on standard hardware

### Example: Complete Workflow

```python
from richerme_ion_analog_extended import (
    IonTrapHardware,
    compute_sinusoidal_modes,
    is_accessible,
    optimize_mode_weights,
    n_body_string_arbitrary,
)

# 1. Set up hardware parameters
hw = IonTrapHardware()
print(f"Two-qubit fidelity: {hw.two_qubit_fidelity}")

# 2. Define system
N = 7  # Number of ions

# 3. Get normal modes (equispaced ions)
B = compute_sinusoidal_modes(N)

# 4. Define desired interaction (nearest-neighbor)
J_desired = np.zeros((N, N))
for i in range(N-1):
    J_desired[i, i+1] = J_desired[i+1, i] = 1.0

# 5. Check accessibility
if is_accessible(J_desired, B):
    print("Exactly realizable!")
    weights = get_mode_weights_if_accessible(J_desired, B)
else:
    print("Not accessible, finding best approximation...")
    result = optimize_mode_weights(J_desired, B)
    weights = result['weights']
    print(f"Infidelity: {result['infidelity']:.6f}")

# 6. Synthesize arbitrary Pauli string gate
U = n_body_string_arbitrary(['Z', 'Y', 'X'], t=0.5)
print(f"Gate shape: {U.shape}")
```

## Coding Principles and Standards

This project follows **SOLID principles** and **Clean Code** practices as outlined in Robert C. Martin's guidelines.

### SOLID Principles

#### 1. Single Responsibility Principle (SRP)
**"A class should have only one reason to change."**

**Examples in this codebase**:
- ✅ `IonTrapHardware`: Only stores hardware parameters
- ✅ `_expm()`: Only does matrix exponentiation
- ✅ `unitary_distance()`: Only computes distance metric
- ❌ **Avoid**: A function that both computes AND visualizes results

**Application**:
```python
# GOOD: Separate responsibilities
def calculate_coupling_matrix(B, omega, mus, Omegas, R):
    """Calculate coupling matrix."""
    return J

def plot_coupling_matrix(J, title):
    """Visualize coupling matrix."""
    plt.imshow(J)
    plt.title(title)

# BAD: Mixed responsibilities
def calculate_and_plot_coupling(B, omega, mus, Omegas, R):
    J = calculate_matrix(...)
    plt.imshow(J)  # Mixing calculation and visualization
    return J
```

#### 2. Open/Closed Principle (OCP)
**"Open for extension, closed for modification."**

**Examples**:
- ✅ `TrappedIonSimulator` can be extended with new interaction types without modifying core
- ✅ Gate synthesis methods use strategy pattern (XX, YY, ZZ gates)

**Application**:
```python
# GOOD: Extensible design
class TrappedIonSimulator:
    def generate_interaction_matrix(self, mode_weights):
        """Generic method - works for any weights."""
        return J

    def power_law_interaction(self, alpha):
        """Specific interaction type - extends base functionality."""
        weights = self._calculate_power_law_weights(alpha)
        return self.generate_interaction_matrix(weights)

# Easy to add new interaction types without modifying base class
```

#### 3. Liskov Substitution Principle (LSP)
**"Derived classes must be substitutable for their base classes."**

**Application**:
- Ensure subclasses maintain the contract of parent classes
- Don't strengthen preconditions or weaken postconditions

#### 4. Interface Segregation Principle (ISP)
**"Clients shouldn't depend on interfaces they don't use."**

**Application**:
```python
# GOOD: Focused interfaces
def optimize_mode_weights(J_desired, B, method='least_squares'):
    """Only requires what it needs."""
    pass

# BAD: Requiring entire simulator when only need B matrix
def optimize_mode_weights(simulator):
    B = simulator.mode_vectors  # Using only small part of large object
    pass
```

#### 5. Dependency Inversion Principle (DIP)
**"Depend on abstractions, not concretions."**

**Examples**:
- ✅ Gate synthesis methods accept any matrix, not specific hardware objects
- ✅ Functions work with numpy arrays (abstraction) not specific data structures

### Clean Code Principles

#### Function Design

**1. Small Functions** (< 20 lines preferred)
```python
# GOOD: Small, focused function
def _expm(H: np.ndarray, t: float) -> np.ndarray:
    """Matrix exponentiation via eigendecomposition."""
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * t * w)) @ v.conj().T

# Each function does ONE thing well
```

**2. Descriptive Names**
```python
# GOOD: Clear, descriptive names
def calculate_equilibrium_positions(self):
    pass

def unitary_distance(U, V):
    pass

# BAD: Unclear abbreviations
def calc_eq_pos(self):
    pass

def ud(U, V):
    pass
```

**3. Function Arguments** (≤ 3 preferred)
```python
# GOOD: Few, clear arguments
def Rz(n: int, i: int, theta: float) -> np.ndarray:
    pass

# If many related parameters, use dataclass
@dataclass
class TrapParams:
    omega_x: float
    omega_y: float
    omega_z: float

def simulate(params: TrapParams):
    pass
```

**4. No Side Effects**
```python
# GOOD: Pure function
def compute_coupling_matrix(B, omega):
    """Returns new matrix, doesn't modify inputs."""
    J = np.zeros((N, N))
    # ... compute ...
    return J

# BAD: Modifies global state
global_J = None
def compute_coupling_matrix(B, omega):
    global global_J
    global_J = np.zeros((N, N))  # Side effect!
```

#### Code Organization

**1. Vertical Formatting**
- Related functions close together
- Called functions below calling functions
- Declare variables close to usage

**2. Horizontal Formatting**
- Lines ≤ 100 characters
- Proper indentation (4 spaces)
- Whitespace around operators

**3. Comments**
```python
# GOOD: Explain WHY, not WHAT
def _expm(H, t):
    """Use eigendecomposition for numerical stability."""  # Why this method
    w, v = np.linalg.eigh(H)  # Code is self-explanatory
    return (v * np.exp(-1j * t * w)) @ v.conj().T

# BAD: Redundant comments
def _expm(H, t):
    # Calculate eigenvalues and eigenvectors
    w, v = np.linalg.eigh(H)  # Obvious from code
    return (v * np.exp(-1j * t * w)) @ v.conj().T
```

#### Error Handling

**1. Use Exceptions, Not Error Codes**
```python
# GOOD: Descriptive exceptions
if self.N < required_qubits:
    raise ValueError(f"Need at least {required_qubits} ions, got {self.N}")

# BAD: Return codes
if self.N < required_qubits:
    return -1  # What does -1 mean?
```

**2. Provide Context**
```python
# GOOD: Clear error messages
raise ValueError(f"Mode matrix B must be {N}×{N}, got {B.shape}")

# BAD: Vague messages
raise ValueError("Wrong shape")
```

#### Testing

**1. Test at Appropriate Levels**
```python
# Unit test: Test individual functions
def test_rz_gate():
    U = Rz(n=2, i=0, theta=np.pi/2)
    assert U.shape == (4, 4)
    assert np.allclose(U @ U.conj().T, np.eye(4))

# Integration test: Test complete workflow
def test_full_gate_synthesis():
    U = n_body_string_arbitrary(['Z', 'Y', 'X'], t=0.5)
    target = target_pauli_string_unitary('ZYX', 0.5)
    assert unitary_distance(U, target) < 1e-12
```

**2. Test One Concept Per Test**
```python
# GOOD: Focused tests
def test_gate_is_unitary():
    U = Rz(2, 0, np.pi/2)
    assert np.allclose(U @ U.conj().T, np.eye(4))

def test_gate_has_correct_shape():
    U = Rz(2, 0, np.pi/2)
    assert U.shape == (4, 4)

# BAD: Multiple concepts
def test_gate_properties():
    U = Rz(2, 0, np.pi/2)
    assert U.shape == (4, 4)  # Testing shape
    assert np.allclose(U @ U.conj().T, np.eye(4))  # Testing unitarity
    assert isinstance(U, np.ndarray)  # Testing type
```

### Code Review Checklist

When writing or reviewing code in this project:

- [ ] **Functions < 20 lines** and do ONE thing
- [ ] **Descriptive names** (no abbreviations unless standard)
- [ ] **No side effects** (pure functions preferred)
- [ ] **≤ 3 function arguments** (use dataclasses for more)
- [ ] **Type hints** on all public functions
- [ ] **Docstrings** explain purpose, args, returns, examples
- [ ] **No magic numbers** (use named constants)
- [ ] **Tests** for new functionality
- [ ] **Numerical precision** validated (≤ 1e-12 for gates)
- [ ] **Comments** explain WHY, not WHAT
- [ ] **Error messages** are descriptive
- [ ] **Code formatted** with consistent style

### Naming Conventions

**Variables**:
- `snake_case` for variables and functions
- `UPPER_CASE` for constants
- `n`, `N` for number of qubits/ions
- `i`, `j`, `k` for indices (ions, modes)
- `m` for multi-tone index
- `t` for time
- `theta`, `phi` for angles
- `U`, `V` for unitary matrices
- `H` for Hamiltonians
- `J` for coupling matrices
- `B` for mode matrices
- `omega` for frequencies
- `mu` for detunings
- `Omega` for Rabi frequencies

**Functions**:
- Verbs for actions: `calculate_`, `compute_`, `build_`, `generate_`
- Prefix `_` for private/internal functions
- No `get_` prefix for simple attribute access (use properties)

### Example: Well-Structured Function

```python
def optimize_mode_weights(
    J_desired: np.ndarray,
    B: np.ndarray,
    method: str = 'least_squares'
) -> Dict[str, Any]:
    """
    Find optimal mode weights {c_k} to best approximate desired interaction.

    Implements optimization framework from Richerme 2025, Section 3.2.

    Args:
        J_desired: Target N×N coupling matrix (Laplacian form)
        B: N×N normal mode matrix (columns are mode vectors)
        method: Optimization method ('least_squares' or 'linear_program')

    Returns:
        Dictionary containing:
            - 'weights': Array of mode weights c_k
            - 'J_achieved': Achieved coupling matrix
            - 'infidelity': Fidelity error (0 if exactly accessible)
            - 'accessible': Boolean indicating exact realizability

    Example:
        >>> B = compute_sinusoidal_modes(N=5)
        >>> J = nearest_neighbor_matrix(N=5)
        >>> result = optimize_mode_weights(J, B)
        >>> print(f"Infidelity: {result['infidelity']:.6f}")

    References:
        Richerme 2025, Section 3.2
        Kyprianidis 2024, Equation 14
    """
    # Validate inputs
    N = B.shape[0]
    if J_desired.shape != (N, N):
        raise ValueError(f"J_desired must be {N}×{N}, got {J_desired.shape}")

    # Build mode matrices
    J_modes = [np.outer(B[:, k], B[:, k]) for k in range(N)]

    # Optimize weights
    weights = _solve_least_squares(J_desired, J_modes)

    # Compute achieved matrix and infidelity
    J_achieved = sum(weights[k] * J_modes[k] for k in range(N))
    infidelity = _calculate_infidelity(J_desired, J_achieved)
    accessible = infidelity < 1e-10

    return {
        'weights': weights,
        'J_achieved': J_achieved,
        'infidelity': infidelity,
        'accessible': accessible,
    }
```

This example demonstrates:
- ✅ Clear, descriptive name
- ✅ Type hints for all parameters
- ✅ Comprehensive docstring with example
- ✅ Input validation
- ✅ Small helper functions for complex operations
- ✅ Clear return type (dictionary with named keys)
- ✅ References to source papers
