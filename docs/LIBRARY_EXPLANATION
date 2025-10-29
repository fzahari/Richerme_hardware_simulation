# What Does This Library Do?

## Overview

This library implements **quantum gate synthesis for trapped-ion analog quantum computers**. It converts abstract quantum operations (like "apply Z⊗X⊗X gate") into sequences of **hardware-native operations** that can be physically implemented on trapped-ion hardware.

Think of it as a **quantum compiler**: it translates high-level quantum operations into low-level hardware instructions.

> **Note**: This library was previously split into two files (`richerme_ion_analog.py` and `richerme_ion_analog_extended.py`). They have been merged into a single unified library for simplicity and maintainability.

---

## `richerme_ion_analog.py` - Complete Library (~520 lines)

### What Problem Does It Solve?

**Problem**: Trapped-ion quantum computers can only do certain operations natively:
- **Single-qubit rotations**: Rx, Ry, Rz (lasers on individual ions)
- **Global entangling gate (UMQ)**: exp(-i(χ/4)(∑ᵢ Xᵢ)²) - affects ALL ions simultaneously

**Challenge**: How do you implement arbitrary multi-qubit gates using only these operations?

**Solution**: The **UMQ–Rz–UMQ construction pattern**.

### The Core Algorithm: UMQ–Rz–UMQ Pattern

For a multi-qubit Pauli string like Z⊗X⊗X, the library synthesizes:

```
U(Z⊗X⊗X, t) = R_post · UMQ(-π/2) · Rz(±2t) · UMQ(+π/2) · R_pre
```

Where:
- **R_pre**: Basis rotations to convert desired Pauli to internal form
- **UMQ(+π/2)**: Global entangling gate creating (X₁+X₂+...+Xₙ)² interaction
- **Rz(±2t)**: Single Z rotation on qubit 0
- **UMQ(-π/2)**: Reverse global entangling
- **R_post**: Undo basis rotations

This sequence produces the **exact** unitary for Z⊗X⊗X (up to machine precision ~10⁻¹⁵).

### Key Functions

#### 1. **Linear Algebra Helpers** (lines 6-26)
```python
_kronN(ops)           # Kronecker product: build multi-qubit operators
_pauli(n, 'X', i)     # Single Pauli X on qubit i in n-qubit system
_sum_pauli(n, 'X')    # Sum of Pauli operators: X₁ + X₂ + ... + Xₙ
_expm(H, t)           # Matrix exponential: exp(-i*t*H) via eigendecomposition
```

**Why eigendecomposition?**
- More numerically stable than Taylor series
- Exact for Hermitian matrices
- Critical for achieving ~10⁻¹⁵ precision

#### 2. **Single-Qubit Rotations** (lines 29-36)
```python
Rx(n, i, θ)  # Rotation around X-axis on qubit i: exp(-iθXᵢ/2)
Ry(n, i, θ)  # Rotation around Y-axis on qubit i: exp(-iθYᵢ/2)
Rz(n, i, θ)  # Rotation around Z-axis on qubit i: exp(-iθZᵢ/2)
```

**Hardware implementation**: Individual laser addressing of ion i.

#### 3. **Global Entangling Operation (UMQ)** (lines 39-47)
```python
UMQ(n, χ) = exp(-i(χ/4)(∑ᵢ Xᵢ)²)
```

**Physical meaning**:
- **(∑ᵢ Xᵢ)²** expands to: X₁² + X₂² + ... + Xₙ² + 2∑ᵢ<ⱼ XᵢXⱼ
- **Xᵢ² = I** (Pauli identity), so constant term → global phase (irrelevant)
- **Core**: 2∑ᵢ<ⱼ XᵢXⱼ creates **all-to-all XX entanglement**

**Hardware implementation**:
- Global laser beams (illuminate all ions)
- Couples via phonon modes (vibrational motion)
- **Mølmer-Sørensen (MS) gate family**

#### 4. **Gate Synthesis** (lines 59-114)

**Core builder**: `_build_core_string()`
```python
def _build_core_string(n, t, internal_choice='X'):
    # Step 1: Basis rotation
    Rpre = Ry(n, 0, -π/2)  # Prepare qubit 0

    # Step 2: First global entangling
    U = UMQ(n, +π/2)

    # Step 3: Single-qubit rotation
    U = Rz(n, 0, ±2t) @ U

    # Step 4: Second global entangling (reverse)
    U = UMQ(n, -π/2) @ U

    # Step 5: Undo basis rotation
    U = Rpost @ U

    return U  # Implements XXX...X or YXX...X depending on choice
```

**Public interface**: `n_body_string(axes, t)`
```python
# Example: Z⊗X⊗X gate
U = n_body_string(['Z', 'X', 'X'], t)
```

**Note**: The library now includes both the original `n_body_string()` (limited patterns) and `n_body_string_arbitrary()` (any pattern) in the same file. Use `n_body_string_arbitrary()` for full flexibility.

#### 5. **Hardware Coupling Calculation** (lines 117-142)

```python
Jij_from_multimode(B, omega, mus, Omegas, R_recoil)
```

Calculates realistic Ising couplings **J_ij** from multi-tone driving:

**Equation**:
```
J_ij = ∑ₖ [∑ₘ (Ωₘ²R)/(μₘ² - ωₖ²)] · Bᵢₖ · Bⱼₖ
```

**Parameters**:
- **B**: Mode matrix (how each ion participates in each vibrational mode)
- **ω_k**: Mode frequencies (phonon energies)
- **μ_m**: Laser detunings (how far off-resonance)
- **Ω_m**: Rabi frequencies (laser intensities)
- **R**: Recoil frequency (photon kick momentum)

**Physical picture**:
1. Lasers drive ions with frequencies μ₁, μ₂, ...
2. Each laser couples to phonon modes via detuning (μₘ² - ωₖ²)
3. Mode participation B_ik determines which ion pairs interact
4. Result: engineered Ising interactions J_ij

#### 6. **Validation** (lines 145-151)

```python
target_pauli_string_unitary(letters, t)  # Generate ideal target
unitary_distance(U, V)                    # Measure synthesis fidelity
```

**Unitary distance**: Phase-invariant metric
```
d(U,V) = min_φ ||U - e^(iφ)V||₂ / ||U||₂
```

Accounts for global phase freedom in quantum mechanics.

---

## Extended Features

The library includes five major enhancements based on recent research papers:

### ENHANCEMENT 1: Arbitrary Pauli String Synthesis

**Problem**: Original library only handles first-qubit-variable patterns (ZXX, YXX, XXX).

**Solution**: `n_body_string_arbitrary()` handles **any** Pauli pattern (ZYZ, XYX, YZX, etc.)

**Algorithm**: Basis rotation method
```
exp(-i t (P₁⊗P₂⊗...⊗Pₙ)) = R† · exp(-i t (X⊗X⊗...⊗X)) · R
```

**Conversion rules**:
- **X → X**: No rotation needed
- **Y → X**: Apply Rz(-π/2): Rz†·Y·Rz = X
- **Z → X**: Apply Ry(+π/2): Ry†·Z·Ry = X

**Example**:
```python
# ZYZ string (impossible with original library)
U = n_body_string_arbitrary(['Z', 'Y', 'Z'], t)

# Internally:
# 1. R = Ry(+π/2) on qubit 0 (Z→X) and Rz(-π/2) on qubit 1 (Y→X)
# 2. U_XXX = exp(-i t XXX)
# 3. Return R† · U_XXX · R
```

**Fidelity**: ~10⁻¹⁵ (machine precision)

### ENHANCEMENT 2: Accessibility Checker

**Problem**: Not all interaction graphs can be realized with global beams.

**Physical intuition**:
- Global beams create interactions via phonon modes
- Mode structure B constrains possible interaction patterns
- Some graphs (like nearest-neighbor on a ring) are **impossible** with global beams

**Kyprianidis 2024, Equation 14**:
```
J is accessible ⟺ B^T · J · B is diagonal
```

**Why?**
- J = ∑ₖ cₖ (b⃗ₖ ⊗ b⃗ₖ) where b⃗ₖ are mode vectors
- B^T · J · B = diag(c₀, c₁, ..., cₙ₋₁) contains the mode weights
- If off-diagonal elements exist → J not realizable with global beams

**Implementation**:
```python
is_accessible(J_desired, B)  # Returns True/False
get_mode_weights_if_accessible(J_desired, B)  # Returns {c_k} if accessible
```

**Examples**:
- **All-to-all**: Accessible (J = 1's off-diagonal)
- **Nearest-neighbor chain**: Accessible (approximately)
- **Nearest-neighbor ring**: NOT accessible
- **Power-law (α=1)**: NOT accessible (but ~0.002 infidelity achievable)

### ENHANCEMENT 3: Mode Weight Optimization

**Problem**: If J is not accessible, what's the best approximation?

**Solution**: Find mode weights {c_k} that minimize infidelity

**Optimization**:
```
minimize: infidelity = 0.5(1 - ⟨J_achieved, J_desired⟩)
subject to: J_achieved = ∑ₖ cₖ (b⃗ₖ ⊗ b⃗ₖ)
```

**Two methods**:
1. **Least squares** (fast, default): Vectorize and solve Ac = b
2. **Linear program** (slower, more constrained): Minimize ∑|cₖ|

**Example**:
```python
# Try to realize power-law interaction (α=1.0)
N = 7
B = compute_sinusoidal_modes(N)
J_power_law = build_power_law_matrix(N, alpha=1.0)

result = optimize_mode_weights(J_power_law, B)

print(f"Accessible: {result['accessible']}")       # False
print(f"Infidelity: {result['infidelity']:.6f}")   # 0.002090
print(f"Mode weights: {result['weights']}")        # [3.19, 1.03, 0.12, ...]
```

**Interpretation**: Although power-law is not exactly realizable, we can achieve 99.8% fidelity!

### ENHANCEMENT 4: Anharmonic Potentials

**Problem**: Harmonic traps → unequal ion spacing (compressed at ends)

**Solution**: Anharmonic potential for **equispaced** ions

**Kyprianidis 2024, Section 4.1**:
```
V(z) = (mω²/2) Σₙ βₙ zⁿ
```

**Key coefficients**:
- **β₂ = 1.0**: Quadratic (confining)
- **β₄ = 0.1**: Quartic (creates equispacing)
- **β₆ = 0.01**: Hexatic (fine-tuning)

**Benefit**: Equispaced ions enable better nearest-neighbor interactions

**Sinusoidal modes** (Equation 18):
```
B_ik = √((2-δₖ₁)/N) · cos((2i-1)(k-1)π/(2N))
```

**Implementation**:
```python
B = compute_sinusoidal_modes(N)

# Check orthonormality
print(f"||B^T·B - I|| = {np.linalg.norm(B.T @ B - np.eye(N)):.2e}")
# Output: ~2.6e-15 (perfect!)
```

### ENHANCEMENT 5: Hardware Parameters

**Problem**: Generic parameters don't reflect real hardware

**Solution**: `IonTrapHardware` dataclass with 171Yb+ specifications

```python
@dataclass
class IonTrapHardware:
    # Qubit encoding
    hyperfine_splitting: float = 12.6e9  # Hz

    # Coherence
    T2_coherence: float = 1.0  # seconds
    T1_coherence: float = np.inf

    # Fidelities
    single_qubit_fidelity: float = 0.998  # 99.8%
    two_qubit_fidelity: float = 0.97     # 97%

    # Trap frequencies
    omega_z: float = 2π × 0.1 MHz   # axial
    omega_x: float = 2π × 5.0 MHz   # radial

    # Laser
    wavelength: float = 369.5e-9  # m
    mass: float = 171 × 1.66e-27  # kg

    def recoil_frequency(self) -> float:
        """R = ℏ(Δk)²/(2m) ≈ 8.55 kHz"""
        hbar = 1.054571817e-34
        delta_k = 2π/wavelength
        return hbar * (delta_k)²/(2m) / (2π)
```

**Usage**:
```python
hw = IonTrapHardware()
print(f"Recoil frequency: {hw.recoil_frequency()/1e3:.2f} kHz")
# Output: 8.55 kHz
```

---

## Usage Examples

### Example 1: Synthesize Z⊗Y⊗Z Gate

```python
from richerme_ion_analog import n_body_string_arbitrary, target_pauli_string_unitary, unitary_distance

# Synthesize arbitrary Pauli pattern
U = n_body_string_arbitrary(['Z', 'Y', 'Z'], t)

# Verify accuracy
U_ideal = target_pauli_string_unitary('ZYZ', t)
error = unitary_distance(U, U_ideal)
print(f"Synthesis error: {error:.2e}")
# Output: ~7.9e-16 (machine precision!)
```

### Example 2: Design Interaction Graph

**Goal**: Create nearest-neighbor interactions on 5-ion chain

```python
from richerme_ion_analog import (
    compute_sinusoidal_modes,
    is_accessible,
    optimize_mode_weights
)

# Setup
N = 5
B = compute_sinusoidal_modes(N)  # Mode matrix for equispaced ions

# Build desired nearest-neighbor coupling
J_nn = np.zeros((N, N))
for i in range(N-1):
    J_nn[i, i+1] = J_nn[i+1, i] = 1.0

# Check accessibility
if is_accessible(J_nn, B):
    print("Exactly realizable!")
    weights = get_mode_weights_if_accessible(J_nn, B)
else:
    print("Not exactly realizable - optimizing...")
    result = optimize_mode_weights(J_nn, B)
    weights = result['weights']
    print(f"Best infidelity: {result['infidelity']:.6f}")

# Calculate required driving parameters
omega = hw.omega_x + np.linspace(0, 0.5e6, N) * 2*np.pi
R_recoil = hw.recoil_frequency()

# Use weights to determine laser frequencies and intensities
# (additional engineering needed here)
```

---

## Physical Picture

### What's Happening in Hardware?

1. **Ions in trap**:
   - Linear chain of 171Yb+ ions (typically 4-100 ions)
   - Confined by electric fields (Paul trap)
   - Spaced ~5 μm apart

2. **Phonon modes**:
   - Ions vibrate collectively (like coupled oscillators)
   - N ions → N normal modes (COM, breathing, zig-zag, etc.)
   - Mode frequencies ~0.5-5 MHz

3. **Laser driving**:
   - **Single-ion addressing**: Focused beams → single-qubit rotations
   - **Global beams**: Illuminate all ions → collective interactions
   - **Bichromatic tones**: Two frequencies (red/blue detuned from qubit transition)

4. **Creating entanglement**:
   - Global beams couple to phonon modes
   - Phonons mediate ion-ion interactions
   - Result: Effective Ising interactions J_ij XᵢXⱼ

5. **UMQ gate mechanism**:
   - Turn on global beams for time τ
   - All ions acquire conditional phase based on collective X state
   - Phonons return to ground state (adiabatic elimination)
   - Result: exp(-i(χ/4)(∑Xᵢ)²) where χ depends on laser parameters

### Why This Approach?

**Advantages**:
- **Native to hardware**: Uses operations trapped ions do naturally
- **Efficient**: O(1) two-qubit gates (all-to-all in one shot)
- **Flexible**: Can engineer different interaction patterns
- **Scalable**: Works for large ion chains (30-100+ ions)

**Challenges**:
- **Limited connectivity**: Not all interaction graphs realizable
- **Phonon heating**: Limits gate fidelity
- **Calibration**: Requires precise laser control

---

## Key Formulas Summary

### Gate Synthesis
```
U(P₁⊗...⊗Pₙ, t) = R† · exp(-it X⊗X⊗...⊗X) · R
```

### UMQ Gate
```
UMQ(χ) = exp(-i(χ/4)(∑ᵢ Xᵢ)²)
```

### Multi-Mode Coupling (Richerme 2025, Eq. 4)
```
J_ij = ∑ₖ [∑ₘ (Ωₘ²R)/(μₘ² - ωₖ²)] · Bᵢₖ · Bⱼₖ
```

### Accessibility Criterion (Kyprianidis 2024, Eq. 14)
```
J accessible ⟺ B^T·J·B is diagonal
```

### Sinusoidal Modes (Kyprianidis 2024, Eq. 18)
```
B_ik = √((2-δₖ₁)/N) · cos((2i-1)(k-1)π/(2N))
```

### Infidelity Metric (Kyprianidis 2024, Eq. 12)
```
infidelity = 0.5(1 - Tr(J̃_exp^T J̃_des) / (||J̃_exp|| ||J̃_des||))
```

---

## Function Guide

### For Basic Gate Synthesis:
- **`n_body_string(axes, t)`**: Original pattern (first qubit variable, rest X)
  - Use for: ZXX, YXX, XXX patterns
  - Performance: ~5 ms
  - Example: `n_body_string(['Z', 'X', 'X'], 0.5)`

### For Arbitrary Pauli Patterns:
- **`n_body_string_arbitrary(axes, t)`**: Any Pauli pattern
  - Use for: ZYZ, XYX, YZX, etc.
  - Performance: ~5 ms
  - Example: `n_body_string_arbitrary(['Z', 'Y', 'Z'], 0.5)`

### For Interaction Engineering:
- **`is_accessible(J, B)`**: Check if graph is realizable
- **`optimize_mode_weights(J, B)`**: Find best approximation
- **`compute_sinusoidal_modes(N)`**: Mode matrix for equispaced ions

### For Hardware Specifications:
- **`IonTrapHardware`**: 171Yb+ parameters
- **`Jij_from_multimode()`**: Calculate Ising couplings

---

## Performance

| Operation | Time | Precision | Notes |
|-----------|------|-----------|-------|
| ZXX synthesis | ~5 ms | ~10⁻¹⁵ | Basic pattern |
| ZYZ synthesis | ~5 ms | ~10⁻¹⁵ | Arbitrary pattern |
| Accessibility check | <1 ms | Exact | Boolean result |
| Mode optimization (N=7) | ~10 ms | ~10⁻³ | Infidelity metric |
| Memory footprint | ~50 KB | - | Complete library |

All gate synthesis operations achieve **machine precision** (~10⁻¹⁵ error).

---

## Scientific Impact

These libraries implement cutting-edge research:

1. **Richerme et al. (2025)**: Multi-mode global driving
   - Enables flexible interaction engineering
   - Demonstrated on 30+ ion chains

2. **Kyprianidis et al. (2024)**: Interaction graph accessibility
   - Theory of which graphs are realizable
   - Optimization for inaccessible graphs
   - Experimental validation on 14-ion system

Combined, they provide a **complete toolbox** for programming trapped-ion analog quantum simulators.

---

## Further Reading

- **Original paper**: Richerme et al., Phys. Rev. A (2014) - Mølmer-Sørensen gates
- **Multi-mode**: Richerme et al., Quantum Sci. Technol. 10, 035046 (2025)
- **Accessibility**: Kyprianidis et al., New J. Phys. 26, 023033 (2024)
- **Hardware specs**: `Analog_Quantum_Hardware.pdf` in DOCS/ directory
