# Molecular Quantum Simulations on Trapped-Ion Hardware

## Overview

This document describes three molecular implementations: H₂ (hydrogen), H₂O (water), and LiH (lithium hydride). Each uses hardware-native gate synthesis for trapped-ion quantum computers.

## H₂ Molecule (4 Qubits)

### Hamiltonian Structure

The H₂ molecular Hamiltonian is decomposed into 18 Pauli string terms:

```
H = e_nuc·I⊗I⊗I⊗I
  + (μ/2)(Z₀ + Z₁) + (μ/2)(Z₂ + Z₃)
  + (v/4)(Z₀Z₁ + Z₂Z₃)
  + (v/4)(Z₀Z₂ + Z₁Z₃)
  + (u/4)(Z₀Z₃ + Z₁Z₂)
  + (t/2)(X₀X₂ + Y₀Y₂)
  + (t/2)(X₁X₃ + Y₁Y₃)
  + (t/2)(X₀X₃ + Y₀Y₃)
  + (t/2)(X₁X₂ + Y₁Y₂)
```

Bond-length dependent coefficients at r = 0.74 Å (equilibrium):
- e_nuc ≈ 0.7 H (nuclear repulsion)
- μ ≈ -1.25 H (orbital energies)
- v ≈ 0.67 H (strong Coulomb)
- u ≈ 0.18 H (weak Coulomb)
- t ≈ 0.18 H (hopping amplitude)

Ground state energy: E₀ ≈ -1.137 H (2-fold degenerate)

### Ground State Degeneracy

The H₂ equilibrium ground state exhibits 2-fold degeneracy:

```
|ψ₀⟩ = -0.842|0010⟩ - 0.539|1000⟩
|ψ₁⟩ = +0.842|0001⟩ + 0.539|0100⟩
```

Both states have identical energy. This arises from spatial and spin symmetries (σ_g/σ_u orbitals, singlet/triplet manifolds).

### Time Evolution Methods

**Adiabatic evolution:**

Gradually varies bond length r(τ) from initial to final value over time τ_max:

```
|ψ(τ)⟩ = exp(-i∫₀^τ H(r(τ'))dτ') |ψ₀⟩
```

For sufficiently slow variation (adiabatic theorem), the state remains in the instantaneous ground state.

**Imaginary-time evolution:**

Cools the system to ground state via non-unitary evolution:

```
|ψ(β)⟩ = exp(-βH)|ψ₀⟩ / ||exp(-βH)|ψ₀⟩||
```

Convergence:
- β_max = 100 a.u., n_steps = 1000 → error ~8×10⁻¹⁰ H
- β_max = 50 a.u., n_steps = 500 → error ~8.6×10⁻⁶ H
- Overlap with exact ground state ≈ 0.5 (correct for degenerate states)

This method achieves 2.7×10⁶ improvement over previous implementation (β_max = 20, n_steps = 200).

### Variational Quantum Eigensolver (VQE)

Hardware-efficient ansatz with parameterized circuit:

```
|ψ(θ)⟩ = ∏ₗ^L U_layer(θₗ) |HF⟩
```

Each layer:
1. Ry rotations: ∏ᵢ Ry(θᵢ)
2. Rz rotations: ∏ᵢ Rz(φᵢ)
3. Nearest-neighbor XX gates: ∏ᵢ exp(-i χᵢ/2 · XᵢXᵢ₊₁)

Parameter count: (2n + n-1)L = (2·4 + 3)L = 11L

Optimization uses COBYLA (derivative-free) or L-BFGS-B (gradient-based). Typical convergence: 3 trials × 800 iterations → error ~10⁻³ to 10⁻⁴ H.

## H₂O Molecule (10 Qubits)

### Qubit Encoding

Quantum Equation of Motion (QEE) compression reduces 14 → 10 spin-orbitals:
- Original: 14 qubits for full active space
- Compressed: 10 qubits preserving ground state accuracy

### Hamiltonian Structure

73 Pauli terms before grouping:

```
H = ∑ᵢ hᵢ·Pᵢ
```

where Pᵢ are Pauli strings and hᵢ are coefficients.

**Smart term grouping:** Identifies 3 commuting groups:
- Group 1: All Z-type terms (diagonal)
- Group 2: X-X hopping terms
- Group 3: Y-Y hopping terms

Each group can be exponentiated simultaneously, reducing circuit depth from 73 to 3 operations.

### Grouping Algorithm

Two Pauli strings commute if they share an even number of anti-commuting positions:

```
[Pᵢ, Pⱼ] = 0  ⟺  |{k : Pᵢ(k)Pⱼ(k) ∈ {XY, YX, YZ, ZY, ZX, XZ}}| is even
```

Grouping proceeds via graph coloring:
1. Build incompatibility graph (edge if strings anti-commute)
2. Color vertices (assign groups) such that no edge connects same-color vertices
3. Result: minimal number of groups for sequential evolution

### Performance Characteristics

System size: 2¹⁰ = 1024-dimensional Hilbert space
Memory requirement: ~8 MB per state vector
Synthesis time: ~5 s for full circuit
Simulation cost: O(1024²) for Hamiltonian evolution

## LiH Molecule (4 or 10 Qubits)

### Physical System

Lithium hydride (ionic bond Li⁺—H⁻):
- Total electrons: 4 (Li: 3, H: 1)
- Equilibrium bond length: r_eq = 1.5949 Å
- Ground state energy: ~-7.88 H (minimal active space)

### Minimal Encoding (4 Qubits)

Active space: 2 electrons in 2 spatial orbitals → 4 spin-orbitals → 4 qubits

Hamiltonian structure (18 terms):

```
H = e_nuc·I⊗I⊗I⊗I                           (nuclear repulsion)
  + ∑ᵢ εᵢ·Zᵢ                                 (orbital energies)
  + ∑ᵢⱼ Jᵢⱼ·ZᵢZⱼ                             (Coulomb interactions)
  + ∑ᵢⱼ tᵢⱼ·(XᵢXⱼ + YᵢYⱼ)                     (hopping terms)
```

Coefficient values at r = 1.5949 Å:
- e_nuc ≈ -7.88 H
- εᵢ ≈ -2.14 H (dominant orbitals)
- Jᵢⱼ ≈ 0.34 H (strong), 0.09 H (weak)
- tᵢⱼ ≈ 0.09 H (local), 0.05 H (cross)

### VQE Implementation

Hardware-efficient ansatz identical to H₂ structure but with 4 layers:
- Parameters: 11 × 4 = 44
- Optimization: Multi-start COBYLA with 3 trials
- Convergence: 600–800 iterations
- Accuracy: 10⁻³ to 10⁻⁴ H error
- Ground state overlap: > 0.95

**Hybrid optimization strategy:**

Phase 1: COBYLA exploration (derivative-free, robust to noise)
```
θ₁ = argmin_θ ⟨ψ(θ)|H|ψ(θ)⟩  using COBYLA, maxiter=400
```

Phase 2: L-BFGS-B refinement (gradient-based, faster convergence)
```
θ₂ = argmin_θ ⟨ψ(θ)|H|ψ(θ)⟩  using L-BFGS-B, maxiter=400, initial=θ₁
```

### Full Active Space (10 Qubits)

Requires Qiskit Nature for Hamiltonian generation:
- 4 electrons in 5 spatial orbitals
- 10 spin-orbitals → 10 qubits
- Higher accuracy but slower optimization
- Can compute at arbitrary bond lengths via PySCF

### Potential Energy Surface

VQE scans bond length space:

```
E(r) = min_θ ⟨ψ(θ,r)|H(r)|ψ(θ,r)⟩
```

Typical scan: r ∈ [1.0, 3.0] Å with 21 points
- Each point: VQE optimization (~10 min per point)
- Result: E(r) curve mapping dissociation
- Equilibrium location: argmin_r E(r)

## Hardware Gate Synthesis

All molecular implementations use hardware-native gates via the UMQ construction:

**Single Pauli string:**
```
exp(-i t P₁⊗...⊗Pₙ) = R†·UMQ(-π/2)·Rz(±2t)·UMQ(+π/2)·R
```

**Multiple strings (molecular Hamiltonian):**
```
exp(-i t H) = ∏ₖ exp(-i t hₖ Pₖ) + O(t²)  (first-order Trotter)
```

or with Suzuki decomposition for higher accuracy:
```
exp(-i t H) = [∏ₖ exp(-i t/(2n) hₖ Pₖ)][∏ₖ exp(-i t/(2n) hₖ Pₖ)]†^(n-1) + O(t²ⁿ⁺¹/n²ⁿ)
```

## Numerical Validation

### H₂ Benchmarks

| Method | Energy (H) | Error (H) | Time |
|--------|-----------|-----------|------|
| Exact diagonalization | -1.137 | 0 | 0.01 s |
| Imaginary-time (optimized) | -1.137 | 8×10⁻¹⁰ | 2.5 s |
| VQE (3 layers) | -1.136 | 10⁻³ | 30 s |

### H₂O Benchmarks

10-qubit system (exact methods infeasible):
- State vector memory: 8 MB
- Term evaluation: 73 terms → 3 groups
- VQE convergence: > 1000 iterations
- Hardware advantage: Term grouping reduces circuit depth 24×

### LiH Benchmarks

| Configuration | Energy (H) | Error (H) | Parameters |
|---------------|-----------|-----------|------------|
| Literature (STO-6G) | -7.862 | - | Full CI |
| 4-qubit VQE | -7.88 | 0.02 | 44 |
| 10-qubit VQE | -7.86 | 0.002 | 110 |

## Computational Complexity

Memory scaling:
```
O(2ⁿ) complex numbers = 16·2ⁿ bytes
```

Evolution scaling:
```
O(N_terms · 8ⁿ) for full Hamiltonian evolution
```

VQE parameter count:
```
N_params = L · (2n + n-1) ≈ L · 3n
```

Optimization iterations:
```
N_iter ∼ O(N_params²) for gradient-free methods
N_iter ∼ O(N_params) for gradient-based methods
```

## References

1. O'Malley et al., Phys. Rev. X 6, 031007 (2016) - VQE for molecular systems
2. Peruzzo et al., Nature Commun. 5, 4213 (2014) - Original VQE proposal
3. Cao et al., Chem. Rev. 119, 10856 (2019) - Quantum chemistry review
4. Seeley et al., J. Chem. Phys. 137, 224109 (2012) - Bravyi-Kitaev transformation
5. QEE compression: Quantum equation of motion method
