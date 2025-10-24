# Richerme Quantum Hardware

Python library for synthesizing quantum gates on trapped-ion analog hardware using the Richerme protocol.

> **Note**: This library has been consolidated from a dual-library structure (`richerme_ion_analog.py` + `richerme_ion_analog_extended.py`) into a single unified library (`richerme_ion_analog.py`) that includes all features. See [LIBRARY_CONSOLIDATION.md](docs/LIBRARY_CONSOLIDATION.md) for migration details.

## Overview

This library implements quantum gate synthesis techniques for trapped-ion analog quantum computers, specifically focusing on efficient construction of multi-qubit Pauli string unitaries using global Mølmer-Sørensen (MS) operations combined with single-qubit rotations.

**Key Features:**
- Hardware-native gate synthesis for trapped-ion systems (171Yb+ ions)
- Arbitrary n-body Pauli string operations
- Multi-mode global driving with interaction graph engineering
- Accessibility checking and mode weight optimization
- Full trapped-ion physics simulators
- Molecular quantum chemistry applications (H2, H2O)
- **NEW**: Truncated Wigner Approximation (TWA) for dissipative dynamics
- **NEW**: GPU-accelerated TWA simulations (10-100x speedup)

## Installation

### Recommended Environment

```bash
# Create and activate conda environment
conda create -n qiskit-fresh python=3.9 -y
conda activate qiskit-fresh

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Requirements

- Python >= 3.8
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- pytest >= 6.0 (for testing)
- qiskit >= 0.34.0 (optional, for quantum chemistry)

## Project Structure

```
Richerme_Quantum_Hardware/
├── richerme_ion_analog.py              # Complete gate synthesis library (unified)
├── rich_sim.py                         # Full trapped-ion physics simulator
├── rich_sim_h2.py                      # H2 molecule simulation
├── rich_sim_h2o.py                     # H2O molecule simulation
│
├── twa_framework.py                    # TWA framework for dissipative dynamics (CPU)
├── rich_sim_h2_twa.py                  # H2 with TWA dissipation (CPU)
├── rich_sim_h2o_twa.py                 # H2O with TWA dissipation (CPU)
├── test_twa_implementation.py          # TWA validation tests
│
├── cudaq_twa_framework.py              # GPU-accelerated TWA framework (10-100x faster)
├── cudaq_rich_sim_h2_twa.py            # GPU-accelerated H2 TWA
├── cudaq_rich_sim_h2o_twa.py           # GPU-accelerated H2O TWA
├── demo_cudaq_speedup.py               # CPU vs GPU benchmark
│
├── setup.py                            # Package configuration
├── requirements.txt                    # Dependencies (CPU)
├── requirements_gpu.txt                # GPU dependencies (+ CuPy)
├── README.md                           # Main documentation
├── DEVELOPMENT.md                      # Development guidelines
│
├── docs/                               # Documentation
│   ├── LIBRARY_EXPLANATION.md         # What do these libraries do?
│   ├── LIBRARY_CONSOLIDATION.md       # Migration guide
│   ├── INTEGRATION_GUIDE.md           # Integration guide
│   ├── SIMULATOR_ANALYSIS.md          # Simulator comparison
│   ├── H2_IMAGINARY_TIME_IMPROVEMENT.md # 2.7M× error reduction
│   ├── H2_UPDATE_SUMMARY.md           # H2 updates
│   ├── H2O_UPDATE_SUMMARY.md          # H2O updates
│   ├── TWA_NUMERICAL_FIXES.md         # TWA stability improvements
│   ├── TWA_README.md                  # TWA documentation (CPU & GPU)
│   ├── CUDAQ_TWA_README.md            # GPU acceleration guide
│   ├── CUDAQ_IMPLEMENTATION_SUMMARY.md # GPU implementation details
│   └── CPP_TWA_SUMMARY.md             # C++ TWA implementation
│
├── tests/                              # Test suite
│   ├── test_richerme_ion_analog.py    # Core library tests (63 tests)
│   ├── test_h2_gates.py               # H2 simulator tests
│   ├── test_h2o_gates.py              # H2O simulator tests
│   └── test_twa_implementation.py     # TWA validation (5 tests)
│
└── examples/                           # Demonstration scripts
    ├── demo_zxx.py                    # Basic synthesis demo
    ├── demo_extended_features.py      # Extended features demo
    └── demo_cudaq_speedup.py          # GPU speedup benchmark
```

## Quick Start

### Basic Gate Synthesis

```python
from richerme_ion_analog import n_body_string, target_pauli_string_unitary, unitary_distance

# Synthesize a 3-qubit Z⊗X⊗X operation
pauli_string = ['Z', 'X', 'X']
evolution_time = 0.5

# Generate using hardware-native gates (UMQ + single-qubit rotations)
U_synthesized = n_body_string(pauli_string, evolution_time)

# Compare with ideal target
U_target = target_pauli_string_unitary('ZXX', evolution_time)

# Check fidelity
fidelity_error = unitary_distance(U_target, U_synthesized)
print(f"Fidelity error: {fidelity_error:.2e}")  # ~1e-15 (machine precision)
```

### Arbitrary Pauli Strings

```python
from richerme_ion_analog import n_body_string_arbitrary

# Synthesize arbitrary Pauli patterns
U_zyz = n_body_string_arbitrary(['Z', 'Y', 'Z'], 0.3)
U_yyx = n_body_string_arbitrary(['Y', 'Y', 'X'], 0.5)
U_xyz = n_body_string_arbitrary(['X', 'Y', 'Z'], 0.2)
```

### Interaction Graph Engineering

```python
from richerme_ion_analog import (
    is_accessible, optimize_mode_weights, compute_sinusoidal_modes
)
import numpy as np

# Check if interaction graph is accessible
N = 5
B = compute_sinusoidal_modes(N)  # Mode matrix for equispaced ions

# All-to-all interaction (accessible)
J_all_to_all = np.ones((N, N)) - np.eye(N)
print(f"All-to-all accessible: {is_accessible(J_all_to_all, B)}")  # True

# Power-law interaction (approximate)
J_power_law = np.zeros((N, N))
for i in range(N):
    for j in range(i+1, N):
        J_power_law[i, j] = J_power_law[j, i] = 1.0 / abs(i - j)**1.0

result = optimize_mode_weights(J_power_law, B)
print(f"Power-law infidelity: {result['infidelity']:.6f}")  # ~0.002
```

### Molecular Simulation

```python
from rich_sim_h2 import H2TimeEvolutionSimulator
from rich_sim import TrappedIonSimulator

# Create trapped-ion system
ion_sim = TrappedIonSimulator(
    num_ions=4,
    trap_frequency_x=2*np.pi*5e6,  # 5 MHz radial
    trap_frequency_z=2*np.pi*0.5e6,  # 0.5 MHz axial
)

# Run H2 VQE simulation with hardware-realistic gates
h2_sim = H2TimeEvolutionSimulator(ion_sim, use_hardware_gates=True)
result = h2_sim.run_vqe(bond_length=0.74, num_iterations=100)

print(f"VQE energy: {result['energy']:.6f} Hartree")
print(f"Error vs exact: {result['error']:.6f} Hartree")
```

### Dissipative Dynamics with TWA (NEW!)

Simulate realistic decoherence effects (T1/T2) using the Truncated Wigner Approximation:

```python
from rich_sim_h2_twa import H2_TWA_Simulator

# Create TWA simulator with 500 stochastic trajectories
h2_twa = H2_TWA_Simulator(n_trajectories=500)

# Compare ideal vs dissipative dynamics
results = h2_twa.compare_with_ideal(r=0.74, total_time=20.0)

print(f"Ideal energy:    {results['ideal']['avg_energies'][-1]:.6f} H")
print(f"With T1+T2:      {results['full']['avg_energies'][-1]:.6f} H")
print(f"Dissipation:     {results['full']['avg_energies'][-1] - results['ideal']['avg_energies'][-1]:.6f} H")
```

**GPU-Accelerated (10-100x faster!)**:

```python
from cudaq_rich_sim_h2_twa import CUDAQ_H2_TWA_Simulator

# GPU version handles many more trajectories
h2_gpu = CUDAQ_H2_TWA_Simulator(n_trajectories=2000, use_gpu=True)
results = h2_gpu.compare_with_ideal(r=0.74, total_time=20.0)
# Runs in ~45 seconds (vs ~8 minutes on CPU!)
```

**Requirements for GPU**:
```bash
pip install cupy-cuda12x  # For CUDA 12.x (or cupy-cuda11x for CUDA 11.x)
```

See [TWA_README.md](docs/TWA_README.md) and [CUDAQ_TWA_README.md](docs/CUDAQ_TWA_README.md) for details.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_richerme_ion_analog.py -v  # 63 tests, ~0.5s
pytest tests/test_h2_gates.py -v             # Hardware gate validation

# Quick test (minimal output)
pytest tests/ -q
```

**Expected output:**
```
================================ 63 passed in 0.52s ================================
```

## Running Examples

```bash
# Basic synthesis demonstration
python examples/demo_zxx.py

# Extended features (arbitrary Pauli, optimization, hardware params)
python examples/demo_extended_features.py
```

## Hardware Specifications (171Yb+)

The library is designed for 171Yb+ trapped-ion systems with the following specifications:

| Parameter | Value |
|-----------|-------|
| **Qubit encoding** | Hyperfine states \|F=0,mF=0⟩ and \|F=1,mF=0⟩ |
| **Hyperfine splitting** | 12.6 GHz |
| **T1 coherence** | Effectively infinite |
| **T2 coherence** | > 1 second |
| **Single-qubit fidelity** | 99.8% |
| **Two-qubit fidelity** | 97% |
| **Laser wavelength** | 369.5 nm |
| **Trap frequencies** | ωx = 10 MHz, ωy = 10.05 MHz, ωz = 0.1 MHz |
| **System scale** | 30-100 ions |

## Key Concepts

### UMQ Construction

The library synthesizes arbitrary multi-qubit Pauli operations using the **UMQ–Rz–UMQ pattern**:

```
U(P₁⊗P₂⊗...⊗Pₙ, t) = R_post · UMQ(-π/2) · Rz(±2t) · UMQ(+π/2) · R_pre
```

Where:
- **UMQ(χ)**: Universal Multi-Qubit gate = exp(-i(χ/4)(∑ᵢ Xᵢ)²) - global MS operation
- **Rz**: Single-qubit Z rotations
- **R_pre, R_post**: Basis rotations to handle Y and Z Pauli operators

### Multi-Mode Global Driving

The library implements **Equation 4** from Richerme 2025 for realistic Ising couplings:

```
J_ij = ∑ₖ [∑ₘ (Ωₘ²R)/(μₘ² - ωₖ²)] · Bᵢₖ · Bⱼₖ
```

Where:
- **Ωₘ**: Rabi frequency of tone m
- **μₘ**: Detuning of tone m
- **ωₖ**: Normal mode frequency k
- **R**: Recoil frequency
- **B**: Mode participation matrix

### Accessibility Criterion

**Equation 14** from Kyprianidis 2024:

```
J is accessible ⟺ B^T · J · B is diagonal
```

If accessible, exact interaction graph J can be realized. If not, `optimize_mode_weights()` finds best approximation.

## Performance Scaling

| System Size | Hilbert Space | Memory | Synthesis Time |
|-------------|---------------|--------|----------------|
| 2 qubits | 4 | ~100 bytes | < 1 ms |
| 3 qubits | 8 | ~500 bytes | ~5 ms |
| 4 qubits | 16 | ~2 KB | ~20 ms |
| 10 qubits (H2O) | 1024 | ~8 MB | ~5 s |

Memory scales as O(2ⁿ), computation as O(8ⁿ).

## Documentation

- **[LIBRARY_EXPLANATION.md](docs/LIBRARY_EXPLANATION.md)**: **What do these libraries do?** Comprehensive explanation of the core algorithms
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Development guidelines, coding standards, testing, comprehensive project documentation
- **[INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)**: How to integrate extended features
- **[SIMULATOR_ANALYSIS.md](docs/SIMULATOR_ANALYSIS.md)**: Detailed simulator comparison
- **[H2_IMAGINARY_TIME_IMPROVEMENT.md](docs/H2_IMAGINARY_TIME_IMPROVEMENT.md)**: 2.7M× error reduction in H2 ground state finding

## Scientific References

1. **Richerme et al. (2025)** - "Multi-mode global driving techniques for trapped-ion quantum simulation"
   - Quantum Sci. Technol. 10, 035046
   - Implements Equation 4 for multi-mode coupling

2. **Kyprianidis et al. (2024)** - "Interaction graph engineering with trapped-ion quantum simulators"
   - New J. Phys. 26, 023033
   - Implements Equations 14 (accessibility) and 18 (sinusoidal modes)

## Contributing

This is a research project for trapped-ion quantum simulation. Contributions should follow:
- **SOLID principles** (see DEVELOPMENT.md)
- **Clean code practices** (small functions, descriptive names, no side effects)
- **Test coverage** (all new features must have tests)
- **Documentation** (docstrings and examples)

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## License

MIT License (see LICENSE file)

## Author

Federico Zahariev

## Acknowledgments

Based on the Richerme protocol for trapped-ion quantum gate synthesis. Implements techniques from:
- Richerme group (Indiana University)
- Kyprianidis et al. (University of Innsbruck)
- Hardware specifications from 171Yb+ trapped-ion systems

---

**For comprehensive project documentation, see [DEVELOPMENT.md](DEVELOPMENT.md)**
