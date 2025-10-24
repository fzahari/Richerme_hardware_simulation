# C++ TWA Implementation for H2 and H2O Molecules

This directory contains high-performance C++ implementations of the Truncated Wigner Approximation (TWA) simulations for molecular quantum dynamics.

## Overview

The C++ implementation provides:
- **H2 molecule simulation**: 4-qubit system with empirical parametrization
- **H2O molecule simulation**: 10-qubit system with multi-term Hamiltonian
- **Hardware-realistic dissipation**: T1 decay and T2 dephasing for 171Yb+ ions
- **OpenMP parallelization**: Optional CPU multi-threading
- **Portable build system**: CMake for cross-platform compilation

## Architecture

### Core Components

1. **TWA Framework** (`twa_framework.hpp/cpp`)
   - Base TWA simulator with RK4 integration
   - Stochastic noise generation (Wiener processes)
   - Dissipation channels (T1 decay, T2 dephasing)
   - Spin conservation checks

2. **H2 Simulator** (`h2_twa_simulator.hpp/cpp`)
   - 4-qubit hydrogen molecule
   - Empirical parametrization as function of bond distance
   - Compare ideal vs dissipative dynamics

3. **H2O Simulator** (`h2o_twa_simulator.hpp/cpp`)
   - 10-qubit water molecule
   - Term-based Hamiltonian storage
   - Numerical stability mechanisms

4. **Main Programs** (`main_h2.cpp`, `main_h2o.cpp`)
   - Command-line interfaces
   - Configurable trajectory counts
   - Console output with statistics

## Build Instructions

### Prerequisites

- **C++ Compiler**: C++17 compatible (GCC ≥ 7, Clang ≥ 5, MSVC ≥ 2017)
- **CMake**: Version 3.15 or higher
- **Eigen3**: Linear algebra library
- **OpenMP**: (Optional) For CPU parallelization

### Installing Dependencies

**macOS**:
```bash
brew install cmake eigen
```

**Ubuntu/Debian**:
```bash
sudo apt-get install cmake libeigen3-dev
```

**OpenMP** (optional, usually included with compiler):
- GCC: Included by default
- Clang on macOS: `brew install libomp`

### Building

```bash
# Navigate to cpp directory
cd cpp

# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build (use -j for parallel compilation)
make -j4

# Executables will be in the build directory:
# - h2_twa_simulation
# - h2o_twa_simulation
```

### Build Options

**Release build** (optimized, ~10x faster):
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

**Debug build** (for development):
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
```

**Specify Eigen path** (if not found automatically):
```bash
cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
```

## Running Simulations

### H2 Molecule Simulation

```bash
# Default: 500 trajectories
./h2_twa_simulation

# Custom trajectory count
./h2_twa_simulation 1000

# Expected runtime: ~30-60 seconds for 500 trajectories
```

**Output**:
```
======================================================================
H2 MOLECULE SIMULATION WITH TWA DISSIPATION (C++)
======================================================================
...
Features:
  - 4-qubit quantum chemistry Hamiltonian
  - Hardware-realistic T1/T2 decoherence (171Yb+)
  - Stochastic trajectory averaging
  - Comparison with ideal (no dissipation) case
======================================================================

Running H2 TWA simulation:
  Qubits: 4
  Time steps: 100, dt = 0.2
  Trajectories: 500
  T1 decay: ON/OFF
  T2 dephasing: ON/OFF

✓ Simulation complete in X.XX seconds
  Final energy: -1.XXX ± 0.XXX H
```

### H2O Molecule Simulation

```bash
# Default: 300 trajectories
./h2o_twa_simulation

# Custom trajectory count
./h2o_twa_simulation 500

# Expected runtime: ~2-5 minutes for 300 trajectories
```

**Output**: Similar format to H2, but with 10 qubits and longer runtime.

## Performance

### Expected Performance (Intel i7/i9 or equivalent)

| System | Trajectories | Time (Release) | Time (Debug) |
|--------|-------------|----------------|--------------|
| H2     | 500         | ~30-60 sec     | ~5-10 min    |
| H2O    | 300         | ~2-5 min       | ~20-40 min   |

### Optimization Tips

1. **Always use Release build** for production runs:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

2. **OpenMP parallelization**: Automatically enabled if found
   - Check build output for: "OpenMP found - enabling parallel trajectory execution"
   - Control threads: `export OMP_NUM_THREADS=4`

3. **Compiler flags**: Already optimized in CMakeLists.txt
   - `-O3`: Aggressive optimization
   - `-march=native`: Use CPU-specific instructions

### Scaling

- **Memory**: O(n_qubits × n_steps) per trajectory
- **Time**: Linear in n_trajectories (embarrassingly parallel)
- **Recommended**:
  - H2: 500-2000 trajectories
  - H2O: 300-1000 trajectories (more expensive due to 10 qubits)

## Comparison with Python Versions

### C++ Advantages

1. **Performance**: ~10-50x faster than Python (CPU-only)
2. **Memory efficiency**: Lower overhead than NumPy
3. **Parallelization**: OpenMP multi-threading
4. **Portability**: Single binary, no Python environment needed

### Python Advantages

1. **Visualization**: Matplotlib plotting built-in
2. **Flexibility**: Easier to modify and experiment
3. **GPU acceleration**: CuPy version for massive speedup
4. **Interactive**: Jupyter notebook support

### Which to Use?

- **C++ version**: Production runs, HPC clusters, batch processing
- **Python CPU version**: Development, analysis, visualization
- **Python GPU version**: Largest trajectory counts (2000+), best performance

## Output Format

The C++ version outputs results to **console only**. Key information includes:

1. **Configuration**: System size, parameters, dissipation settings
2. **Progress updates**: Every 50 trajectories
3. **Final statistics**: Average energy, standard deviation, magnetization
4. **Warnings**: Numerical instabilities (rare with current parameters)

### Saving Results

To save output to file:
```bash
./h2_twa_simulation > h2_results.txt
./h2o_twa_simulation 500 > h2o_results_500traj.txt
```

### Visualization

For plotting, use the Python versions or export data for post-processing:
```bash
# Run C++ for performance, then analyze with Python
./h2_twa_simulation > results.txt
python analyze_cpp_output.py results.txt
```

## Code Structure

### TWA Framework Core

```cpp
namespace twa {
    // 3D spin vector
    struct Spin3D {
        double x, y, z;
        Spin3D cross(const Spin3D& other) const;
        double norm_squared() const;
    };

    // Dissipation channel
    enum class ChannelType { DECAY, PUMPING, DEPHASING };

    // Main TWA simulator
    class TWASpinSimulator {
        // Initialize with system size and trajectory count
        TWASpinSimulator(int n_qubits, int n_trajectories);

        // Add dissipation channels
        void add_dissipation(ChannelType type, double rate, std::vector<int> qubits);

        // RK4 integration step
        std::vector<Spin3D> rk4_step(double t, const std::vector<Spin3D>& spins,
                                      double dt, const HamiltonianGradientFunc& grad_func,
                                      const NoiseData& noise);
    };
}
```

### H2 Simulator Usage

```cpp
#include "h2_twa_simulator.hpp"

int n_trajectories = 500;
twa::H2TWASimulator h2_twa(n_trajectories);

// Run comparison: ideal vs T1+T2 dissipation
auto results = h2_twa.compare_with_ideal(
    0.74,   // Bond distance (Angstroms)
    20.0,   // Total time (a.u.)
    100     // Number of steps
);
```

### H2O Simulator Usage

```cpp
#include "h2o_twa_simulator.hpp"

int n_trajectories = 300;
double energy_scale = 1e15;  // Match model units
twa::H2OTWASimulator h2o_twa(n_trajectories, energy_scale);

// Run comparison
auto results = h2o_twa.compare_dissipation_effects(
    5.0,    // Total time (a.u.)
    200     // Number of steps
);
```

## Hardware Parameters (171Yb+ Ions)

Built-in hardware specifications from experimental systems:

- **T1 (energy relaxation)**: 1000 s (effectively infinite)
- **T2 (dephasing)**: 1.0 s
- **Derived rates**:
  - γ_decay = 1/(2·T1) = 5×10⁻⁴ Hz
  - κ_dephasing = 1/T2 = 1.0 Hz

These are automatically scaled by `energy_scale` parameter to match model units.

## Troubleshooting

### Build Errors

**"Eigen3 not found"**:
```bash
# Install Eigen3
brew install eigen  # macOS
sudo apt-get install libeigen3-dev  # Ubuntu

# Or specify path manually
cmake -DEIGEN3_INCLUDE_DIR=/path/to/eigen3 ..
```

**"C++17 required"**:
```bash
# Ensure compiler supports C++17
g++ --version  # Should be ≥ 7.0
clang++ --version  # Should be ≥ 5.0

# Or specify compiler explicitly
cmake -DCMAKE_CXX_COMPILER=g++-11 ..
```

### Runtime Issues

**Numerical instabilities** (rare):
```
WARNING: Trajectory X became unstable at step Y
```
- Expected for <5% of trajectories
- Handled gracefully (excluded from averages)
- If >20% fail, check time step or parameters

**Slow performance**:
1. Verify Release build: `cmake -DCMAKE_BUILD_TYPE=Release ..`
2. Check OpenMP: Should see "OpenMP found" during cmake
3. Reduce trajectory count for testing

**Memory issues** (very rare):
- H2O with 1000+ trajectories may use ~1-2 GB
- Reduce trajectory count if needed

## Technical Notes

### Numerical Methods

- **Integration**: 4th-order Runge-Kutta (RK4)
- **Stochastic processes**: Stratonovich interpretation
- **Random number generation**: Mersenne Twister (std::mt19937)
- **Spin renormalization**: Optional, every 10 steps for H2O

### Differences from Python Version

1. **No visualization**: Console output only
2. **Single trajectory per thread**: OpenMP parallelizes outer loop
3. **Fixed random seed**: Currently uses default seed (for reproducibility testing)
4. **No NaN handling in averages**: Uses manual checks instead of np.nanmean()

### Future Enhancements

Potential additions (not yet implemented):
- CUDA kernels for GPU acceleration
- HDF5/CSV output for post-processing
- Python bindings (pybind11)
- Adaptive time stepping
- MPI for distributed computing

## References

1. **TWA Method**: Polkovnikov, A. (2010). "Truncated Wigner approximation for quantum systems"
2. **Hardware specifications**: Richerme group experimental systems (171Yb+ ions)
3. **Python implementation**: See `../rich_sim_h2_twa.py` and `../rich_sim_h2o_twa.py`

## Contact

For questions about the C++ implementation, see the main project README.md.
