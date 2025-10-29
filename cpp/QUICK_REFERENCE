# Quick Reference Guide

**Richerme Ion Analog C++ Library** - Essential commands and code snippets

## Installation

```bash
# macOS
brew install eigen
cd cpp && chmod +x verify_build.sh && ./verify_build.sh

# Linux
sudo apt-get install libeigen3-dev
cd cpp && mkdir build && cd build && cmake .. && make && ctest
```

## Building Your Code

### Method 1: Direct compilation
```bash
g++ -std=c++17 -O3 -march=native \
    your_code.cpp richerme_ion_analog.cpp \
    -I/usr/local/include/eigen3 \
    -o your_program
```

### Method 2: CMake integration
```cmake
# Your CMakeLists.txt
add_subdirectory(path/to/cpp)
target_link_libraries(your_target PRIVATE richerme_ion_analog)
```

## Essential Code Patterns

### 1. Basic Gate Synthesis
```cpp
#include "richerme_ion_analog.h"
using namespace richerme;

// Z⊗X⊗X pattern
std::vector<char> axes = {'Z', 'X', 'X'};
Matrix U = GateSynthesis::n_body_string(axes, 0.5);

// Verify fidelity
Matrix U_target = GateSynthesis::target_pauli_string_unitary("ZXX", 0.5);
double error = GateSynthesis::unitary_distance(U_target, U);
// error ≈ 1e-15
```

### 2. Arbitrary Pauli Strings
```cpp
// Any combination: ZYZ, XYX, YZY, etc.
std::vector<char> axes = {'Z', 'Y', 'Z'};
Matrix U = GateSynthesis::n_body_string_arbitrary(axes, 0.3);
```

### 3. UMQ Global Entangling Gate
```cpp
int n_qubits = 3;
double chi = M_PI / 4.0;  // Interaction strength
Matrix U_umq = GateSynthesis::UMQ(n_qubits, chi);

// Verify unitarity: U†U = I
int dim = 1 << n_qubits;  // 2^n
Matrix identity = Matrix::Identity(dim, dim);
Matrix product = U_umq.adjoint() * U_umq;
double unitarity_error = (product - identity).norm();
// unitarity_error ≈ 1e-15
```

### 4. Single-Qubit Rotations
```cpp
int n = 3;        // Total qubits
int qubit = 0;    // Target qubit
double angle = M_PI / 2;

Matrix Rx_gate = GateSynthesis::Rx(n, qubit, angle);
Matrix Ry_gate = GateSynthesis::Ry(n, qubit, angle);
Matrix Rz_gate = GateSynthesis::Rz(n, qubit, angle);
```

### 5. Interaction Graph Accessibility
```cpp
int N = 5;  // Number of ions
RealMatrix B = InteractionEngineering::compute_sinusoidal_modes(N);

// All-to-all interaction
RealMatrix J_desired = RealMatrix::Ones(N, N) - RealMatrix::Identity(N, N);

// Check accessibility
bool accessible = InteractionEngineering::is_accessible(J_desired, B);

if (accessible) {
    RealVector weights = InteractionEngineering::get_mode_weights_if_accessible(
        J_desired, B);
    // Use weights to program hardware
}
```

### 6. Mode Weight Optimization
```cpp
RealMatrix J_desired = /* ... your target interaction ... */;
RealMatrix B = InteractionEngineering::compute_sinusoidal_modes(N);

auto result = InteractionEngineering::optimize_mode_weights(
    J_desired, B, "least_squares");

std::cout << "Achieved infidelity: " << result.infidelity << "\n";
std::cout << "Mode weights: " << result.weights.transpose() << "\n";
```

### 7. Multi-Mode Coupling
```cpp
int N = 5;
RealMatrix B = InteractionEngineering::compute_sinusoidal_modes(N);
RealVector omega(N);  // Mode frequencies
RealVector mus(2);    // Drive detunings
RealVector Omegas(2); // Drive strengths

IonTrapHardware hw;
double R = hw.recoil_frequency();

RealMatrix J = InteractionEngineering::Jij_from_multimode(
    B, omega, mus, Omegas, R);
```

### 8. Hardware Specifications
```cpp
IonTrapHardware hw;

std::cout << "Hyperfine splitting: "
          << hw.hyperfine_splitting / 1e9 << " GHz\n";
std::cout << "T2 coherence: " << hw.T2_coherence << " s\n";
std::cout << "Single-qubit fidelity: " << hw.single_qubit_fidelity << "\n";
std::cout << "Recoil frequency: " << hw.recoil_frequency() / 1e3 << " kHz\n";
```

### 9. Error Checking
```cpp
try {
    Matrix U = GateSynthesis::n_body_string({'Z', 'Y', 'X'}, 0.5);
    // This will throw - invalid pattern (only first can be non-X)
} catch (const std::invalid_argument& e) {
    std::cerr << "Error: " << e.what() << "\n";
}
```

### 10. High-Level Interface
```cpp
// Create synthesizer with hardware-native gates
RichermeIonAnalog synthesizer(true);  // use_hardware_gates = true

// Automatically chooses best method
std::vector<char> axes = {'Z', 'X', 'X'};
Matrix U = synthesizer.synthesize_gate(axes, 0.5);

// Check interaction accessibility
RealMatrix J = /* ... */;
RealMatrix B = /* ... */;
bool accessible = synthesizer.check_accessibility(J, B);
```

## Common Patterns

### Compile and run
```bash
g++ -std=c++17 -O3 -march=native \
    my_program.cpp richerme_ion_analog.cpp \
    -I/usr/local/include/eigen3 \
    -o my_program && ./my_program
```

### Run tests
```bash
cd build
ctest --verbose
```

### Run example
```bash
cd build
./example_basic
```

### Clean rebuild
```bash
rm -rf build
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)  # macOS
make -j$(nproc)              # Linux
```

## Type Aliases

```cpp
using Complex = std::complex<double>;
using Matrix = Eigen::MatrixXcd;      // Complex matrix
using Vector = Eigen::VectorXcd;      // Complex vector
using RealMatrix = Eigen::MatrixXd;   // Real matrix
using RealVector = Eigen::VectorXd;   // Real vector
```

## Common Eigen Operations

```cpp
// Matrix creation
Matrix U = Matrix::Identity(8, 8);
Matrix H = Matrix::Zero(8, 8);

// Matrix operations
Matrix U_dag = U.adjoint();           // Hermitian conjugate
Matrix product = A * B;               // Matrix multiplication
double norm = U.norm();               // Frobenius norm
Complex trace = U.trace();            // Trace

// Element access
Complex element = U(i, j);            // Read element
U(i, j) = Complex(1.0, 0.0);         // Write element

// Submatrices
Matrix block = U.block(0, 0, 4, 4);  // Extract 4×4 block
```

## Performance Tips

### 1. Compiler optimization
```bash
# Maximum performance
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native -ffast-math" ..
```

### 2. Pre-compute repeated operations
```cpp
// BAD - recomputes every time
for (int i = 0; i < 1000; ++i) {
    Matrix U = GateSynthesis::n_body_string({'Z','X','X'}, t);
}

// GOOD - compute once
Matrix U = GateSynthesis::n_body_string({'Z','X','X'}, t);
for (int i = 0; i < 1000; ++i) {
    // Use U
}
```

### 3. Use const references for large objects
```cpp
void process_unitary(const Matrix& U) {  // Reference, no copy
    // ...
}
```

## Precision Guidelines

```cpp
// Conservative thresholds
const double STRICT_TOLERANCE = 1e-13;
const double NORMAL_TOLERANCE = 1e-11;
const double LOOSE_TOLERANCE = 1e-9;

// Usage
double error = GateSynthesis::unitary_distance(U_target, U_synth);
if (error < NORMAL_TOLERANCE) {
    // Success
}
```

## Debugging

### Print matrices
```cpp
std::cout << "Matrix U:\n" << U << "\n";
```

### Check unitarity
```cpp
bool is_unitary(const Matrix& U) {
    Matrix identity = Matrix::Identity(U.rows(), U.cols());
    double error = (U.adjoint() * U - identity).norm();
    return error < 1e-12;
}
```

### Measure timing
```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
Matrix U = GateSynthesis::n_body_string({'Z','X','X'}, 0.5);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    end - start).count();
std::cout << "Time: " << duration << " μs\n";
```

## Memory Management

```cpp
// Eigen matrices are stack-allocated for small sizes
// For large matrices, they use heap automatically

int n = 10;  // 10 qubits → 1024×1024 matrices
Matrix U = GateSynthesis::UMQ(n, M_PI/4);  // ~8 MB, heap-allocated

// Memory is automatically freed when U goes out of scope
```

## Common Errors

### 1. Wrong pattern for n_body_string()
```cpp
// ERROR: All qubits after first must be X
Matrix U = GateSynthesis::n_body_string({'Z', 'Y', 'Z'}, t);
// Throws: std::invalid_argument

// FIX: Use n_body_string_arbitrary()
Matrix U = GateSynthesis::n_body_string_arbitrary({'Z', 'Y', 'Z'}, t);
```

### 2. Dimension mismatch
```cpp
Matrix A(8, 8);
Matrix B(4, 4);
Matrix C = A * B;  // ERROR: incompatible dimensions
```

### 3. Forgetting to link Eigen3
```bash
# ERROR:
g++ my_program.cpp -o my_program  # Missing Eigen

# FIX:
g++ my_program.cpp -I/usr/local/include/eigen3 -o my_program
```

## Further Reading

- **Full API**: `README.md`
- **Build instructions**: `BUILD_INSTRUCTIONS.md`
- **Numerical precision**: `NUMERICAL_PRECISION.md`
- **Project status**: `STATUS.md`
- **Examples**: `examples/example_basic.cpp`
- **Tests**: `tests/test_richerme.cpp`

---

**Quick tip**: Start with `verify_build.sh` to ensure everything works, then study `example_basic.cpp` for usage patterns.
