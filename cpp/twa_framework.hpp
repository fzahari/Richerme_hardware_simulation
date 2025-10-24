/**
 * Truncated Wigner Approximation (TWA) Framework - C++ Header
 *
 * GPU-accelerated TWA for dissipative spin systems using CUDA.
 *
 * Based on: "User-Friendly Truncated Wigner Approximation for Dissipative Spin Dynamics"
 * Hosseinabadi, Chelpanova, and Marino, PRX Quantum 6, 030344 (2025)
 */

#ifndef TWA_FRAMEWORK_HPP
#define TWA_FRAMEWORK_HPP

#include <vector>
#include <string>
#include <memory>
#include <random>
#include <functional>

namespace twa {

// Forward declarations
struct Spin3D;
class DissipationChannel;
class TWASpinSimulator;

/**
 * 3D spin vector representation
 */
struct Spin3D {
    double x, y, z;

    Spin3D() : x(0.0), y(0.0), z(0.0) {}
    Spin3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    double norm_squared() const { return x*x + y*y + z*z; }
    void normalize(double target = 3.0);

    // Cross product
    Spin3D cross(const Spin3D& other) const {
        return Spin3D(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    // Scalar multiplication
    Spin3D operator*(double scalar) const {
        return Spin3D(x * scalar, y * scalar, z * scalar);
    }

    // Addition
    Spin3D operator+(const Spin3D& other) const {
        return Spin3D(x + other.x, y + other.y, z + other.z);
    }

    // In-place addition
    Spin3D& operator+=(const Spin3D& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

/**
 * Dissipation channel types
 */
enum class ChannelType {
    DECAY,      // T1 energy relaxation
    PUMPING,    // Spin pumping
    DEPHASING   // T2 dephasing
};

/**
 * Dissipation channel descriptor
 */
class DissipationChannel {
public:
    ChannelType type;
    double rate;              // γ for decay/pumping, κ for dephasing
    std::vector<int> qubits;  // Affected qubit indices

    DissipationChannel(ChannelType t, double r, const std::vector<int>& q)
        : type(t), rate(r), qubits(q) {}
};

/**
 * Noise data for stochastic evolution
 */
struct NoiseData {
    std::vector<double> decay_x;      // Decay noise (x component)
    std::vector<double> decay_y;      // Decay noise (y component)
    std::vector<double> dephasing;    // Dephasing noise

    NoiseData(int n_qubits)
        : decay_x(n_qubits, 0.0),
          decay_y(n_qubits, 0.0),
          dephasing(n_qubits, 0.0) {}
};

/**
 * Hamiltonian gradient function signature
 * Takes spin configuration, returns gradient for each spin
 */
using HamiltonianGradientFunc = std::function<std::vector<Spin3D>(const std::vector<Spin3D>&)>;

/**
 * TWA Spin Simulator - CPU version
 */
class TWASpinSimulator {
public:
    TWASpinSimulator(int n_qubits, int n_trajectories);
    ~TWASpinSimulator() = default;

    // Dissipation management
    void add_dissipation(ChannelType type, double rate, const std::vector<int>& qubits);
    void clear_dissipation();

    // Initial state sampling
    std::vector<Spin3D> discrete_sample_initial_state(const std::string& state = "ground");

    // Noise generation
    NoiseData generate_noise(double dt);

    // Equations of motion
    std::vector<Spin3D> equations_of_motion(
        double t,
        const std::vector<Spin3D>& spins,
        const HamiltonianGradientFunc& grad_func,
        const NoiseData& noise
    );

    // RK4 integration step
    std::vector<Spin3D> rk4_step(
        double t,
        const std::vector<Spin3D>& spins,
        double dt,
        const HamiltonianGradientFunc& grad_func,
        const NoiseData& noise
    );

    // Spin conservation check
    void check_spin_conservation(std::vector<Spin3D>& spins, bool renormalize = false);

    // Getters
    int get_n_qubits() const { return n_qubits_; }
    int get_n_trajectories() const { return n_trajectories_; }
    const std::vector<DissipationChannel>& get_dissipation_channels() const {
        return dissipation_channels_;
    }

private:
    int n_qubits_;
    int n_trajectories_;
    std::vector<DissipationChannel> dissipation_channels_;
    std::mt19937 rng_;
    std::normal_distribution<double> normal_dist_;
    std::uniform_int_distribution<int> sign_dist_;
};

/**
 * Hardware dissipation rates for 171Yb+ trapped ions
 */
class IonTrapDissipationRates {
public:
    // SI units
    double T1_SI;              // Energy relaxation time (seconds)
    double T2_SI;              // Dephasing time (seconds)
    double gamma_decay_SI;     // Decay rate (Hz)
    double kappa_dephasing_SI; // Dephasing rate (Hz)

    // Atomic units
    static constexpr double AU_TIME = 2.4189e-17;   // seconds per a.u.
    static constexpr double AU_ENERGY = 27.211;     // eV

    double gamma_decay_au;
    double kappa_dephasing_au;

    // Scaled rates (matched to Hamiltonian units)
    double energy_scale;
    double gamma_decay;
    double kappa_dephasing;

    // Gate error rates
    double single_qubit_error;
    double two_qubit_error;

    IonTrapDissipationRates(double energy_scale = 1.0);

    void print_info() const;
};

/**
 * Simulation results container
 */
struct SimulationResults {
    std::vector<double> times;
    std::vector<double> avg_energies;
    std::vector<double> std_energies;
    std::vector<double> magnetization;

    // Full trajectory data (optional, memory intensive)
    std::vector<std::vector<std::vector<Spin3D>>> all_spins;  // [traj][step][qubit]
    std::vector<std::vector<double>> all_energies;            // [traj][step]
};

} // namespace twa

#endif // TWA_FRAMEWORK_HPP
