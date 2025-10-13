/**
 * H2 Molecule TWA Simulator - C++ Header
 *
 * Simulates H2 molecule with dissipative dynamics using TWA.
 * 4-qubit quantum chemistry Hamiltonian with T1/T2 decoherence.
 */

#ifndef H2_TWA_SIMULATOR_HPP
#define H2_TWA_SIMULATOR_HPP

#include "twa_framework.hpp"
#include <memory>

namespace twa {

class H2TWASimulator {
public:
    H2TWASimulator(int n_trajectories = 500, double energy_scale = 1.0);
    ~H2TWASimulator() = default;

    // Hamiltonian evaluation
    double build_h2_classical_hamiltonian(double r, const std::vector<Spin3D>& spins);

    // Hamiltonian gradient
    std::vector<Spin3D> hamiltonian_gradient(double r, const std::vector<Spin3D>& spins);

    // Simulation
    SimulationResults simulate_twa_dynamics(
        double r,                // Bond distance (Angstroms)
        double total_time,       // Total simulation time (a.u.)
        int n_steps = 100,       // Number of time steps
        bool add_T1 = true,      // Include T1 decay
        bool add_T2 = true       // Include T2 dephasing
    );

    // Comparison simulation
    struct ComparisonResults {
        SimulationResults ideal;
        SimulationResults T2_only;
        SimulationResults full;
    };

    ComparisonResults compare_with_ideal(
        double r = 0.74,
        double total_time = 20.0,
        int n_steps = 100
    );

    // Getters
    TWASpinSimulator& get_twa() { return *twa_; }
    const IonTrapDissipationRates& get_hardware() const { return hardware_; }

private:
    static constexpr int N_QUBITS = 4;
    int n_trajectories_;

    std::unique_ptr<TWASpinSimulator> twa_;
    IonTrapDissipationRates hardware_;

    // Helper for creating gradient function
    HamiltonianGradientFunc make_gradient_func(double r);
};

} // namespace twa

#endif // H2_TWA_SIMULATOR_HPP
