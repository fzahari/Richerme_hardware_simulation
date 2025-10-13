/**
 * H2O Molecule TWA Simulator - C++ Header
 *
 * Simulates H2O molecule with dissipative dynamics using TWA.
 * 10-qubit quantum chemistry Hamiltonian with T1/T2 decoherence.
 */

#ifndef H2O_TWA_SIMULATOR_HPP
#define H2O_TWA_SIMULATOR_HPP

#include "twa_framework.hpp"
#include <memory>
#include <tuple>

namespace twa {

class H2OTWASimulator {
public:
    H2OTWASimulator(int n_trajectories = 300, double energy_scale = 1e15);
    ~H2OTWASimulator() = default;

    // Hamiltonian evaluation
    double build_h2o_classical_hamiltonian(const std::vector<Spin3D>& spins);

    // Hamiltonian gradient
    std::vector<Spin3D> hamiltonian_gradient(const std::vector<Spin3D>& spins);

    // Simulation
    SimulationResults simulate_twa_dynamics(
        double total_time,       // Total simulation time (a.u.)
        int n_steps = 200,       // Number of time steps
        bool add_T1 = true,      // Include T1 decay
        bool add_T2 = true,      // Include T2 dephasing
        bool renormalize_spins = true  // Renormalize periodically
    );

    // Comparison simulation
    struct ComparisonResults {
        SimulationResults ideal;
        SimulationResults full;
    };

    ComparisonResults compare_dissipation_effects(
        double total_time = 5.0,
        int n_steps = 200
    );

    // Getters
    TWASpinSimulator& get_twa() { return *twa_; }
    const IonTrapDissipationRates& get_hardware() const { return hardware_; }

private:
    static constexpr int N_QUBITS = 10;
    int n_trajectories_;

    std::unique_ptr<TWASpinSimulator> twa_;
    IonTrapDissipationRates hardware_;

    // Hamiltonian terms: (type, qubits, coefficient)
    // type: 'Z', 'ZZ', 'XX', 'YY'
    using HamiltonianTerm = std::tuple<char, std::vector<int>, double>;
    std::vector<HamiltonianTerm> hamiltonian_terms_;

    void generate_h2o_hamiltonian();
    HamiltonianGradientFunc make_gradient_func();
};

} // namespace twa

#endif // H2O_TWA_SIMULATOR_HPP
