/**
 * H2 Molecule TWA Simulator - C++ Implementation
 */

#include "h2_twa_simulator.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <chrono>

namespace twa {

H2TWASimulator::H2TWASimulator(int n_trajectories, double energy_scale)
    : n_trajectories_(n_trajectories),
      hardware_(energy_scale) {

    twa_ = std::make_unique<TWASpinSimulator>(N_QUBITS, n_trajectories);

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "H2 TWA SIMULATOR (C++)" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Qubits: " << N_QUBITS << std::endl;
    std::cout << "  Trajectories: " << n_trajectories_ << std::endl;
    std::cout << "  T1 = " << hardware_.T1_SI << " s" << std::endl;
    std::cout << "  T2 = " << hardware_.T2_SI << " s" << std::endl;
    std::cout << "======================================================================\n" << std::endl;

    hardware_.print_info();
}

double H2TWASimulator::build_h2_classical_hamiltonian(
    double r,
    const std::vector<Spin3D>& s) {

    // Empirical parameters for H2 at distance r
    double e_nuc = 1.0 / r;
    double t = 0.52917 * std::exp(-1.5 * (r - 0.741));
    double mu = -1.1256 + 0.2 * r;
    double u = 0.6744 / (1.0 + 0.1 * r);
    double v = 0.1815 * std::exp(-0.5 * r);

    double H = e_nuc;

    // Single-qubit Z terms
    H += (mu / 2.0) * s[0].z;
    H += (mu / 2.0) * s[1].z;
    H += (mu / 4.0) * s[2].z;
    H += (mu / 4.0) * s[3].z;

    // Hopping terms: X_i X_j and Y_i Y_j
    H += (-t / 2.0) * s[0].x * s[2].x;  // X0 X2
    H += (-t / 2.0) * s[0].y * s[2].y;  // Y0 Y2
    H += (-t / 2.0) * s[1].x * s[3].x;  // X1 X3
    H += (-t / 2.0) * s[1].y * s[3].y;  // Y1 Y3

    // Two-qubit ZZ interactions
    H += (u / 4.0) * s[0].z * s[1].z;   // Z0 Z1
    H += (u / 4.0) * s[2].z * s[3].z;   // Z2 Z3
    H += (v / 4.0) * s[0].z * s[2].z;   // Z0 Z2
    H += (v / 4.0) * s[1].z * s[3].z;   // Z1 Z3
    H += (v / 8.0) * s[0].z * s[3].z;   // Z0 Z3
    H += (v / 8.0) * s[1].z * s[2].z;   // Z1 Z2

    return H;
}

std::vector<Spin3D> H2TWASimulator::hamiltonian_gradient(
    double r,
    const std::vector<Spin3D>& s) {

    // Parameters
    double t = 0.52917 * std::exp(-1.5 * (r - 0.741));
    double mu = -1.1256 + 0.2 * r;
    double u = 0.6744 / (1.0 + 0.1 * r);
    double v = 0.1815 * std::exp(-0.5 * r);

    std::vector<Spin3D> grad(N_QUBITS);

    // ∂H/∂s^x
    grad[0].x = (-t / 2.0) * s[2].x;
    grad[1].x = (-t / 2.0) * s[3].x;
    grad[2].x = (-t / 2.0) * s[0].x;
    grad[3].x = (-t / 2.0) * s[1].x;

    // ∂H/∂s^y
    grad[0].y = (-t / 2.0) * s[2].y;
    grad[1].y = (-t / 2.0) * s[3].y;
    grad[2].y = (-t / 2.0) * s[0].y;
    grad[3].y = (-t / 2.0) * s[1].y;

    // ∂H/∂s^z (all Z terms)
    grad[0].z = (mu / 2.0) + (u / 4.0) * s[1].z + (v / 4.0) * s[2].z + (v / 8.0) * s[3].z;
    grad[1].z = (mu / 2.0) + (u / 4.0) * s[0].z + (v / 4.0) * s[3].z + (v / 8.0) * s[2].z;
    grad[2].z = (mu / 4.0) + (u / 4.0) * s[3].z + (v / 4.0) * s[0].z + (v / 8.0) * s[1].z;
    grad[3].z = (mu / 4.0) + (u / 4.0) * s[2].z + (v / 4.0) * s[1].z + (v / 8.0) * s[0].z;

    return grad;
}

HamiltonianGradientFunc H2TWASimulator::make_gradient_func(double r) {
    return [this, r](const std::vector<Spin3D>& spins) {
        return this->hamiltonian_gradient(r, spins);
    };
}

SimulationResults H2TWASimulator::simulate_twa_dynamics(
    double r,
    double total_time,
    int n_steps,
    bool add_T1,
    bool add_T2) {

    auto start_time = std::chrono::high_resolution_clock::now();

    double dt = total_time / n_steps;

    std::cout << "\nRunning TWA simulation:" << std::endl;
    std::cout << "  Bond distance: R = " << r << " Å" << std::endl;
    std::cout << "  Time steps: " << n_steps << ", dt = " << dt << std::endl;
    std::cout << "  Trajectories: " << n_trajectories_ << std::endl;
    std::cout << "  T1 decay: " << (add_T1 ? "ON" : "OFF") << std::endl;
    std::cout << "  T2 dephasing: " << (add_T2 ? "ON" : "OFF") << std::endl;

    // Set up dissipation channels
    twa_->clear_dissipation();
    std::vector<int> all_qubits = {0, 1, 2, 3};

    if (add_T1) {
        twa_->add_dissipation(ChannelType::DECAY, hardware_.gamma_decay, all_qubits);
    }
    if (add_T2) {
        twa_->add_dissipation(ChannelType::DEPHASING, hardware_.kappa_dephasing, all_qubits);
    }

    // Storage
    SimulationResults results;
    results.times.resize(n_steps);
    results.all_energies.resize(n_trajectories_, std::vector<double>(n_steps));
    results.all_spins.resize(n_trajectories_,
        std::vector<std::vector<Spin3D>>(n_steps, std::vector<Spin3D>(N_QUBITS)));

    auto grad_func = make_gradient_func(r);

    // Run trajectories
    for (int traj = 0; traj < n_trajectories_; ++traj) {
        // Initialize state (Hartree-Fock: |0011⟩)
        auto s = twa_->discrete_sample_initial_state("ground");
        // Override sz for |0011⟩ state
        s[0].z = -1.0;  // |0⟩
        s[1].z = -1.0;  // |0⟩
        s[2].z = 1.0;   // |1⟩
        s[3].z = 1.0;   // |1⟩

        // Time evolution
        for (int step = 0; step < n_steps; ++step) {
            double t = step * dt;
            results.times[step] = t;

            // Store state
            results.all_spins[traj][step] = s;
            results.all_energies[traj][step] = build_h2_classical_hamiltonian(r, s);

            // Generate noise
            auto noise = twa_->generate_noise(dt);

            // RK4 step
            s = twa_->rk4_step(t, s, dt, grad_func, noise);
        }

        if ((traj + 1) % 100 == 0) {
            std::cout << "  Completed trajectory " << (traj + 1)
                     << "/" << n_trajectories_ << std::endl;
        }
    }

    // Compute averages
    results.avg_energies.resize(n_steps);
    results.std_energies.resize(n_steps);
    results.magnetization.resize(n_steps);

    for (int step = 0; step < n_steps; ++step) {
        // Energy statistics
        std::vector<double> energies(n_trajectories_);
        for (int traj = 0; traj < n_trajectories_; ++traj) {
            energies[traj] = results.all_energies[traj][step];
        }

        double mean = std::accumulate(energies.begin(), energies.end(), 0.0) / n_trajectories_;
        double sq_sum = std::inner_product(energies.begin(), energies.end(),
                                          energies.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / n_trajectories_ - mean * mean);

        results.avg_energies[step] = mean;
        results.std_energies[step] = stdev;

        // Magnetization
        double mag = 0.0;
        for (int traj = 0; traj < n_trajectories_; ++traj) {
            for (int q = 0; q < N_QUBITS; ++q) {
                mag += results.all_spins[traj][step][q].z;
            }
        }
        results.magnetization[step] = mag / n_trajectories_;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\n✓ Simulation complete in " << duration.count() / 1000.0
             << " seconds" << std::endl;
    std::cout << "  Final energy: " << results.avg_energies[n_steps-1]
             << " ± " << results.std_energies[n_steps-1] << " H" << std::endl;

    return results;
}

H2TWASimulator::ComparisonResults H2TWASimulator::compare_with_ideal(
    double r,
    double total_time,
    int n_steps) {

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "COMPARING IDEAL VS. DISSIPATIVE DYNAMICS" << std::endl;
    std::cout << "======================================================================" << std::endl;

    ComparisonResults comp;

    std::cout << "\n[1] Running ideal dynamics (no T1/T2)..." << std::endl;
    comp.ideal = simulate_twa_dynamics(r, total_time, n_steps, false, false);

    std::cout << "\n[2] Running with T2 dephasing only..." << std::endl;
    comp.T2_only = simulate_twa_dynamics(r, total_time, n_steps, false, true);

    std::cout << "\n[3] Running with T1 decay + T2 dephasing..." << std::endl;
    comp.full = simulate_twa_dynamics(r, total_time, n_steps, true, true);

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "SIMULATION COMPLETE" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "\nFinal energies:" << std::endl;
    std::cout << "  Ideal:   " << comp.ideal.avg_energies.back()
             << " ± " << comp.ideal.std_energies.back() << " H" << std::endl;
    std::cout << "  T2 only: " << comp.T2_only.avg_energies.back()
             << " ± " << comp.T2_only.std_energies.back() << " H" << std::endl;
    std::cout << "  T1 + T2: " << comp.full.avg_energies.back()
             << " ± " << comp.full.std_energies.back() << " H" << std::endl;

    return comp;
}

} // namespace twa
