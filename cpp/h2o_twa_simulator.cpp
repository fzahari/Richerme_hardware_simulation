/**
 * H2O Molecule TWA Simulator - C++ Implementation
 */

#include "h2o_twa_simulator.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <chrono>
#include <algorithm>

namespace twa {

H2OTWASimulator::H2OTWASimulator(int n_trajectories, double energy_scale)
    : n_trajectories_(n_trajectories),
      hardware_(energy_scale) {

    twa_ = std::make_unique<TWASpinSimulator>(N_QUBITS, n_trajectories);

    generate_h2o_hamiltonian();

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "H2O TWA SIMULATOR (C++)" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "  Qubits: " << N_QUBITS << std::endl;
    std::cout << "  Hamiltonian terms: " << hamiltonian_terms_.size() << std::endl;
    std::cout << "  Trajectories: " << n_trajectories_ << std::endl;
    std::cout << "  T1 = " << hardware_.T1_SI << " s (SI)" << std::endl;
    std::cout << "  T2 = " << hardware_.T2_SI << " s (SI)" << std::endl;
    std::cout << "======================================================================\n" << std::endl;

    hardware_.print_info();
}

void H2OTWASimulator::generate_h2o_hamiltonian() {
    hamiltonian_terms_.clear();

    // Diagonal Z terms (one-body energies)
    for (int i = 0; i < N_QUBITS; ++i) {
        double coeff = -10.0 + 2.0 * i;
        hamiltonian_terms_.emplace_back('Z', std::vector<int>{i}, coeff);
    }

    // Two-qubit ZZ terms (Coulomb repulsion)
    for (int i = 0; i < N_QUBITS; ++i) {
        for (int j = i + 1; j < N_QUBITS; ++j) {
            double coeff = 0.5 * std::exp(-0.3 * std::abs(i - j));
            hamiltonian_terms_.emplace_back('Z', std::vector<int>{i, j}, coeff);
        }
    }

    // Hopping terms (XX and YY)
    for (int i = 0; i < N_QUBITS - 1; ++i) {
        double coeff = -1.5 * std::exp(-0.1 * i);
        hamiltonian_terms_.emplace_back('X', std::vector<int>{i, i + 1}, coeff);
        hamiltonian_terms_.emplace_back('Y', std::vector<int>{i, i + 1}, coeff);
    }
}

double H2OTWASimulator::build_h2o_classical_hamiltonian(
    const std::vector<Spin3D>& s) {

    double H = 0.0;

    for (const auto& [type, qubits, coeff] : hamiltonian_terms_) {
        if (type == 'Z') {
            if (qubits.size() == 1) {
                // Single Z term
                H += coeff * s[qubits[0]].z;
            } else if (qubits.size() == 2) {
                // ZZ term
                H += coeff * s[qubits[0]].z * s[qubits[1]].z;
            }
        } else if (type == 'X') {
            // XX term
            H += coeff * s[qubits[0]].x * s[qubits[1]].x;
        } else if (type == 'Y') {
            // YY term
            H += coeff * s[qubits[0]].y * s[qubits[1]].y;
        }
    }

    return H;
}

std::vector<Spin3D> H2OTWASimulator::hamiltonian_gradient(
    const std::vector<Spin3D>& s) {

    std::vector<Spin3D> grad(N_QUBITS);

    for (const auto& [type, qubits, coeff] : hamiltonian_terms_) {
        if (type == 'Z') {
            if (qubits.size() == 1) {
                // ∂(c·s^z_i)/∂s^z_i = c
                grad[qubits[0]].z += coeff;
            } else if (qubits.size() == 2) {
                // ∂(c·s^z_i·s^z_j)/∂s^z_i = c·s^z_j
                grad[qubits[0]].z += coeff * s[qubits[1]].z;
                grad[qubits[1]].z += coeff * s[qubits[0]].z;
            }
        } else if (type == 'X') {
            // ∂(c·s^x_i·s^x_j)/∂s^x_i = c·s^x_j
            grad[qubits[0]].x += coeff * s[qubits[1]].x;
            grad[qubits[1]].x += coeff * s[qubits[0]].x;
        } else if (type == 'Y') {
            // ∂(c·s^y_i·s^y_j)/∂s^y_i = c·s^y_j
            grad[qubits[0]].y += coeff * s[qubits[1]].y;
            grad[qubits[1]].y += coeff * s[qubits[0]].y;
        }
    }

    return grad;
}

HamiltonianGradientFunc H2OTWASimulator::make_gradient_func() {
    return [this](const std::vector<Spin3D>& spins) {
        return this->hamiltonian_gradient(spins);
    };
}

SimulationResults H2OTWASimulator::simulate_twa_dynamics(
    double total_time,
    int n_steps,
    bool add_T1,
    bool add_T2,
    bool renormalize_spins) {

    auto start_time = std::chrono::high_resolution_clock::now();

    double dt = total_time / n_steps;

    std::cout << "\nRunning H2O TWA simulation:" << std::endl;
    std::cout << "  Qubits: " << N_QUBITS << std::endl;
    std::cout << "  Time steps: " << n_steps << ", dt = " << dt << std::endl;
    std::cout << "  Trajectories: " << n_trajectories_ << std::endl;
    std::cout << "  T1 decay: " << (add_T1 ? "ON" : "OFF") << std::endl;
    std::cout << "  T2 dephasing: " << (add_T2 ? "ON" : "OFF") << std::endl;
    std::cout << "  Spin renormalization: " << (renormalize_spins ? "ON" : "OFF") << std::endl;

    // Set up dissipation channels
    twa_->clear_dissipation();
    std::vector<int> all_qubits(N_QUBITS);
    std::iota(all_qubits.begin(), all_qubits.end(), 0);

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

    auto grad_func = make_gradient_func();

    int failed_trajectories = 0;

    // Run trajectories
    for (int traj = 0; traj < n_trajectories_; ++traj) {
        // Initialize state (Hartree-Fock: first 5 qubits excited)
        auto s = twa_->discrete_sample_initial_state("ground");
        for (int i = 0; i < 5; ++i) {
            s[i].z = 1.0;  // |1⟩
        }
        for (int i = 5; i < N_QUBITS; ++i) {
            s[i].z = -1.0;  // |0⟩
        }

        bool trajectory_failed = false;

        // Time evolution
        for (int step = 0; step < n_steps; ++step) {
            double t = step * dt;
            results.times[step] = t;

            // Store state
            results.all_spins[traj][step] = s;
            double E = build_h2o_classical_hamiltonian(s);
            results.all_energies[traj][step] = E;

            // Check for numerical instability
            if (step > 0 && (std::isnan(E) || std::isinf(E) || std::abs(E) > 1e6)) {
                std::cout << "  WARNING: Trajectory " << traj
                         << " became unstable at step " << step << std::endl;
                trajectory_failed = true;
                failed_trajectories++;
                // Fill remaining with NaN
                for (int s = step; s < n_steps; ++s) {
                    results.all_energies[traj][s] = std::nan("");
                }
                break;
            }

            // Generate noise
            auto noise = twa_->generate_noise(dt);

            // RK4 step
            s = twa_->rk4_step(t, s, dt, grad_func, noise);

            // Renormalize spins periodically
            if (renormalize_spins && (step + 1) % 10 == 0) {
                twa_->check_spin_conservation(s, true);
            }
        }

        if ((traj + 1) % 50 == 0) {
            std::cout << "  Completed trajectory " << (traj + 1)
                     << "/" << n_trajectories_ << std::endl;
        }
    }

    // Compute averages (ignoring NaN values)
    results.avg_energies.resize(n_steps);
    results.std_energies.resize(n_steps);
    results.magnetization.resize(n_steps);

    for (int step = 0; step < n_steps; ++step) {
        // Energy statistics (excluding NaN)
        std::vector<double> valid_energies;
        for (int traj = 0; traj < n_trajectories_; ++traj) {
            double E = results.all_energies[traj][step];
            if (!std::isnan(E)) {
                valid_energies.push_back(E);
            }
        }

        if (!valid_energies.empty()) {
            double mean = std::accumulate(valid_energies.begin(), valid_energies.end(), 0.0)
                         / valid_energies.size();
            double sq_sum = std::inner_product(valid_energies.begin(), valid_energies.end(),
                                              valid_energies.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / valid_energies.size() - mean * mean);

            results.avg_energies[step] = mean;
            results.std_energies[step] = stdev;
        } else {
            results.avg_energies[step] = std::nan("");
            results.std_energies[step] = std::nan("");
        }

        // Magnetization (excluding NaN)
        double mag = 0.0;
        int valid_count = 0;
        for (int traj = 0; traj < n_trajectories_; ++traj) {
            if (!std::isnan(results.all_energies[traj][step])) {
                for (int q = 0; q < N_QUBITS; ++q) {
                    mag += results.all_spins[traj][step][q].z;
                }
                valid_count++;
            }
        }
        results.magnetization[step] = valid_count > 0 ? mag / valid_count : std::nan("");
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    int successful = n_trajectories_ - failed_trajectories;
    if (failed_trajectories > 0) {
        std::cout << "  WARNING: " << failed_trajectories << " trajectories failed" << std::endl;
    }
    std::cout << "  Successful trajectories: " << successful
             << "/" << n_trajectories_ << std::endl;

    std::cout << "\n✓ Simulation complete in " << duration.count() / 1000.0
             << " seconds" << std::endl;
    std::cout << "  Final energy: " << results.avg_energies[n_steps-1]
             << " ± " << results.std_energies[n_steps-1] << " H" << std::endl;

    return results;
}

H2OTWASimulator::ComparisonResults H2OTWASimulator::compare_dissipation_effects(
    double total_time,
    int n_steps) {

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "H2O: COMPARING DISSIPATION EFFECTS" << std::endl;
    std::cout << "======================================================================" << std::endl;

    ComparisonResults comp;

    std::cout << "\n[1] Running ideal dynamics (no T1/T2)..." << std::endl;
    comp.ideal = simulate_twa_dynamics(total_time, n_steps, false, false);

    std::cout << "\n[2] Running with T1 decay + T2 dephasing..." << std::endl;
    comp.full = simulate_twa_dynamics(total_time, n_steps, true, true);

    std::cout << "\n======================================================================" << std::endl;
    std::cout << "SIMULATION COMPLETE" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << "\nFinal energies:" << std::endl;
    std::cout << "  Ideal:   " << comp.ideal.avg_energies.back()
             << " ± " << comp.ideal.std_energies.back() << " H" << std::endl;
    std::cout << "  T1 + T2: " << comp.full.avg_energies.back()
             << " ± " << comp.full.std_energies.back() << " H" << std::endl;

    std::cout << "\nFinal magnetization:" << std::endl;
    std::cout << "  Ideal:   " << comp.ideal.magnetization.back() << std::endl;
    std::cout << "  T1 + T2: " << comp.full.magnetization.back() << std::endl;

    return comp;
}

} // namespace twa
