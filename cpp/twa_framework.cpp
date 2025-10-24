/**
 * Truncated Wigner Approximation (TWA) Framework - C++ Implementation
 */

#include "twa_framework.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace twa {

// Spin3D methods
void Spin3D::normalize(double target) {
    double norm_sq = norm_squared();
    if (norm_sq > 1e-10) {
        double scale = std::sqrt(target / norm_sq);
        x *= scale;
        y *= scale;
        z *= scale;
    }
}

// TWASpinSimulator implementation
TWASpinSimulator::TWASpinSimulator(int n_qubits, int n_trajectories)
    : n_qubits_(n_qubits),
      n_trajectories_(n_trajectories),
      rng_(std::random_device{}()),
      normal_dist_(0.0, 1.0),
      sign_dist_(0, 1) {

    std::cout << "TWA Spin Simulator initialized:" << std::endl;
    std::cout << "  Qubits: " << n_qubits_ << std::endl;
    std::cout << "  Trajectories: " << n_trajectories_ << std::endl;
}

void TWASpinSimulator::add_dissipation(ChannelType type, double rate,
                                       const std::vector<int>& qubits) {
    dissipation_channels_.emplace_back(type, rate, qubits);

    std::string type_str;
    switch(type) {
        case ChannelType::DECAY: type_str = "decay"; break;
        case ChannelType::PUMPING: type_str = "pumping"; break;
        case ChannelType::DEPHASING: type_str = "dephasing"; break;
    }

    std::cout << "Added " << type_str << " channel: rate=" << rate
              << ", qubits=[";
    for (size_t i = 0; i < qubits.size(); ++i) {
        std::cout << qubits[i];
        if (i < qubits.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void TWASpinSimulator::clear_dissipation() {
    dissipation_channels_.clear();
}

std::vector<Spin3D> TWASpinSimulator::discrete_sample_initial_state(
    const std::string& state) {

    std::vector<Spin3D> spins(n_qubits_);

    for (int k = 0; k < n_qubits_; ++k) {
        double sx = (sign_dist_(rng_) == 0) ? -1.0 : 1.0;
        double sy = (sign_dist_(rng_) == 0) ? -1.0 : 1.0;
        double sz;

        if (state == "ground") {
            sz = -1.0;
        } else if (state == "excited") {
            sz = 1.0;
        } else if (state == "superposition") {
            sz = (sign_dist_(rng_) == 0) ? -1.0 : 1.0;
        } else {
            std::cerr << "Unknown initial state: " << state << std::endl;
            sz = -1.0;
        }

        spins[k] = Spin3D(sx, sy, sz);
    }

    return spins;
}

NoiseData TWASpinSimulator::generate_noise(double dt) {
    NoiseData noise(n_qubits_);

    for (const auto& channel : dissipation_channels_) {
        double rate = channel.rate;

        // Safety check
        if (rate * dt > 1.0) {
            std::cout << "WARNING: rate*dt = " << rate*dt
                     << " > 1, noise may be too large!" << std::endl;
        }

        double sigma = std::sqrt(std::max(rate / dt, 1e-20));

        if (channel.type == ChannelType::DECAY) {
            for (int k = 0; k < n_qubits_; ++k) {
                noise.decay_x[k] = normal_dist_(rng_) * sigma;
                noise.decay_y[k] = normal_dist_(rng_) * sigma;
            }
        } else if (channel.type == ChannelType::DEPHASING) {
            for (int k = 0; k < n_qubits_; ++k) {
                noise.dephasing[k] = normal_dist_(rng_) * sigma;
            }
        }
    }

    return noise;
}

std::vector<Spin3D> TWASpinSimulator::equations_of_motion(
    [[maybe_unused]] double t,
    const std::vector<Spin3D>& spins,
    const HamiltonianGradientFunc& grad_func,
    const NoiseData& noise) {

    std::vector<Spin3D> dsdt(n_qubits_);

    // Coherent part: {s, H}_p = 2(s × ∇H)
    auto grad_H = grad_func(spins);

    for (int k = 0; k < n_qubits_; ++k) {
        dsdt[k] = spins[k].cross(grad_H[k]) * 2.0;
    }

    // Dissipative part
    for (const auto& channel : dissipation_channels_) {
        if (channel.type == ChannelType::DECAY) {
            double gamma = channel.rate;

            for (int k : channel.qubits) {
                if (k < n_qubits_) {
                    double xi_x = noise.decay_x[k];
                    double xi_y = noise.decay_y[k];
                    double sz = spins[k].z;
                    double sx = spins[k].x;
                    double sy = spins[k].y;

                    // From TWA paper Table I (spin loss channel)
                    dsdt[k].x += (gamma / 2.0) * sx * sz + xi_x * sz;
                    dsdt[k].y += (gamma / 2.0) * sy * sz + xi_y * sz;
                    dsdt[k].z += -(gamma / 2.0) * (sx*sx + sy*sy) - (xi_x * sx + xi_y * sy);
                }
            }
        } else if (channel.type == ChannelType::DEPHASING) {
            for (int k : channel.qubits) {
                if (k < n_qubits_) {
                    double eta = noise.dephasing[k];

                    // From TWA paper Table I (dephasing channel)
                    dsdt[k].x += 2.0 * eta * spins[k].y;
                    dsdt[k].y += -2.0 * eta * spins[k].x;
                    // dsdt[k].z unchanged by pure dephasing
                }
            }
        }
    }

    return dsdt;
}

std::vector<Spin3D> TWASpinSimulator::rk4_step(
    double t,
    const std::vector<Spin3D>& spins,
    double dt,
    const HamiltonianGradientFunc& grad_func,
    const NoiseData& noise) {

    // RK4 for stochastic differential equations (Stratonovich)
    auto k1 = equations_of_motion(t, spins, grad_func, noise);

    std::vector<Spin3D> s_temp(n_qubits_);
    for (int k = 0; k < n_qubits_; ++k) {
        s_temp[k] = spins[k] + k1[k] * (dt / 2.0);
    }
    auto k2 = equations_of_motion(t + dt/2.0, s_temp, grad_func, noise);

    for (int k = 0; k < n_qubits_; ++k) {
        s_temp[k] = spins[k] + k2[k] * (dt / 2.0);
    }
    auto k3 = equations_of_motion(t + dt/2.0, s_temp, grad_func, noise);

    for (int k = 0; k < n_qubits_; ++k) {
        s_temp[k] = spins[k] + k3[k] * dt;
    }
    auto k4 = equations_of_motion(t + dt, s_temp, grad_func, noise);

    std::vector<Spin3D> s_new(n_qubits_);
    for (int k = 0; k < n_qubits_; ++k) {
        s_new[k] = spins[k] + (k1[k] + k2[k] * 2.0 + k3[k] * 2.0 + k4[k]) * (dt / 6.0);
    }

    return s_new;
}

void TWASpinSimulator::check_spin_conservation(std::vector<Spin3D>& spins,
                                              bool renormalize) {
    if (!renormalize) return;

    constexpr double target = 3.0;  // |s|² for spin-1/2

    for (auto& spin : spins) {
        double norm_sq = spin.norm_squared();
        if (norm_sq > 10.0 || norm_sq < 0.1) {
            spin.normalize(target);
        }
    }
}

// IonTrapDissipationRates implementation
IonTrapDissipationRates::IonTrapDissipationRates(double energy_scale_)
    : T1_SI(1000.0),
      T2_SI(1.0),
      energy_scale(energy_scale_),
      single_qubit_error(1.0 - 0.998),
      two_qubit_error(1.0 - 0.970) {

    // Convert to rates in Hz
    gamma_decay_SI = 1.0 / T1_SI;
    kappa_dephasing_SI = 1.0 / T2_SI;

    // Convert to atomic units
    gamma_decay_au = gamma_decay_SI * AU_TIME;
    kappa_dephasing_au = kappa_dephasing_SI * AU_TIME;

    // Rescale to match Hamiltonian energy units
    gamma_decay = gamma_decay_au * energy_scale;
    kappa_dephasing = kappa_dephasing_au * energy_scale;
}

void IonTrapDissipationRates::print_info() const {
    std::cout << "\nDissipation rates initialized:" << std::endl;
    std::cout << "  T1 = " << T1_SI << " s → γ = " << gamma_decay << " (scaled)" << std::endl;
    std::cout << "  T2 = " << T2_SI << " s → κ = " << kappa_dephasing << " (scaled)" << std::endl;
    std::cout << "  Energy scale: " << energy_scale << std::endl;
}

} // namespace twa
