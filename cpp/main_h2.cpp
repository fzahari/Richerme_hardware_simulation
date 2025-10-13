/**
 * H2 Molecule TWA Simulation - Main Program
 *
 * Run with: ./h2_twa_simulation
 */

#include "h2_twa_simulator.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "======================================================================" << std::endl;
    std::cout << "H2 MOLECULE SIMULATION WITH TWA DISSIPATION (C++)" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "This simulation uses the Truncated Wigner Approximation (TWA)" << std::endl;
    std::cout << "to model dissipative quantum dynamics of the H2 molecule." << std::endl;
    std::cout << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  - 4-qubit quantum chemistry Hamiltonian" << std::endl;
    std::cout << "  - Hardware-realistic T1/T2 decoherence (171Yb+)" << std::endl;
    std::cout << "  - Stochastic trajectory averaging" << std::endl;
    std::cout << "  - Comparison with ideal (no dissipation) case" << std::endl;
    std::cout << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;

    try {
        // Parse command line arguments
        int n_trajectories = 500;
        if (argc > 1) {
            n_trajectories = std::atoi(argv[1]);
        }

        // Create simulator
        twa::H2TWASimulator h2_twa(n_trajectories);

        // Run comparison
        auto results = h2_twa.compare_with_ideal(
            0.74,   // Bond distance (Angstroms)
            20.0,   // Total time (a.u.)
            100     // Number of steps
        );

        std::cout << "\n✓ Simulation complete!" << std::endl;
        std::cout << "\nNote: C++ version outputs to console only." << std::endl;
        std::cout << "For visualization, use Python version with matplotlib." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
