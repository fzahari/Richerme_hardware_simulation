/**
 * H2O Molecule TWA Simulation - Main Program
 *
 * Run with: ./h2o_twa_simulation [n_trajectories]
 */

#include "h2o_twa_simulator.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "======================================================================" << std::endl;
    std::cout << "H2O MOLECULE SIMULATION WITH TWA DISSIPATION (C++)" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;
    std::cout << "This simulation uses the Truncated Wigner Approximation (TWA)" << std::endl;
    std::cout << "to model dissipative quantum dynamics of the H2O molecule." << std::endl;
    std::cout << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  - 10-qubit quantum chemistry Hamiltonian (QEE compressed)" << std::endl;
    std::cout << "  - Hardware-realistic T1/T2 decoherence (171Yb+)" << std::endl;
    std::cout << "  - Stochastic trajectory averaging" << std::endl;
    std::cout << "  - Comparison with ideal (no dissipation) case" << std::endl;
    std::cout << std::endl;
    std::cout << "Note: This is computationally more intensive than H2 (10 vs 4 qubits)" << std::endl;
    std::cout << "======================================================================" << std::endl;
    std::cout << std::endl;

    try {
        // Parse command line arguments
        int n_trajectories = 300;
        if (argc > 1) {
            n_trajectories = std::atoi(argv[1]);
        }

        // Create simulator with properly scaled dissipation
        // energy_scale=1e15 gives reasonable dissipation for this model
        twa::H2OTWASimulator h2o_twa(n_trajectories, 1e15);

        // Run comparison with shorter time and more steps for stability
        auto results = h2o_twa.compare_dissipation_effects(
            5.0,    // Total time (a.u.)
            200     // Number of steps
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
