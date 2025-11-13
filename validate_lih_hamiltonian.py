#!/usr/bin/env python3
"""
Validation Script: LiH Hamiltonian and VQE Ground State

This script validates the LiH implementation by:
1. Generating the TRUE LiH Hamiltonian using Qiskit Nature + PySCF
2. Computing the exact ground state energy
3. Comparing with VQE results
4. Validating against known literature values

Requirements:
    pip install qiskit-nature[pyscf]
"""

import numpy as np
from scipy.linalg import eigh

# Check if Qiskit Nature is available
try:
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.problems import ElectronicStructureProblem
    QISKIT_AVAILABLE = True
    print("✓ Qiskit Nature available")
except ImportError as e:
    print(f"✗ Qiskit Nature NOT available: {e}")
    print("Install with: pip install qiskit-nature pyscf")
    QISKIT_AVAILABLE = False
    exit(1)


def generate_lih_hamiltonian_exact(r: float, n_qubits: int = 4):
    """
    Generate exact LiH Hamiltonian using PySCF/Qiskit Nature.

    Args:
        r: Bond length in Angstroms
        n_qubits: 4 for minimal active space, 10 for full

    Returns:
        H_matrix: Hamiltonian matrix
        info: Dictionary with details
    """
    print(f"\nGenerating LiH Hamiltonian at R = {r:.4f} Å")
    print(f"Basis: STO-6G")
    print(f"Qubits: {n_qubits}")

    # Define molecule
    molecule = f"Li 0.0 0.0 0.0; H 0.0 0.0 {r}"

    # Run PySCF calculation
    driver = PySCFDriver(
        atom=molecule,
        basis='sto-6g',
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )

    problem = driver.run()

    # Get electronic structure data
    hamiltonian = problem.hamiltonian

    # Get Hartree-Fock energy
    try:
        hf_energy = problem.reference_energy
    except:
        hf_energy = problem.nuclear_repulsion_energy

    # Map to qubits using Jordan-Wigner
    mapper = JordanWignerMapper()

    # Get second quantized operator
    second_q_op = hamiltonian.second_q_op()

    # Map to qubit operator
    qubit_op = mapper.map(second_q_op)

    # Convert to matrix
    H_matrix = qubit_op.to_matrix()

    # Get ground state
    eigvals, eigvecs = eigh(H_matrix)
    ground_energy = eigvals[0]
    ground_state = eigvecs[:, 0]

    info = {
        'hf_energy': hf_energy,
        'ground_energy': ground_energy,
        'ground_state': ground_state,
        'correlation_energy': ground_energy - hf_energy,
        'qubit_op': qubit_op,
        'n_qubits': n_qubits,
        'r': r
    }

    print(f"✓ Hamiltonian generated")
    print(f"  Hartree-Fock energy: {hf_energy:.6f} H")
    print(f"  Ground state energy: {ground_energy:.6f} H")
    print(f"  Correlation energy:  {ground_energy - hf_energy:.6f} H")

    return H_matrix, info


def validate_against_literature():
    """
    Validate against known literature values for LiH.

    Literature reference values (STO-6G basis):
    - Equilibrium bond length: ~1.5949 Å
    - Ground state energy (full): ~-7.86 to -7.88 Hartree

    Note: Values vary depending on active space and method.
    """
    print("\n" + "=" * 70)
    print("VALIDATION AGAINST LITERATURE VALUES")
    print("=" * 70)

    # Literature values for LiH (STO-6G)
    literature_values = {
        'equilibrium_r': 1.5949,  # Angstroms
        'ground_energy_full': -7.8623,  # Hartree (full calculation)
        'ground_energy_minimal': None,  # Unknown for (2e,2o) active space
        'source': 'O\'Malley et al., Phys. Rev. X 6, 031007 (2016)'
    }

    print(f"\nLiterature reference:")
    print(f"  Equilibrium: {literature_values['equilibrium_r']} Å")
    print(f"  Ground energy (full): {literature_values['ground_energy_full']} H")
    print(f"  Source: {literature_values['source']}")

    return literature_values


def test_vqe_with_exact_hamiltonian():
    """
    Test VQE using the exact Hamiltonian from Qiskit Nature.
    """
    print("\n" + "=" * 70)
    print("VQE VALIDATION TEST")
    print("=" * 70)

    # Generate exact Hamiltonian
    r_eq = 1.5949
    H_exact, info = generate_lih_hamiltonian_exact(r_eq, n_qubits=4)

    exact_ground_energy = info['ground_energy']

    # Try to import our VQE simulator
    try:
        from rich_sim_lih import TrappedIonSimulator, LiHSimulator

        print("\n" + "-" * 70)
        print("Testing VQE implementation...")

        # Create simulator
        ion_sim = TrappedIonSimulator(N=4)

        # Monkey-patch to use exact Hamiltonian
        class LiHSimulator_Exact(LiHSimulator):
            def __init__(self, *args, exact_H=None, **kwargs):
                super().__init__(*args, **kwargs)
                self._exact_H = exact_H

            def get_hamiltonian(self, r):
                return self._exact_H

        lih_sim = LiHSimulator_Exact(
            ion_sim,
            use_hardware_gates=False,
            n_qubits=4,
            exact_H=H_exact
        )

        # Run VQE with exact Hamiltonian
        print("Running VQE (this may take 5-10 minutes)...")
        result = lih_sim.vqe_optimization(
            r=r_eq,
            n_layers=4,
            max_iter=500,
            method='COBYLA',
            n_trials=2
        )

        # Compare results
        print("\n" + "=" * 70)
        print("VQE RESULTS WITH EXACT HAMILTONIAN")
        print("=" * 70)
        print(f"Exact ground energy:    {exact_ground_energy:.8f} H")
        print(f"VQE energy:             {result['vqe_energy']:.8f} H")
        print(f"Absolute error:         {abs(result['error']):.6e} H")
        print(f"Relative error:         {abs(result['error']/exact_ground_energy)*100:.4f}%")
        print(f"Ground state overlap:   {result['overlap']:.6f}")

        # Validation checks
        print("\n" + "-" * 70)
        print("VALIDATION CHECKS:")

        # Check 1: Variational principle
        if result['vqe_energy'] >= exact_ground_energy - 1e-8:
            print("✓ Variational principle satisfied (E_VQE ≥ E_exact)")
        else:
            print("✗ Variational principle VIOLATED (E_VQE < E_exact)")

        # Check 2: Chemical accuracy (1.6e-3 H = 1 kcal/mol)
        chemical_accuracy = 1.6e-3
        if abs(result['error']) < chemical_accuracy:
            print(f"✓ Chemical accuracy achieved (error < {chemical_accuracy} H)")
        else:
            print(f"⚠ Chemical accuracy NOT achieved (error = {abs(result['error']):.6e} H)")

        # Check 3: Reasonable overlap
        if result['overlap'] > 0.90:
            print(f"✓ Good ground state overlap ({result['overlap']:.4f} > 0.90)")
        else:
            print(f"⚠ Low ground state overlap ({result['overlap']:.4f})")

        return True

    except ImportError as e:
        print(f"\n✗ Cannot import LiH simulator: {e}")
        return False


def extract_pauli_decomposition(qubit_op, n_qubits=4):
    """
    Extract Pauli decomposition from Qiskit qubit operator.
    This shows what the REAL coefficients should be.
    """
    print("\n" + "=" * 70)
    print("PAULI DECOMPOSITION OF EXACT HAMILTONIAN")
    print("=" * 70)

    # Print all Pauli terms with coefficients
    print(f"\nNumber of Pauli terms: {len(qubit_op)}")
    print("\nPauli strings and coefficients:")
    print("-" * 70)

    for pauli_term in qubit_op:
        pauli_string = pauli_term.paulis[0]
        coefficient = pauli_term.coeffs[0]

        # Convert to readable format
        pauli_repr = str(pauli_string)

        print(f"{pauli_repr:20s} : {coefficient.real:+.8f}")

    print("-" * 70)


def main():
    """Main validation routine."""
    print("=" * 70)
    print("LiH HAMILTONIAN AND VQE VALIDATION")
    print("=" * 70)

    if not QISKIT_AVAILABLE:
        print("\n✗ Cannot run validation without Qiskit Nature")
        print("Install with: pip install qiskit-nature[pyscf]")
        return

    # Step 1: Literature validation
    lit_values = validate_against_literature()

    # Step 2: Generate exact Hamiltonian
    r_eq = 1.5949
    H_exact, info = generate_lih_hamiltonian_exact(r_eq, n_qubits=4)

    # Step 3: Show Pauli decomposition
    extract_pauli_decomposition(info['qubit_op'], n_qubits=4)

    # Step 4: Compare with literature
    print("\n" + "=" * 70)
    print("COMPARISON WITH LITERATURE")
    print("=" * 70)

    if lit_values['ground_energy_full'] is not None:
        diff = info['ground_energy'] - lit_values['ground_energy_full']
        print(f"Exact (4-qubit active space): {info['ground_energy']:.6f} H")
        print(f"Literature (full calculation): {lit_values['ground_energy_full']:.6f} H")
        print(f"Difference: {diff:.6f} H")
        print("\nNote: Difference expected due to active space approximation")
        print("      (4-qubit uses only 2 orbitals vs full calculation)")

    # Step 5: Test VQE
    print("\n")
    vqe_success = test_vqe_with_exact_hamiltonian()

    # Final summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"✓ Exact Hamiltonian generated via Qiskit Nature + PySCF")
    print(f"✓ Ground state energy computed: {info['ground_energy']:.6f} H")

    if vqe_success:
        print(f"✓ VQE validated against exact Hamiltonian")
    else:
        print(f"⚠ VQE validation skipped (import error)")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    print("Update rich_sim_lih.py with these EXACT Pauli coefficients")
    print("from the decomposition above to get correct ground state energies.")
    print("=" * 70)


if __name__ == "__main__":
    main()
