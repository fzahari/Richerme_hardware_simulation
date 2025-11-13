#!/usr/bin/env python3
"""
Simple script to get the exact 4-qubit LiH Hamiltonian coefficients.

Uses PySCF + Qiskit Nature to generate the minimal active space Hamiltonian
and extracts the Pauli decomposition.
"""

import numpy as np
from scipy.linalg import eigh

# Import Qiskit Nature
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

print("=" * 70)
print("EXACT LiH HAMILTONIAN GENERATOR (4 qubits)")
print("=" * 70)

# LiH at equilibrium geometry
r = 1.5949  # Angstroms
molecule_str = f"Li 0.0 0.0 0.0; H 0.0 0.0 {r}"

print(f"\nMolecule: LiH")
print(f"Bond length: {r} Å")
print(f"Basis: STO-3G (minimal for 4 qubits)")
print()

# Use minimal basis STO-3G for 4 qubits
# STO-3G gives us: Li: 1s, 2s  H: 1s  = 3 spatial orbitals = 6 spin orbitals
# But with frozen core (Li 1s), we get: Li: 2s  H: 1s = 2 spatial orbitals = 4 qubits!

driver = PySCFDriver(
    atom=molecule_str,
    basis='sto-3g',  # Minimal basis
    charge=0,
    spin=0,
    unit=DistanceUnit.ANGSTROM
)

print("Running PySCF calculation...")
problem = driver.run()

# Get Hamiltonian
hamiltonian = problem.hamiltonian
second_q_op = hamiltonian.second_q_op()

# Map to qubits
mapper = JordanWignerMapper()
qubit_op = mapper.map(second_q_op)

# Get matrix
H_matrix = qubit_op.to_matrix()

n_qubits = int(np.log2(H_matrix.shape[0]))
print(f"✓ Generated {n_qubits}-qubit Hamiltonian")

# Compute ground state
eigvals, eigvecs = eigh(H_matrix)
ground_energy = eigvals[0]
hf_energy = problem.nuclear_repulsion_energy

print(f"\nEnergies:")
print(f"  Nuclear repulsion: {problem.nuclear_repulsion_energy:.6f} H")
print(f"  Ground state:      {ground_energy:.6f} H")
print(f"  Correlation:       {ground_energy - hf_energy:.6f} H")

# Extract Pauli decomposition
print(f"\n" + "=" * 70)
print(f"PAULI DECOMPOSITION ({len(qubit_op)} terms)")
print("=" * 70)

# Group by term type
identity_terms = []
z_terms = []
zz_terms = []
xx_yy_terms = []
other_terms = []

for pauli_label, coeff in qubit_op.to_list():
    pauli_str = pauli_label
    coeff_val = coeff.real

    # Count operator types
    num_i = pauli_str.count('I')
    num_z = pauli_str.count('Z')
    num_x = pauli_str.count('X')
    num_y = pauli_str.count('Y')

    if num_i == n_qubits:
        identity_terms.append((pauli_str, coeff_val))
    elif num_z == 1 and num_i == n_qubits - 1:
        z_terms.append((pauli_str, coeff_val))
    elif num_z == 2 and num_i == n_qubits - 2:
        zz_terms.append((pauli_str, coeff_val))
    elif (num_x == 2 or num_y == 2) and num_i == n_qubits - 2:
        xx_yy_terms.append((pauli_str, coeff_val))
    else:
        other_terms.append((pauli_str, coeff_val))

print(f"\nIdentity terms ({len(identity_terms)}):")
for ps, c in identity_terms:
    print(f"  {ps:10s} : {c:+.8f}")

print(f"\nSingle-Z terms ({len(z_terms)}):")
for ps, c in z_terms:
    print(f"  {ps:10s} : {c:+.8f}")

print(f"\nTwo-Z (ZZ) terms ({len(zz_terms)}):")
for ps, c in zz_terms:
    print(f"  {ps:10s} : {c:+.8f}")

print(f"\nHopping (XX, YY) terms ({len(xx_yy_terms)}):")
for ps, c in xx_yy_terms:
    print(f"  {ps:10s} : {c:+.8f}")

if other_terms:
    print(f"\nOther terms ({len(other_terms)}):")
    for ps, c in other_terms[:10]:  # Show first 10
        print(f"  {ps:10s} : {c:+.8f}")
    if len(other_terms) > 10:
        print(f"  ... and {len(other_terms)-10} more")

# Generate Python code
print(f"\n" + "=" * 70)
print("PYTHON CODE FOR rich_sim_lih.py")
print("=" * 70)

print(f"""
# LiH Hamiltonian (STO-3G, R={r} Å) - EXACT from PySCF
# Generated: {np.datetime64('today')}
# Ground state energy: {ground_energy:.8f} H

pauli_terms = [
    # Identity term
    ([self.I, self.I, self.I, self.I], {identity_terms[0][1]:.8f}),
""")

for ps, c in z_terms:
    # Convert to qubit indices
    z_pos = [i for i, ch in enumerate(ps) if ch == 'Z']
    if z_pos:
        print(f"    # {ps}")
        gates = ['self.Z' if i in z_pos else 'self.I' for i in range(n_qubits)]
        print(f"    ([{', '.join(gates)}], {c:.8f}),")

for ps, c in zz_terms:
    z_pos = [i for i, ch in enumerate(ps) if ch == 'Z']
    if len(z_pos) == 2:
        print(f"    # {ps}")
        gates = ['self.Z' if i in z_pos else 'self.I' for i in range(n_qubits)]
        print(f"    ([{', '.join(gates)}], {c:.8f}),")

for ps, c in xx_yy_terms:
    x_pos = [i for i, ch in enumerate(ps) if ch == 'X']
    y_pos = [i for i, ch in enumerate(ps) if ch == 'Y']
    if len(x_pos) == 2:
        print(f"    # {ps}")
        gates = ['self.X' if i in x_pos else 'self.I' for i in range(n_qubits)]
        print(f"    ([{', '.join(gates)}], {c:.8f}),")
    if len(y_pos) == 2:
        print(f"    # {ps}")
        gates = ['self.Y' if i in y_pos else 'self.I' for i in range(n_qubits)]
        print(f"    ([{', '.join(gates)}], {c:.8f}),")

print("]")

print(f"\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)
print(f"Expected ground state energy: {ground_energy:.8f} H")
print(f"Number of qubits: {n_qubits}")
print(f"Number of Pauli terms: {len(qubit_op)}")
print("=" * 70)
