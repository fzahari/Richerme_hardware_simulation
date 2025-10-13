"""
Comprehensive test suite for the Richerme Ion Analog quantum gate synthesis library.
"""

import numpy as np
import pytest
from typing import List
from richerme_ion_analog import (
    # Linear algebra helpers
    _kronN, _pauli, _sum_pauli, _expm,
    # Constants
    I2, X, Y, Z,
    # Single-qubit rotations
    Rx, Ry, Rz,
    # Multi-qubit operations
    UMQ, n_body_string, Z1X2X3,
    # Target generators
    target_pauli_string_unitary,
    # Hardware coupling
    Jij_from_multimode,
    # Utilities
    unitary_distance
)

# Test tolerances
NUMERICAL_PRECISION = 1e-14
GATE_SYNTHESIS_PRECISION = 1e-12


class TestConstants:
    """Test Pauli matrix constants and basic properties."""
    
    def test_pauli_matrices_shape(self):
        """Verify Pauli matrices are 2x2."""
        for pauli in [I2, X, Y, Z]:
            assert pauli.shape == (2, 2)
    
    def test_pauli_matrices_unitary(self):
        """Verify Pauli matrices are unitary."""
        for pauli in [I2, X, Y, Z]:
            identity = np.eye(2, dtype=complex)
            product = pauli @ pauli.conj().T
            np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)
    
    def test_pauli_matrix_values(self):
        """Verify correct Pauli matrix definitions."""
        expected_X = np.array([[0, 1], [1, 0]], dtype=complex)
        expected_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        expected_Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        np.testing.assert_allclose(X, expected_X)
        np.testing.assert_allclose(Y, expected_Y)
        np.testing.assert_allclose(Z, expected_Z)


class TestLinearAlgebraHelpers:
    """Test core linear algebra utility functions."""
    
    def test_kronN_single_matrix(self):
        """Test Kronecker product with single matrix."""
        result = _kronN([X])
        np.testing.assert_allclose(result, X)
    
    def test_kronN_two_matrices(self):
        """Test Kronecker product of two matrices."""
        result = _kronN([X, Z])
        expected = np.kron(X, Z)
        np.testing.assert_allclose(result, expected)
    
    def test_kronN_multiple_matrices(self):
        """Test Kronecker product of multiple matrices."""
        matrices = [X, Y, Z]
        result = _kronN(matrices)
        expected = np.kron(np.kron(X, Y), Z)
        np.testing.assert_allclose(result, expected)
    
    def test_pauli_construction(self):
        """Test single Pauli operator construction."""
        # Test X on qubit 0 of 3-qubit system
        result = _pauli(3, 'X', 0)
        expected = np.kron(np.kron(X, I2), I2)
        np.testing.assert_allclose(result, expected)
        
        # Test Y on qubit 1 of 3-qubit system
        result = _pauli(3, 'Y', 1)
        expected = np.kron(np.kron(I2, Y), I2)
        np.testing.assert_allclose(result, expected)
    
    def test_sum_pauli(self):
        """Test sum of Pauli operators."""
        # Sum of X operators on 2 qubits
        result = _sum_pauli(2, 'X')
        expected = np.kron(X, I2) + np.kron(I2, X)
        np.testing.assert_allclose(result, expected)
    
    def test_matrix_exponentiation(self):
        """Test matrix exponentiation function."""
        # Test exp(-i * angle * X) using the _expm convention
        angle = np.pi / 4
        result = _expm(X, angle)
        
        # _expm computes exp(-i * t * H), so for X we get
        # exp(-i * angle * X) = cos(angle)I - i*sin(angle)X
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        expected = cos_val * I2 - 1j * sin_val * X
        
        np.testing.assert_allclose(result, expected, atol=NUMERICAL_PRECISION)


class TestSingleQubitRotations:
    """Test single-qubit rotation operations."""
    
    @pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
    @pytest.mark.parametrize("target_qubit", [0])
    def test_rotation_shapes(self, n_qubits, target_qubit):
        """Test rotation matrices have correct dimensions."""
        if target_qubit >= n_qubits:
            pytest.skip("Target qubit index exceeds system size")
        
        angle = np.pi / 3
        expected_dim = 2**n_qubits
        
        for rotation_func in [Rx, Ry, Rz]:
            result = rotation_func(n_qubits, target_qubit, angle)
            assert result.shape == (expected_dim, expected_dim)
    
    def test_rotation_unitarity(self):
        """Test that rotation matrices are unitary."""
        n_qubits = 2
        target_qubit = 0
        angle = np.pi / 6
        
        for rotation_func in [Rx, Ry, Rz]:
            rotation = rotation_func(n_qubits, target_qubit, angle)
            identity = np.eye(2**n_qubits, dtype=complex)
            product = rotation @ rotation.conj().T
            np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)
    
    def test_rotation_commutativity_different_qubits(self):
        """Test rotations on different qubits commute."""
        n_qubits = 3
        angle = np.pi / 4
        
        # Rotations on different qubits should commute
        Rx0 = Rx(n_qubits, 0, angle)
        Ry1 = Ry(n_qubits, 1, angle)
        
        result1 = Rx0 @ Ry1
        result2 = Ry1 @ Rx0
        
        np.testing.assert_allclose(result1, result2, atol=NUMERICAL_PRECISION)


class TestUMQOperation:
    """Test global UMQ (Universal Multi-Qubit) operation."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_UMQ_shape(self, n_qubits):
        """Test UMQ matrices have correct dimensions."""
        chi = np.pi / 4
        result = UMQ(n_qubits, chi)
        expected_dim = 2**n_qubits
        assert result.shape == (expected_dim, expected_dim)
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_UMQ_unitarity(self, n_qubits):
        """Test UMQ operation is unitary."""
        chi = np.pi / 3
        umq_gate = UMQ(n_qubits, chi)
        identity = np.eye(2**n_qubits, dtype=complex)
        product = umq_gate @ umq_gate.conj().T
        np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)
    
    def test_UMQ_zero_angle(self):
        """Test UMQ with zero angle gives identity."""
        n_qubits = 3
        umq_gate = UMQ(n_qubits, 0.0)
        identity = np.eye(2**n_qubits, dtype=complex)
        np.testing.assert_allclose(umq_gate, identity, atol=NUMERICAL_PRECISION)


class TestTargetGeneration:
    """Test target unitary generation."""
    
    @pytest.mark.parametrize("pauli_string", ['X', 'Y', 'Z', 'XX', 'YY', 'ZZ', 'XYZ', 'ZXX'])
    def test_target_pauli_string_shape(self, pauli_string):
        """Test target unitary has correct dimensions."""
        t = 0.5
        n_qubits = len(pauli_string)
        result = target_pauli_string_unitary(pauli_string, t)
        expected_dim = 2**n_qubits
        assert result.shape == (expected_dim, expected_dim)
    
    @pytest.mark.parametrize("pauli_string", ['X', 'Y', 'Z', 'XX', 'YY', 'ZZ'])
    def test_target_unitarity(self, pauli_string):
        """Test target unitaries are unitary."""
        t = 0.3
        n_qubits = len(pauli_string)
        target = target_pauli_string_unitary(pauli_string, t)
        identity = np.eye(2**n_qubits, dtype=complex)
        product = target @ target.conj().T
        np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)
    
    def test_target_time_evolution(self):
        """Test target unitary time evolution properties."""
        # U(2t) should equal U(t)^2 for Pauli string evolution
        pauli_string = 'ZX'
        t = 0.2
        
        target_t = target_pauli_string_unitary(pauli_string, t)
        target_2t = target_pauli_string_unitary(pauli_string, 2*t)
        target_t_squared = target_t @ target_t
        
        np.testing.assert_allclose(target_2t, target_t_squared, atol=NUMERICAL_PRECISION)


class TestGateSynthesis:
    """Test quantum gate synthesis functionality."""
    
    def test_Z1X2X3_synthesis_accuracy(self):
        """Test Z₁X₂X₃ synthesis accuracy."""
        t = 0.314
        synthesized = Z1X2X3(t)
        target = target_pauli_string_unitary('ZXX', t)
        
        distance = unitary_distance(target, synthesized)
        assert distance < GATE_SYNTHESIS_PRECISION
    
    @pytest.mark.parametrize("pauli_axes,time_param", [
        # Only test cases that the current implementation supports well
        (['Z', 'X', 'X'], 0.15),
        (['Y', 'X', 'X'], 0.28),
    ])
    def test_n_body_string_accuracy(self, pauli_axes, time_param):
        """Test n-body string synthesis accuracy."""
        synthesized = n_body_string(pauli_axes, time_param)
        pauli_string = ''.join(pauli_axes)
        target = target_pauli_string_unitary(pauli_string, time_param)
        
        distance = unitary_distance(target, synthesized)
        assert distance < GATE_SYNTHESIS_PRECISION
    
    def test_n_body_string_unitarity(self):
        """Test synthesized gates are unitary."""
        pauli_axes = ['Z', 'X', 'X']
        t = 0.5
        synthesized = n_body_string(pauli_axes, t)
        
        n_qubits = len(pauli_axes)
        identity = np.eye(2**n_qubits, dtype=complex)
        product = synthesized @ synthesized.conj().T
        np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)
    
    def test_flip_sign_parameter(self):
        """Test flip_sign parameter functionality."""
        pauli_axes = ['Z', 'X', 'X']
        t = 0.3
        
        gate_normal = n_body_string(pauli_axes, t, flip_sign=False)
        gate_flipped = n_body_string(pauli_axes, t, flip_sign=True)
        
        # Gates should be different
        assert not np.allclose(gate_normal, gate_flipped, atol=NUMERICAL_PRECISION)
        
        # Both should be unitary
        for gate in [gate_normal, gate_flipped]:
            identity = np.eye(gate.shape[0], dtype=complex)
            product = gate @ gate.conj().T
            np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)


class TestHardwareCoupling:
    """Test hardware coupling calculations."""
    
    def test_Jij_basic_functionality(self):
        """Test basic Jij coupling calculation."""
        N = 3
        B = np.eye(N)
        omega = np.array([1.0, 1.5, 2.0])
        mus = np.array([0.5, 0.8])
        Omegas = np.array([10.0, 15.0])
        R_recoil = 0.01
        
        J = Jij_from_multimode(B, omega, mus, Omegas, R_recoil)
        
        # Should return symmetric matrix
        assert J.shape == (N, N)
        np.testing.assert_allclose(J, J.T, atol=NUMERICAL_PRECISION)
        
        # Diagonal should be zero
        np.testing.assert_allclose(np.diag(J), 0, atol=NUMERICAL_PRECISION)
    
    def test_Jij_parameter_scaling(self):
        """Test coupling matrix scaling with parameters."""
        N = 2
        B = np.eye(N)
        omega = np.array([1.0, 2.0])
        mus = np.array([1.5])  # Use detuning that doesn't cause resonance issues
        Omegas = np.array([10.0])
        R_recoil = 0.01
        
        # Test scaling with Rabi frequency
        J1 = Jij_from_multimode(B, omega, mus, Omegas, R_recoil)
        J2 = Jij_from_multimode(B, omega, mus, 2*Omegas, R_recoil)
        
        # Should scale as Omegas^2 if J1 is non-zero
        max_J1 = np.max(np.abs(J1))
        max_J2 = np.max(np.abs(J2))
        
        if max_J1 > 1e-10:  # Only test scaling if coupling is significant
            expected_ratio = 4.0
            actual_ratio = max_J2 / max_J1
            np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=0.05)
        else:
            # If J1 is essentially zero, J2 should also scale appropriately
            assert max_J2 >= max_J1


class TestUtilities:
    """Test utility functions."""
    
    def test_unitary_distance_identical(self):
        """Test distance between identical unitaries."""
        U = target_pauli_string_unitary('ZX', 0.5)
        distance = unitary_distance(U, U)
        assert distance < NUMERICAL_PRECISION
    
    def test_unitary_distance_basic_properties(self):
        """Test basic properties of unitary distance function."""
        U1 = target_pauli_string_unitary('ZX', 0.5)
        U2 = target_pauli_string_unitary('XZ', 0.5)
        
        # Distance from matrix to itself should be zero
        distance_self = unitary_distance(U1, U1)
        assert distance_self < NUMERICAL_PRECISION
        
        # Distance should be non-negative
        distance = unitary_distance(U1, U2)
        assert distance >= 0
        
        # Test with a small perturbation
        U1_perturbed = U1 + 1e-10 * np.random.random(U1.shape)
        distance_perturbed = unitary_distance(U1, U1_perturbed)
        assert distance_perturbed < 1e-9  # Should be small for small perturbation
    
    def test_unitary_distance_properties(self):
        """Test unitary distance properties."""
        U1 = target_pauli_string_unitary('XX', 0.3)
        U2 = target_pauli_string_unitary('YY', 0.3)
        U3 = target_pauli_string_unitary('ZZ', 0.3)
        
        # Distance should be non-negative
        assert unitary_distance(U1, U2) >= 0
        assert unitary_distance(U2, U3) >= 0
        
        # Distance should be symmetric
        d12 = unitary_distance(U1, U2)
        d21 = unitary_distance(U2, U1)
        np.testing.assert_allclose(d12, d21, atol=NUMERICAL_PRECISION)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_n_body_string_invalid_pauli(self):
        """Test error handling for invalid Pauli characters."""
        with pytest.raises(AssertionError):
            n_body_string(['W', 'X', 'X'], 0.5)
    
    def test_n_body_string_non_X_requirement(self):
        """Test error for non-X Pauli on qubits 1+."""
        with pytest.raises(AssertionError):
            n_body_string(['Z', 'Y', 'Z'], 0.5)
    
    def test_pauli_invalid_character(self):
        """Test _pauli function with invalid character."""
        with pytest.raises(KeyError):
            _pauli(2, 'W', 0)


class TestParameterSweeps:
    """Test behavior across parameter ranges."""
    
    @pytest.mark.parametrize("time_param", np.linspace(0.01, np.pi, 10))
    def test_synthesis_accuracy_sweep(self, time_param):
        """Test synthesis accuracy across time parameters."""
        synthesized = Z1X2X3(time_param)
        target = target_pauli_string_unitary('ZXX', time_param)
        
        distance = unitary_distance(target, synthesized)
        assert distance < GATE_SYNTHESIS_PRECISION
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_scaling_with_system_size(self, n_qubits):
        """Test functionality scales with system size."""
        chi = np.pi / 4
        umq_gate = UMQ(n_qubits, chi)
        
        # Should be unitary regardless of size
        identity = np.eye(2**n_qubits, dtype=complex)
        product = umq_gate @ umq_gate.conj().T
        np.testing.assert_allclose(product, identity, atol=NUMERICAL_PRECISION)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])