from __future__ import annotations
import numpy as np
from typing import List

# ===== Linear algebra helpers =====
def _kronN(ops: List[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0+0.0j]])
    for a in ops:
        out = np.kron(out, a)
    return out

I2 = np.eye(2, dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Y = np.array([[0,-1j],[1j,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

def _pauli(n:int, P:str, i:int) -> np.ndarray:
    base = {'I':I2,'X':X,'Y':Y,'Z':Z}[P]
    return _kronN([base if k==i else I2 for k in range(n)])

def _sum_pauli(n:int, P:str) -> np.ndarray:
    return sum(_pauli(n,P,i) for i in range(n))

def _expm(H: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(H)
    return (v * np.exp(-1j * t * w)) @ v.conj().T

# ===== Single-qubit rotations =====
def Rz(n:int, i:int, theta: float) -> np.ndarray:
    return _expm(0.5 * _pauli(n,'Z',i), theta)

def Rx(n:int, i:int, theta: float) -> np.ndarray:
    return _expm(0.5 * _pauli(n,'X',i), theta)

def Ry(n:int, i:int, theta: float) -> np.ndarray:
    return _expm(0.5 * _pauli(n,'Y',i), theta)

# ===== Global entangling operation (UMQ) =====
def UMQ(n:int, chi: float) -> np.ndarray:
    """
    Global MS/GMS-like block:
      UMQ(chi) = exp(-i * (chi/4) * (sum_i X_i)^2)
    Global phase from the N*I term is irrelevant for dynamics.
    """
    Sx = _sum_pauli(n,'X')
    H = 0.25 * (Sx @ Sx)  # Hermitian
    return _expm(H, chi)

# ===== Targets =====
def target_pauli_string_unitary(letters: str, t: float) -> np.ndarray:
    """
    Return U = exp(-i * t * (⊗_i P_i)) for letters like 'ZXX'.
    """
    n = len(letters)
    H = _kronN([{'I':I2,'X':X,'Y':Y,'Z':Z}[ch] for ch in letters])
    return _expm(H, t)

# ===== n-body via Fig.-6 style pattern =====
def _build_core_string(n:int, t: float, internal_choice: str='X', flip_sign: bool=False) -> np.ndarray:
    """
    internal_choice: 'X' builds exp(-i t * X1 X2 ... Xn) up to 1q conjugations,
                     'Y' builds exp(-i t * Y1 X2 ... Xn) up to phase.
    Sequence:
      R_pre(±π/2 on qubit 0)
      UMQ(+π/2)
      Rz_0(±2t)   # sign may flip with convention; 'flip_sign' toggles it
      UMQ(-π/2)
      R_post(∓π/2 on qubit 0)
    """
    if internal_choice == 'X':
        Rpre = Ry(n, 0, -np.pi/2)
        Rpost = Ry(n, 0, +np.pi/2)
    elif internal_choice == 'Y':
        # IMPORTANT sign convention for exactness
        Rpre = Rx(n, 0, -np.pi/2)
        Rpost = Rx(n, 0, +np.pi/2)
    else:
        raise ValueError("internal_choice must be 'X' or 'Y'")

    U = Rpre
    U = UMQ(n, +np.pi/2) @ U
    angle = -2*t if flip_sign else +2*t
    U = Rz(n, 0, angle) @ U
    U = UMQ(n, -np.pi/2) @ U
    U = Rpost @ U
    return U

def n_body_string(axes: List[str], t: float, flip_sign: bool=False) -> np.ndarray:
    """
    Build U = exp(-i t * (P1 ⊗ P2 ⊗ ... ⊗ Pn)) for axes like ['Z','X','X'].
    This helper expects X on qubits 1..n-1 (first qubit can be X/Y/Z).
    """
    n = len(axes)
    assert all(a in ('X','Y','Z') for a in axes), "Only X/Y/Z supported"
    assert all(a=='X' for a in axes[1:]), "Helper expects X on qubits 1..n-1"

    target_first = axes[0]
    if target_first == 'X':
        Ucore = _build_core_string(n, t, internal_choice='X', flip_sign=flip_sign)
        U = Ucore
    elif target_first == 'Y':
        Ucore = _build_core_string(n, t, internal_choice='Y', flip_sign=flip_sign)
        U = Ucore
    else:  # 'Z'
        # Make YXX... then conjugate Y->Z on qubit 0 by Rx(+π/2) … Rx(-π/2)
        Ucore = _build_core_string(n, t, internal_choice='Y', flip_sign=flip_sign)
        U = Rx(n, 0, +np.pi/2) @ Ucore @ Rx(n, 0, -np.pi/2)
    return U

def Z1X2X3(t: float, flip_sign: bool=False) -> np.ndarray:
    """
    Convenience for 3-qubit U = exp(-i t * (Z ⊗ X ⊗ X)).
    """
    return n_body_string(['Z','X','X'], t, flip_sign=flip_sign)

# ===== (Optional) Multi-mode J_ij calculator for hardware realism =====
def Jij_from_multimode(
    B: np.ndarray,           # (N x N) normal-mode matrix; columns are mode vectors
    omega: np.ndarray,       # (N,) mode frequencies
    mus: np.ndarray,         # (M,) bichromatic tone detunings
    Omegas: np.ndarray,      # (M,) on-resonance Rabi frequencies per tone
    R_recoil: float          # recoil frequency prefactor
) -> np.ndarray:
    """
    Compute pairwise Ising couplings synthesized by multi-tone global driving:
      J_ij = sum_k [ (sum_m Omega_m^2 * R_recoil / (mu_m^2 - omega_k^2)) * B_{ik} B_{jk} ].
    Diagonal is zeroed; matrix is symmetrized.
    """
    N = B.shape[0]
    J = np.zeros((N,N), dtype=float)
    coeffs = np.zeros_like(omega, dtype=float)

    for k in range(N):
        s = 0.0
        for m in range(len(mus)):
            s += (Omegas[m]**2 * R_recoil) / (mus[m]**2 - omega[k]**2)
        coeffs[k] = s
        J += coeffs[k] * np.outer(B[:,k], B[:,k])

    J = 0.5*(J + J.T)
    np.fill_diagonal(J, 0.0)
    return J

# ===== Phase-invariant distance =====
def unitary_distance(U: np.ndarray, V: np.ndarray) -> float:
    """
    Spectral-norm distance up to global phase.
    """
    phase = np.angle(np.trace(U.conj().T @ V))
    return np.linalg.norm(U - np.exp(1j*phase)*V, ord=2) / np.linalg.norm(U, ord=2)

