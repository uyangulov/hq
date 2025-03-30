import numpy as np


def op_to_superop(op):
    if op.shape == (2, 2):  # 1-qubit case
        return np.einsum('ab,cd->abcd', op, op.conj())
    elif op.shape == (2, 2, 2, 2):  # 2-qubit case
        return np.einsum('abcd,efgh->abcdefgh', op, op.conj())
    else:
        raise ValueError("Invalid operator shape for superoperator conversion")


def apply_superop_1q(rho, super_op, i):
    n = rho.ndim // 2
    rho = np.tensordot(super_op, rho, axes=[[1, 3], [i, n+i]])
    rho = np.moveaxis(rho, source=[0, 1], destination=[i, n+i])
    return rho


def apply_superop_2q(rho, super_op, i, j):
    n = rho.ndim // 2
    rho = np.tensordot(super_op, rho, axes=[[2, 3, 6, 7], [i, j, n+i, n+j]])
    rho = np.moveaxis(rho, source=[0, 1, 2, 3], destination=[i, j, n+i, n+j])
    return rho


def apply_gate_1q(psi, gate, i):
    psi = np.tensordot(psi, gate, axes=[[i], [-1]])
    psi = np.moveaxis(psi, 0, i)
    return psi


def apply_gate_2q(psi, gate, i, j):
    psi = np.tensordot(psi, gate, axes=[[i, j], [-2, -1]])
    psi = np.moveaxis(psi, [0, 1], [i, j])
    return psi


def apply_random_gate_1q(psi, random_func, sigma, i):
    gate = random_func(sigma)
    psi = apply_gate_1q(psi, gate, i)
    return psi