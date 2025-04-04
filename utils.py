import numpy as np


def op_to_superop(op):
    superop = np.kron(op, op.conj())
    if op.shape == (2, 2):
        return superop.reshape([2] * 4)
    return superop.reshape([2] * 8)


def apply_superop_1q(rho, super_op, i):
    n = rho.ndim // 2
    rho = np.tensordot(super_op, rho, axes=[[2, 3], [i, n+i]])
    rho = np.moveaxis(rho, source=[0, 1], destination=[i, n+i])
    return rho


def apply_superop_2q(rho, super_op, i, j):
    n = rho.ndim // 2
    rho = np.tensordot(super_op, rho, axes=[[4, 5, 6, 7], [i, j, n+i, n+j]])
    rho = np.moveaxis(rho, source=[0, 1, 2, 3], destination=[i, j, n+i, n+j])
    return rho


def apply_gate_1q(psi, gate, i):
    psi = np.tensordot(gate, psi, axes=[[-1], [i]])
    psi = np.moveaxis(psi, 0, i)
    return psi


def apply_gate_2q(psi, gate, i, j):
    psi = np.tensordot(gate.reshape([2] * 4), psi, axes=[[-2, -1], [i, j]])
    psi = np.moveaxis(psi, [0, 1], [i, j])
    return psi


def apply_random_gate_1q(psi, random_func, sigma, i):
    gate = random_func(sigma)
    psi = apply_gate_1q(psi, gate, i)
    return psi
