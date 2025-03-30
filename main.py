import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils

H = np.array([
    [1, 1],
    [1, -1]
], dtype=np.complex64) / np.sqrt(2)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=np.complex64)

CZ = np.diag([1, 1, 1, -1]).astype(np.complex64)
GH = utils.op_to_superop(H)
GCZ = utils.op_to_superop(CZ)
I2 = np.eye(2, dtype=np.complex64)


def init_state(n, randomize=False):
    N = 2**n
    if randomize:
        psi = np.random.rand(N) + 1j * np.random.rand(N)
        psi /= np.linalg.norm(psi)
    else:
        psi = np.zeros(N, dtype=np.complex64)
        psi[0] = 1
    rho = np.outer(psi, np.conj(psi))
    psi = psi.reshape([2] * n)
    rho = rho.reshape([2] * 2 * n)
    return psi, rho


def gauss(sigma):
    return np.exp(-sigma**2 / 2)


def binom_prob(sigma):
    return 0.5 * (1 - gauss(sigma))


def normal_noise_gate(sigma):
    theta = sigma * np.random.randn()
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta/2) * X


def binomial_noise_gate(sigma):
    p = binom_prob(sigma)
    ops = [X, I2]
    return ops[np.random.choice([0, 1], p=[p, 1 - p])]


def perfect_noise_superop(sigma):
    p = binom_prob(sigma)
    first = (1-p) * utils.op_to_superop(I2)
    second = p * utils.op_to_superop(X)
    return first + second


def simulate_perfect(rho_init, n, reps, sigma):
    N = 2**n
    rho = rho_init.copy()
    op = perfect_noise_superop(sigma)
    for rep in range(reps):
        for i in range(n):
            rho = utils.apply_superop_1q(rho, GH, i)
            rho = utils.apply_superop_1q(rho, op, i)
        for i in range(n):
            rho = utils.apply_superop_2q(rho, GCZ, i, (i+1) % n)
    probs = np.diag(rho.reshape((N, N))).real
    return probs


def run_average(psi_init, n, reps, runs, probs_perfect, sigma, noise_kind):
    random_func = normal_noise_gate if noise_kind == 'normal' else binomial_noise_gate
    N = 2**n
    probs = np.zeros(N)
    for _ in range(runs):
        psi = psi_init.copy()
        for rep in range(reps):
            for i in range(n):
                psi = utils.apply_gate_1q(psi, H, i)
                op = random_func(sigma)
                psi = utils.apply_gate_1q(psi, op, i)
            for i in range(n):
                psi = utils.apply_gate_2q(psi, CZ, i, (i+1) % n)
        probs += np.abs(psi.reshape(N,))**2
    error = np.linalg.norm(probs / runs - probs_perfect)
    return error


def collect_stats(psi_init, n, reps, max_runs, probs_perfect, sigma, noise_kind):
    errors = []
    for runs in tqdm(max_runs, desc=f"{noise_kind} circuit"):
        error = run_average(
            psi_init, n, reps, runs, probs_perfect, sigma, noise_kind)
        errors.append(error)
    return errors


n_qubits = 8
runs = np.logspace(0, 3, num=20, base=10, dtype=int)
sigma = 1
p_X = binom_prob(sigma)
p_I = 1 - p_X

psi_init, rho_init = init_state(n_qubits, randomize=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.flatten()

for idx, reps in enumerate(range(1, 7, 1)):

    print(f"Reps = {reps}")

    probs_perfect = simulate_perfect(rho_init, n_qubits, reps, sigma)

    errors_binomial = collect_stats(
        psi_init, n_qubits, reps, runs, probs_perfect, sigma=sigma, noise_kind="binomial")
    errors_normal = collect_stats(
        psi_init, n_qubits, reps, runs, probs_perfect, sigma=sigma, noise_kind="normal")

    ax = axes[idx]
    ax.loglog(runs, errors_normal, marker='o',
              linestyle='-', label="Normal Noise")
    ax.loglog(runs, errors_binomial,
              linestyle='-', alpha=.5, label="Binomial Noise")
    ax.loglog(runs, 1/np.sqrt(runs),
              label=r"$\frac{1}{\sqrt{N}}$", ls=':')

    print(errors_binomial)
    print(errors_normal)

    ax.set_title(
        rf"block reps = {reps}, n_qubits = {n_qubits}" +
        "\n" +
        rf"$p_X$ = {p_X:.3f}, $p_I$ = {p_I:.3f}, $\sigma$ = {sigma}")

    ax.grid(linestyle=":")
    ax.set_ylabel(r"$|| \bar{p} - p_{true}||$")
    ax.set_xlabel("N_Shots")
    ax.legend()

plt.suptitle("Error vs Number of Shots for Different block reps")
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("graph")
plt.show()
