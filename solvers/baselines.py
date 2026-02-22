"""
solvers/baselines.py  —  Classical Comparison Solvers
======================================================

Fair baselines for benchmarking Deerskin-PKAS:

1. SimulatedAnnealing   — the gold standard classical solver
2. OriginalPKAS         — the broken original (P-KAS without Deerskin)
                          reproduces the collapse seen in q4.png
3. QuantumInspired      — QAOA-style tanh relaxation + rounding
"""

import numpy as np
import time
from typing import Tuple, List


# ---------------------------------------------------------------------------
# 1. Classical Simulated Annealing
# ---------------------------------------------------------------------------

class SimulatedAnnealing:
    """
    Textbook MaxCut SA.
    Explicit temperature schedule: T(t) = T0 * alpha^t
    This is what Deerskin replaces with geometric viscosity.
    """

    def __init__(
        self,
        adj: np.ndarray,
        T0: float = 2.0,
        alpha: float = 0.9995,
        max_iter: int = 5000,
    ):
        self.adj = np.asarray(adj, dtype=float)
        self.n = adj.shape[0]
        self.T0 = T0
        self.alpha = alpha
        self.max_iter = max_iter

    def _delta_cut(self, spins: np.ndarray, flip_node: int) -> int:
        """Change in cut value if we flip node i."""
        i = flip_node
        neighbours = np.where(self.adj[i] > 0)[0]
        delta = 0
        for j in neighbours:
            # Before flip: same spin = no cut; after flip: cut gained
            if spins[i] == spins[j]:
                delta += 1   # gain a cut
            else:
                delta -= 1   # lose a cut
        return delta

    def solve(self) -> Tuple[np.ndarray, int, List[int]]:
        rng = np.random.default_rng(42)
        spins = rng.choice([-1, 1], size=self.n)
        best_spins = spins.copy()

        n_edges = int(self.adj.sum()) // 2
        cur_cut  = int(np.sum([
            self.adj[u, v]
            for u in range(self.n)
            for v in range(u + 1, self.n)
            if self.adj[u, v] and spins[u] != spins[v]
        ]))
        best_cut = cur_cut
        history  = [cur_cut]
        T = self.T0

        for step in range(self.max_iter):
            i     = rng.integers(0, self.n)
            delta = self._delta_cut(spins, i)

            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-9)):
                spins[i] = -spins[i]
                cur_cut  += delta
                if cur_cut > best_cut:
                    best_cut  = cur_cut
                    best_spins = spins.copy()

            T *= self.alpha
            history.append(cur_cut)

        return best_spins, best_cut, history


# ---------------------------------------------------------------------------
# 2. Original P-KAS  (the broken version — reproduced faithfully)
# ---------------------------------------------------------------------------

class OriginalPKAS:
    """
    The original P-KAS as described in the conversation history.
    Kuramoto phases + calcium + W matrix, but NO geometric Prior.
    Temperature is fixed; no anti-collapse mechanism.

    This faithfully reproduces the phase collapse seen in q4.png.
    """

    def __init__(
        self,
        adj: np.ndarray,
        dt: float = 0.05,
        max_iter: int = 5000,
        field_strength: float = 1.0,
        ca_decay: float = 0.95,
        w_lr: float = 0.001,
    ):
        self.adj = np.asarray(adj, dtype=float)
        self.n = adj.shape[0]
        self.dt = dt
        self.max_iter = max_iter
        self.field_strength = field_strength
        self.ca_decay = ca_decay
        self.w_lr = w_lr

    def solve(self) -> Tuple[np.ndarray, int, List[int]]:
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, self.n)
        W = rng.uniform(-0.1, 0.1, (self.n, self.n))
        np.fill_diagonal(W, 0)
        W = (W + W.T) / 2

        Ca = np.zeros((self.n, self.n))
        history  = []
        best_cut = 0
        best_spins = np.ones(self.n, dtype=int)

        for step in range(self.max_iter):
            # Phase dynamics (original P-KAS, no geometric prior)
            phase_diffs   = phases[:, None] - phases[None, :]
            coupling_force = np.sum(W * np.sin(-phase_diffs), axis=1)  # anti-sync
            field_force    = self.field_strength * np.sin(phases)        # bias field
            noise          = 0.1 * rng.standard_normal(self.n)
            phases        += self.dt * (coupling_force + field_force + noise)
            phases         = phases % (2 * np.pi)

            # Calcium
            coherence = np.cos(phase_diffs) ** 2
            Ca = (Ca + coherence * self.adj * self.dt) * self.ca_decay
            Ca = np.clip(Ca, 0, 1)

            # W update
            if step % 10 == 0:
                W += self.w_lr * Ca * self.adj
                W = (W + W.T) / 2
                np.fill_diagonal(W, 0)
                W *= 0.9999

            # Binarise
            spins  = np.where(np.cos(phases) >= 0, 1, -1)
            n_cut  = int(np.sum([
                self.adj[u, v]
                for u in range(self.n)
                for v in range(u + 1, self.n)
                if self.adj[u, v] and spins[u] != spins[v]
            ]))
            history.append(n_cut)
            if n_cut > best_cut:
                best_cut   = n_cut
                best_spins = spins.copy()

        return best_spins, best_cut, history


# ---------------------------------------------------------------------------
# 3. Quantum-Inspired (tanh relaxation)
# ---------------------------------------------------------------------------

class QuantumInspired:
    """
    QAOA-inspired continuous relaxation.
    Variables are real-valued in [-1, +1] via tanh.
    Gradient descent on the cut objective.
    Final step: round to ±1.
    """

    def __init__(
        self,
        adj: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 5000,
        beta_start: float = 0.5,
        beta_end: float = 10.0,
    ):
        self.adj = np.asarray(adj, dtype=float)
        self.n = adj.shape[0]
        self.lr = lr
        self.max_iter = max_iter
        self.beta_start = beta_start
        self.beta_end   = beta_end

    def solve(self) -> Tuple[np.ndarray, int, List[int]]:
        rng = np.random.default_rng(42)
        x = rng.standard_normal(self.n) * 0.1   # soft variables
        history  = []
        best_cut = 0
        best_spins = np.ones(self.n, dtype=int)

        betas = np.linspace(self.beta_start, self.beta_end, self.max_iter)

        for step, beta in enumerate(betas):
            s = np.tanh(beta * x)   # soft spins in (-1, 1)

            # Gradient of MaxCut relaxation: ∂/∂x[i] Σ_{(i,j)∈E} (1 - s_i·s_j)/2
            grad_s = -0.5 * (self.adj @ s)   # ∂cut/∂s (continuous)
            # Chain rule: ∂/∂x = ∂cut/∂s * ∂s/∂x
            grad_x = grad_s * beta * (1 - s ** 2)

            x -= self.lr * grad_x
            x += rng.standard_normal(self.n) * 0.01 / (1 + step * 0.001)

            # Round and evaluate
            spins  = np.sign(x).astype(int)
            spins[spins == 0] = 1
            n_cut  = int(np.sum([
                self.adj[u, v]
                for u in range(self.n)
                for v in range(u + 1, self.n)
                if self.adj[u, v] and spins[u] != spins[v]
            ]))
            history.append(n_cut)
            if n_cut > best_cut:
                best_cut   = n_cut
                best_spins = spins.copy()

        return best_spins, best_cut, history
