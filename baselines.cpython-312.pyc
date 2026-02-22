"""
core/w_matrix.py  —  W-Matrix (Slow Latent Space / "Forever Memory")
=====================================================================

In P-KAS: the W matrix stores the slow-learning coupling strengths
between oscillators.  It is the "forever memory" — the accumulated
structural wisdom of all past struggles.

In Deerskin-PKAS: W additionally encodes geometric orientation biases.
Edges that chronically accumulate calcium get strengthened here —
the solver learns structural priors about problem difficulty.

Three learning modes
--------------------
1. Hebbian:   W[i,j] += lr * Ca[i,j]                 (pure calcium-driven)
2. Anti-Hebb: W[i,j] -= lr * (1 - Ca[i,j]) * adj    (decay non-struggling edges)
3. Geometric: W[i,j] += lr * stress_gradient(i,j)    (Deerskin gauge force)

The combination creates a coupling matrix that:
  - Strengthens connections where the solver struggles (calcium)
  - Weakens connections that are never contested (anti-Hebbian)
  - Biases toward geometric orientations that minimised stress (melt)
"""

import numpy as np
from core.calcium import CalciumMemory


class WMatrix:
    """
    Slow-learning coupling matrix encoding structural problem knowledge.

    Parameters
    ----------
    n_nodes   : graph size
    lr        : learning rate (much slower than phase dynamics)
    decay     : weight decay to prevent runaway growth
    geo_scale : scale factor for geometric gradient updates
    """

    def __init__(
        self,
        n_nodes:   int,
        lr:        float = 0.001,
        decay:     float = 0.9999,
        geo_scale: float = 0.1,
    ):
        self.n = n_nodes
        self.lr = lr
        self.decay = decay
        self.geo_scale = geo_scale

        # Initialise with small random coupling
        rng = np.random.default_rng(42)
        self.W = rng.uniform(-0.1, 0.1, (n_nodes, n_nodes))
        np.fill_diagonal(self.W, 0.0)
        self.W = (self.W + self.W.T) / 2.0   # enforce symmetry

        # Learning history
        self.update_count = 0

    # ------------------------------------------------------------------
    def hebbian_update(
        self,
        calcium: CalciumMemory,
        adj: np.ndarray,
    ) -> None:
        """
        Calcium-Hebbian: strengthen edges where solver struggled most.

        Edges that chronically accumulate calcium get larger W entries,
        meaning the phase coupling there is stronger in future steps.
        This is the "lingering trail" — the solver remembers hard spots.
        """
        potentiate = calcium.potentiation_mask().astype(float)
        self.W += self.lr * calcium.Ca * potentiate * adj
        self.W -= self.lr * 0.1 * self.W * (1.0 - adj)  # anti-Hebbian on non-edges
        # Symmetrise and decay
        self.W = (self.W + self.W.T) / 2.0
        self.W *= self.decay
        np.fill_diagonal(self.W, 0.0)
        self.update_count += 1

    def geometric_update(
        self,
        spins: np.ndarray,
        adj: np.ndarray,
        encoder,          # GraphGeometryEncoder
    ) -> None:
        """
        Deerskin geometric update: W learns orientation biases.

        Edges where same-spin neighbours create high geometric stress
        get their coupling strengthened — the W matrix encodes a
        structural map of the problem's difficulty landscape.
        """
        n = self.n
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    stress = encoder.edge_stress(int(spins[u]), int(spins[v]))
                    delta = self.lr * self.geo_scale * stress
                    self.W[u, v] += delta
                    self.W[v, u] += delta
        np.fill_diagonal(self.W, 0.0)

    def coupling_force(self, phases: np.ndarray) -> np.ndarray:
        """
        Phase coupling force from current W state.

        F[i] = Σ_j  W[i,j] * sin(phase_j - phase_i)

        This is the Kuramoto-like anti-sync drive for MaxCut:
        connected nodes with large W coupling are pushed to
        opposite phases (→ opposite spins → cut edge).
        """
        phase_diffs = phases[None, :] - phases[:, None]   # (n, n)
        return np.sum(self.W * np.sin(phase_diffs), axis=1)  # (n,)

    def effective_coupling(self, adj: np.ndarray) -> np.ndarray:
        """W masked to actual graph edges — for visualisation."""
        return self.W * adj

    def reset(self) -> None:
        """Reset to small random state (new problem instance)."""
        rng = np.random.default_rng(self.update_count)
        self.W = rng.uniform(-0.1, 0.1, (self.n, self.n))
        np.fill_diagonal(self.W, 0.0)
        self.W = (self.W + self.W.T) / 2.0
