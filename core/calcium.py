"""
core/calcium.py  —  Calcium Memory (Struggle Trace)
====================================================

Calcium in neuroscience: a slow second-messenger that lingers
at synapses where activity was intense.  It records *where the
system struggled* — a temporal integral of local conflict.

In P-KAS: Ca(t) = coherence * dt  accumulated, then decayed.
In Deerskin-PKAS: Ca records geometric stress hotspots across
the graph — which edges are chronically misaligned.

Three timescales
----------------
  Fast:   phase oscillation  (dt ~ 0.05)
  Medium: calcium trace      (decay ~ 0.95 per step)
  Slow:   W-matrix learning  (lr ~ 0.001)
  Glacial: viscous melt      (melt_rate ~ 0.003)

The calcium bridges fast dynamics to slow geometric scarring.
High chronic calcium on an edge → Prior melts toward that edge's
orientation → solver learns a structural bias for that problem class.
"""

import numpy as np
from typing import Optional


class CalciumMemory:
    """
    Per-edge calcium concentration tracking struggle across the graph.

    Parameters
    ----------
    n_nodes    : number of graph nodes
    decay      : multiplicative decay per step (0.95 = slow fade)
    buildup    : accumulation rate per unit coherence
    threshold  : calcium level that triggers W-matrix potentiation
    """

    def __init__(
        self,
        n_nodes: int,
        decay:     float = 0.95,
        buildup:   float = 1.0,
        threshold: float = 0.3,
    ):
        self.n = n_nodes
        self.decay = decay
        self.buildup = buildup
        self.threshold = threshold

        # Ca[i, j] = calcium at edge (i, j)
        self.Ca = np.zeros((n_nodes, n_nodes), dtype=float)
        # Running integral for heatmap / learning target
        self.Ca_integral = np.zeros((n_nodes, n_nodes), dtype=float)

    # ------------------------------------------------------------------
    def step(self, phases: np.ndarray, adj: np.ndarray, dt: float = 0.05) -> None:
        """
        Update calcium based on current phase coherence.

        Coherence between nodes i and j:
            coh(i,j) = cos²(phase_i - phase_j)

        High coherence on a connected edge → those nodes are fighting
        to anti-synchronise (MaxCut drive) → calcium builds there.

        Parameters
        ----------
        phases : (n,) oscillator phases in radians
        adj    : (n, n) adjacency matrix
        dt     : integration timestep
        """
        phase_diff = phases[:, None] - phases[None, :]           # (n, n)
        coherence  = np.cos(phase_diff) ** 2                      # (n, n)
        edge_coherence = coherence * adj                          # mask to edges

        # Build up where edges are active; decay everywhere
        self.Ca = (self.Ca + self.buildup * edge_coherence * dt) * self.decay
        self.Ca = np.clip(self.Ca, 0.0, 1.0)

        # Accumulate integral for learning signal
        self.Ca_integral += self.Ca

    def step_geometric(
        self,
        spins: np.ndarray,
        adj: np.ndarray,
        stress_fn,          # callable(spin_u, spin_v) -> float
        dt: float = 0.05,
    ) -> None:
        """
        Alternative: build calcium from geometric stress rather than phase coherence.

        Used when running Deerskin-only mode (no oscillators).
        Geometric stress on an edge → calcium accumulates there.
        """
        n = self.n
        stress_matrix = np.zeros((n, n))
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    s = stress_fn(int(spins[u]), int(spins[v]))
                    stress_matrix[u, v] = s
                    stress_matrix[v, u] = s

        self.Ca = (self.Ca + self.buildup * stress_matrix * dt) * self.decay
        self.Ca = np.clip(self.Ca, 0.0, 1.0)
        self.Ca_integral += self.Ca

    # ------------------------------------------------------------------
    @property
    def node_struggle(self) -> np.ndarray:
        """Per-node struggle score: mean calcium across all edges."""
        return self.Ca.mean(axis=1)

    @property
    def normalised_integral(self) -> np.ndarray:
        """Normalised struggle map — learning target for the GNN."""
        mx = self.Ca_integral.max()
        return self.Ca_integral / mx if mx > 0 else self.Ca_integral

    def attention_weight(self, u: int, v: int) -> float:
        """
        Calcium attention: how much the solver should focus on edge (u, v).
        Exceeds threshold → W-matrix learning accelerated there.
        """
        return float(self.Ca[u, v])

    def potentiation_mask(self) -> np.ndarray:
        """
        Boolean mask: edges where calcium exceeds threshold.
        These edges should have their W-matrix connections strengthened.
        """
        return self.Ca > self.threshold

    def reset(self) -> None:
        self.Ca[:] = 0.0
        self.Ca_integral[:] = 0.0
