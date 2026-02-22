"""
solvers/deerskin_pkas.py  —  Deerskin-PKAS MaxCut Solver
=========================================================

The Fusion
----------
P-KAS brought:
  ✓ Kuramoto-like phase oscillators per node
  ✓ Calcium memory tracking struggle hotspots
  ✓ W-matrix slow learning the coupling landscape

P-KAS failed because:
  ✗ Phase collapse at ~1000 iterations (no recovery mechanism)
  ✗ Temperature was still externally scheduled (arbitrary)
  ✗ No geometric prior — explored symmetrically, found nothing

Deerskin adds:
  ✓ Geometric Prior replacing temperature schedule
  ✓ Viscous melt: plasticity decreases when stress is chronic
  ✓ Moiré stress as the annealing drive (not random noise)
  ✓ Anti-collapse: when geometric stress spikes, melt temporarily
    increases to let the Prior reorient — then stiffens again

The key insight
---------------
SA asks: "what temperature makes random flips useful?"
Deerskin asks: "what orientation makes this configuration stable?"

The solver finds the spin assignment that minimises total geometric
Free Energy — the configuration where every edge is in a low-stress
orientation relative to the Prior.  The Prior scars toward the
solution over time.  No temperature. No arbitrary schedule.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.geometry import GeometricPrior, GraphGeometryEncoder
from core.calcium import CalciumMemory
from core.w_matrix import WMatrix


@dataclass
class SolverConfig:
    """All hyperparameters in one place."""
    # Phase dynamics
    dt:             float = 0.05
    max_iter:       int   = 5000
    field_strength: float = 0.5    # external geometric field drive

    # Calcium
    ca_decay:       float = 0.95
    ca_buildup:     float = 1.0
    ca_threshold:   float = 0.3

    # W-matrix
    w_lr:           float = 0.001
    w_decay:        float = 0.9999
    w_update_every: int   = 10

    # Geometric Prior (replaces SA temperature)
    prior_size:     int   = 32
    prior_freq:     float = 2.0
    melt_rate:      float = 0.003   # initial viscosity
    melt_min:       float = 0.0005  # minimum viscosity (fully scarred)
    melt_decay:     float = 0.9995  # viscosity decreases as solver matures

    # Anti-collapse (Deerskin's fix for P-KAS phase collapse)
    collapse_detect:  bool  = True
    collapse_window:  int   = 50    # steps to check for collapse
    collapse_thresh:  float = 0.15  # cut fraction drop triggers rescue
    rescue_melt_boost: float = 5.0  # temporary melt rate multiplier


class DeerskinPKAS:
    """
    Deerskin-PKAS solver for MaxCut (and extensible to other QUBO problems).

    Architecture
    ------------
    Each node i carries:
      - phase[i]   : fast oscillator (updated every step)
      - spin[i]    : binarised phase (sign of cos(phase))
      - calcium[i,j]: slow struggle trace on each edge

    The Prior geometry:
      - Monitors total geometric stress at each step
      - Exerts a field force on phases (geometric drive)
      - Scars (melts) when stress is chronic
      - Anti-collapses by temporarily increasing plasticity

    No temperature. No schedule. The solver learns its own
    annealing curve from the geometry of the problem.
    """

    def __init__(self, adj: np.ndarray, config: Optional[SolverConfig] = None):
        self.adj = np.asarray(adj, dtype=float)
        self.n = adj.shape[0]
        self.cfg = config or SolverConfig()

        # --- Core components ---
        self.prior = GeometricPrior(
            size=self.cfg.prior_size,
            freq=self.cfg.prior_freq,
        )
        self.encoder = GraphGeometryEncoder(self.prior)
        self.calcium = CalciumMemory(
            self.n,
            decay=self.cfg.ca_decay,
            buildup=self.cfg.ca_buildup,
            threshold=self.cfg.ca_threshold,
        )
        self.W = WMatrix(
            self.n,
            lr=self.cfg.w_lr,
            decay=self.cfg.w_decay,
        )

        # --- State ---
        rng = np.random.default_rng(42)
        self.phases = rng.uniform(0, 2 * np.pi, self.n)
        self.melt_rate = self.cfg.melt_rate

        # --- History for analysis ---
        self.cut_history:    List[int]   = []
        self.stress_history: List[float] = []
        self.melt_history:   List[float] = []
        self.best_spins: Optional[np.ndarray] = None
        self.best_cut:   int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def spins(self) -> np.ndarray:
        """Binarise phases: cos(phase) > 0 → +1, else -1."""
        return np.where(np.cos(self.phases) >= 0, 1, -1)

    def _phase_force(self) -> np.ndarray:
        """
        Combined phase force:
          1. Kuramoto anti-sync (W-matrix coupling)
          2. Geometric field force (Deerskin moiré gradient)
        """
        # 1. W-matrix Kuramoto coupling
        kuramoto = self.W.coupling_force(self.phases)

        # 2. Geometric field force
        #    Each node's phase is nudged by the gradient of geometric stress
        #    w.r.t. the global eye orientation
        cur_eye_theta = np.mean(self.phases) % (2 * np.pi)
        geo_gradient  = self.prior.stress_gradient(cur_eye_theta)
        geo_force     = -self.cfg.field_strength * geo_gradient * np.ones(self.n)

        # 3. Per-node noise scaled by node calcium (more noise where struggling)
        node_struggle = self.calcium.node_struggle
        noise = np.random.randn(self.n) * (0.1 + 0.3 * node_struggle)

        return kuramoto + geo_force + noise

    def _update_prior(self) -> float:
        """Compute stress and apply viscous melt. Returns current stress."""
        spins = self.spins
        # Eye orientation = mean of current spin-derived thetas
        eye_theta = float(np.mean([
            self.encoder.spin_to_theta(int(s)) for s in spins
        ]))
        stress = self.prior.stress(eye_theta)
        self.prior.viscous_melt(eye_theta, melt_rate=self.melt_rate)
        return stress

    def _decay_viscosity(self) -> None:
        """Gradually stiffen the membrane — less plastic over time."""
        self.melt_rate = max(
            self.cfg.melt_min,
            self.melt_rate * self.cfg.melt_decay
        )

    def _detect_and_rescue_collapse(self) -> bool:
        """
        P-KAS's fatal flaw was phase collapse around iteration 1000.
        Detect it: if cut value dropped sharply, temporarily increase melt.
        This allows the Prior to reorient and escape the attractor.
        """
        w = self.cfg.collapse_window
        if len(self.cut_history) < w * 2:
            return False
        recent  = np.mean(self.cut_history[-w:])
        earlier = np.mean(self.cut_history[-w*2:-w])
        n_edges = int(self.adj.sum()) // 2
        if earlier > 0 and (earlier - recent) / n_edges > self.cfg.collapse_thresh:
            # Collapse detected — boost melt temporarily
            self.melt_rate = min(
                self.cfg.melt_rate * self.cfg.rescue_melt_boost,
                0.05
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Main solve loop
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[np.ndarray, int, dict]:
        """
        Run Deerskin-PKAS optimisation.

        Returns
        -------
        best_spins   : (n,) array of +1/-1
        best_cut     : integer cut value
        history      : dict with convergence curves for analysis
        """
        cfg = self.cfg

        for step in range(cfg.max_iter):
            # --- 1. Phase dynamics ---
            force        = self._phase_force()
            self.phases += cfg.dt * force
            # Keep phases in [0, 2π]
            self.phases  = self.phases % (2 * np.pi)

            # --- 2. Calcium update ---
            self.calcium.step(self.phases, self.adj, dt=cfg.dt)

            # --- 3. W-matrix update (every N steps) ---
            if step % cfg.w_update_every == 0:
                self.W.hebbian_update(self.calcium, self.adj)
                self.W.geometric_update(self.spins, self.adj, self.encoder)

            # --- 4. Geometric Prior update (Deerskin core) ---
            stress = self._update_prior()
            self._decay_viscosity()

            # --- 5. Track state ---
            cur_spins = self.spins
            cur_cut   = self.encoder.cut_value(cur_spins, self.adj)

            self.cut_history.append(cur_cut)
            self.stress_history.append(stress)
            self.melt_history.append(self.melt_rate)

            if cur_cut > self.best_cut:
                self.best_cut  = cur_cut
                self.best_spins = cur_spins.copy()

            # --- 6. Anti-collapse rescue ---
            if cfg.collapse_detect and step % 50 == 0:
                self._detect_and_rescue_collapse()

        return (
            self.best_spins,
            self.best_cut,
            {
                "cut":    self.cut_history,
                "stress": self.stress_history,
                "melt":   self.melt_history,
                "prior_theta": self.prior.melt_history,
            }
        )


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import networkx as nx

    print("Deerskin-PKAS quick test — 50-node 3-regular MaxCut")
    G = nx.random_regular_graph(3, 50, seed=7)
    adj = nx.to_numpy_array(G)
    n_edges = int(adj.sum()) // 2
    print(f"Edges: {n_edges}  |  Theoretical max ≈ {n_edges}")

    solver = DeerskinPKAS(adj)
    spins, cut, history = solver.solve()

    print(f"Best cut: {cut}/{n_edges} ({100*cut/n_edges:.1f}%)")
    print(f"Final melt rate: {solver.melt_rate:.6f}")
    print(f"Final stress: {history['stress'][-1]:.4f}")
