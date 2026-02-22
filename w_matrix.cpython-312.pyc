"""
core/geometry.py  —  Deerskin Geometric Field Engine
=====================================================

The Prior:  a rigid 2D cosine lattice — the system's unyielding expectation.
The Eye:    the sensory field twisted by current problem state.
Moiré:      their interference — Free Energy / geometric stress.

This replaces temperature schedules.
Instead of cooling, the membrane scars (viscous melt).

Theory
------
In gauge theory terms: force arises when you try to move straight
while the field forces a twist.  The brain (or solver) carries a
6D gyroscope.  Mismatch between Prior and world creates stress that
flows out as action.  Learning = the tissue physically rotating under
chronic stress until it can no longer be deformed.

Here that manifests as:
  - spin assignments  →  eye orientations
  - edge violations   →  moiré clash between adjacent orientations
  - solution          →  the configuration where global stress is minimised
  - annealing temp    →  REPLACED by viscosity (melt_rate)
  - temperature decay →  REPLACED by stress-driven scarring
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Geometric lattice
# ---------------------------------------------------------------------------

@dataclass
class GeometricPrior:
    """
    2D cosine lattice carrying the system's rigid internal expectation.

    Parameters
    ----------
    size  : grid resolution for stress evaluation
    freq  : spatial frequency of the lattice rings
    theta : orientation of the Prior (radians)
    x, y  : translational offsets
    """
    size:  int   = 32
    freq:  float = 2.0
    theta: float = 0.0
    x:     float = 0.0
    y:     float = 0.0

    # viscosity history for plotting
    melt_history: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        g = np.linspace(-1, 1, self.size)
        self._xg, self._yg = np.meshgrid(g, g)

    # -----------------------------------------------------------------
    def _lattice(self, theta: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
        rx = (self._xg - x) * np.cos(theta) - (self._yg - y) * np.sin(theta)
        ry = (self._xg - x) * np.sin(theta) + (self._yg - y) * np.cos(theta)
        return np.cos(rx * self.freq * np.pi) * np.cos(ry * self.freq * np.pi)

    def anchor(self) -> np.ndarray:
        """The Prior lattice — what this system expects."""
        return self._lattice(self.theta, self.x, self.y)

    def eye(self, eye_theta: float, eye_x: float = 0.0, eye_y: float = 0.0) -> np.ndarray:
        """The Eye lattice — twisted by the current world state."""
        return self._lattice(eye_theta, eye_x, eye_y)

    def moire(self, eye_theta: float, eye_x: float = 0.0, eye_y: float = 0.0) -> np.ndarray:
        """Physical moiré interference field (Anchor × Eye)."""
        return self.anchor() * self.eye(eye_theta, eye_x, eye_y)

    def stress(self, eye_theta: float, eye_x: float = 0.0, eye_y: float = 0.0) -> float:
        """
        Scalar Free Energy in [0, 1].

        0 = perfect harmony.  1 = maximum clash.
        Replaces temperature in annealing.
        """
        harmony = float(np.mean(self.moire(eye_theta, eye_x, eye_y)))
        # Baseline harmony for perfectly aligned fields ≈ 0.25
        return float(np.clip(1.0 - harmony / 0.25, 0.0, 1.0))

    def stress_gradient(self, eye_theta: float, eps: float = 0.02) -> float:
        """dStress/dθ  — the gauge force / torque resisting misalignment."""
        return (self.stress(eye_theta + eps) - self.stress(eye_theta - eps)) / (2.0 * eps)

    def viscous_melt(self, eye_theta: float, melt_rate: float = 0.003) -> None:
        """
        Epigenetic scarring: Prior slowly rotates toward the Eye.

        Chronic stress → tissue becomes plastic → geometry rotates.
        Learning without backpropagation: membrane physically realigns
        to minimise long-run free energy.

        This is what distinguishes Deerskin from SA:
          SA lowers temperature externally.
          Deerskin lowers plasticity internally, driven by stress itself.
        """
        self.theta += (eye_theta - self.theta) * melt_rate
        self.melt_history.append(float(self.theta))


# ---------------------------------------------------------------------------
# Graph → geometry bridge
# ---------------------------------------------------------------------------

class GraphGeometryEncoder:
    """
    Maps graph-optimisation problems onto the geometric field.

    Nodes    →  angular positions on the manifold.
    Spins    →  orientation offsets (aligned vs orthogonal to Prior).
    Edges    →  stress between adjacent orientations.
    Solution →  spin assignment minimising total geometric stress.
    """

    SPIN_THETA = {+1: 0.0, -1: np.pi / 2.0}   # aligned / orthogonal

    def __init__(self, prior: GeometricPrior):
        self.prior = prior

    # -----------------------------------------------------------------
    def spin_to_theta(self, spin: int) -> float:
        return self.SPIN_THETA[int(spin)]

    def edge_stress(self, spin_u: int, spin_v: int) -> float:
        """
        Geometric stress for a single edge.

        Same partition  → both orthogonal  → low stress  → BAD for MaxCut
        Cross partition → opposite orientations → medium stress → GOOD for MaxCut

        The solver is rewarded for creating maximum stress contrast across cuts.
        """
        θu = self.spin_to_theta(spin_u)
        θv = self.spin_to_theta(spin_v)
        # We want same-spin edges to be costly (they don't contribute cuts)
        same_spin = int(spin_u) == int(spin_v)
        return self.prior.stress((θu + θv) / 2.0) * (1.0 if same_spin else 0.3)

    def total_stress(self, spins: np.ndarray, adj: np.ndarray) -> float:
        """Total geometric Free Energy of the graph. Minimise to maximise cut."""
        n = len(spins)
        total = 0.0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    total += self.edge_stress(int(spins[u]), int(spins[v]))
        return total

    @staticmethod
    def cut_value(spins: np.ndarray, adj: np.ndarray) -> int:
        """Standard MaxCut objective."""
        n = len(spins)
        return int(np.sum([
            adj[u, v]
            for u in range(n)
            for v in range(u + 1, n)
            if adj[u, v] and spins[u] != spins[v]
        ]))
