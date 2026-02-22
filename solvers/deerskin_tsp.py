"""
solvers/deerskin_tsp.py  —  Deerskin-guided TSP Solver
=======================================================

Extends the fractal_instinct.py work into the Deerskin framework.

The connection to feynmantsp2.py
---------------------------------
fractal_instinct.py trained a GNN to predict swap_map (where 2-opt
struggled) then used it to bias nearest-neighbour initialisation.

Result: identical tour lengths (692.97) — guidance didn't help.

Why? The GNN's "instinct scores" were fed as a soft distance modifier
but had no structural relationship to tour geometry.

Deerskin-TSP fixes this by:
  1. Encoding each city as a geometric orientation (angle on manifold)
  2. Edge stress = moiré clash between adjacent city orientations
  3. Calcium accumulates on edges where 2-opt swaps happen (struggle trace)
  4. The Prior scars toward the low-stress tour orientation
  5. Greedy construction follows Prior orientation rather than Euclidean distance

The geometric Prior now has a *structural* relationship to the problem:
high-stress edges in orientation-space correspond to edges that 2-opt
wants to remove.  The Prior learns to avoid them.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.geometry import GeometricPrior, GraphGeometryEncoder
from core.calcium import CalciumMemory
from scipy.spatial.distance import cdist
from typing import List, Tuple


class DeerskinTSP:
    """
    Deerskin-guided TSP solver.

    Architecture
    ------------
    Each city i maps to an orientation θ_i on the geometric manifold.
    The ordering of cities in the tour → a sequence of orientations.
    A good tour = one where adjacent orientations are in low moiré stress.

    The solver:
      1. Runs standard 2-opt, recording calcium (swap trace)
      2. Builds orientation-guided tour using calcium + Prior stress
      3. Melts Prior toward the stress-minimising orientation
      4. Iterates: calcium informs next Prior, Prior informs next construction
    """

    def __init__(
        self,
        coords: np.ndarray,
        prior_freq: float = 2.0,
        melt_rate: float = 0.01,
        n_rounds: int = 3,
    ):
        self.coords  = np.asarray(coords, dtype=float)
        self.n       = len(coords)
        self.dist    = cdist(self.coords, self.coords)
        self.prior   = GeometricPrior(size=32, freq=prior_freq)
        self.calcium = CalciumMemory(self.n, decay=0.9, buildup=1.0)
        self.melt_rate = melt_rate
        self.n_rounds  = n_rounds

        # Map city indices to orientations: evenly spread on [0, π/2]
        # (both spin orientations used in Deerskin geometry)
        self.city_theta = np.linspace(0, np.pi / 2, self.n)

    # ------------------------------------------------------------------
    def _tour_length(self, path: List[int]) -> float:
        return sum(self.dist[path[i], path[(i+1) % self.n]] for i in range(self.n))

    def _nearest_neighbour(self, score_bias: np.ndarray, alpha: float = 0.6) -> List[int]:
        """
        Biased NN tour: guided_dist = euclidean * (1 - alpha * node_score)

        score_bias: (n,) — high score → prefer this city (low geometric stress)
        """
        edge_bias = 1.0 - alpha * (
            (score_bias[:, None] + score_bias[None, :]) / 2.0
        )
        guided = self.dist * np.clip(edge_bias, 0.05, 2.0)
        np.fill_diagonal(guided, np.inf)

        visited = [False] * self.n
        path = [0]; visited[0] = True; cur = 0
        for _ in range(self.n - 1):
            choices = [j for j in range(self.n) if not visited[j]]
            nxt = min(choices, key=lambda j: guided[cur, j])
            path.append(nxt); visited[nxt] = True; cur = nxt
        return path

    def _two_opt_with_calcium(self, path: List[int], max_iter: int = 500
                               ) -> Tuple[List[int], float, np.ndarray]:
        """
        2-opt with calcium accumulation on swapped edges.
        Returns (improved_path, length, swap_matrix).
        """
        n = self.n
        swap_map = np.zeros((n, n))
        best = path[:]
        best_len = self._tour_length(best)
        improved = True
        it = 0

        while improved and it < max_iter:
            improved = False; it += 1
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1: continue
                    new_path = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_len  = self._tour_length(new_path)
                    if new_len < best_len - 1e-10:
                        # Record calcium on the edges we're removing
                        u, v = best[i-1], best[i]
                        swap_map[u, v] += 1; swap_map[v, u] += 1
                        u, v = best[j], best[(j+1) % n]
                        swap_map[u, v] += 1; swap_map[v, u] += 1
                        best = new_path; best_len = new_len
                        improved = True; break
                if improved: break

        if swap_map.max() > 0:
            swap_map /= swap_map.max()
        return best, best_len, swap_map

    def _update_prior_from_tour(self, path: List[int], swap_map: np.ndarray) -> None:
        """
        Viscous melt: Prior scars toward the orientation of low-stress tour edges.

        Edges with high swap_map = where 2-opt struggled = high stress in orientation space.
        The Prior melts *away* from those orientations.
        """
        # Compute weighted mean orientation of tour edges (weighted by low swap = good edges)
        total_w = 0.0; weighted_theta = 0.0
        for i in range(self.n):
            u = path[i]; v = path[(i+1) % self.n]
            weight = 1.0 - swap_map[u, v]   # good edges (low swap) get high weight
            edge_theta = (self.city_theta[u] + self.city_theta[v]) / 2.0
            weighted_theta += weight * edge_theta
            total_w += weight

        if total_w > 0:
            target_theta = weighted_theta / total_w
            self.prior.viscous_melt(target_theta, melt_rate=self.melt_rate)

    def _prior_node_scores(self, swap_map: np.ndarray) -> np.ndarray:
        """
        Node scores for guided construction.

        Low stress relative to current Prior → prefer this node.
        High calcium (chronic struggle) → node is a bottleneck → route carefully.
        """
        scores = np.zeros(self.n)
        for i in range(self.n):
            # Geometric stress of this node's orientation
            s = self.prior.stress(self.city_theta[i])
            # Calcium load (struggle history)
            ca = float(swap_map[i].mean())
            # Combined: low stress + high calcium = "interesting bottleneck"
            scores[i] = (1.0 - s) * (0.5 + 0.5 * ca)

        if scores.max() > 0:
            scores /= scores.max()
        return scores

    # ------------------------------------------------------------------
    def solve(self) -> Tuple[List[int], float, dict]:
        """
        Multi-round Deerskin TSP.

        Round 1: plain 2-opt → build swap_map (calcium)
        Round 2+: Prior-guided construction → 2-opt → update Prior
        """
        # Round 0: plain nearest-neighbour + 2-opt baseline
        plain_nn   = self._nearest_neighbour(np.zeros(self.n), alpha=0.0)
        best_path, best_len, swap_map = self._two_opt_with_calcium(plain_nn)

        all_lengths = [best_len]

        for rnd in range(self.n_rounds):
            # Update Prior from last round's struggle
            self._update_prior_from_tour(best_path, swap_map)

            # Compute biased node scores
            node_scores = self._prior_node_scores(swap_map)

            # Guided construction
            guided_path = self._nearest_neighbour(node_scores, alpha=0.7)

            # 2-opt on guided path
            guided_path, guided_len, new_swap = self._two_opt_with_calcium(guided_path)

            # Accumulate calcium
            swap_map = 0.5 * swap_map + 0.5 * new_swap   # blend old + new

            all_lengths.append(guided_len)

            if guided_len < best_len:
                best_len = guided_len
                best_path = guided_path

        return best_path, best_len, {
            "round_lengths": all_lengths,
            "prior_theta":   list(self.prior.melt_history),
            "swap_map":      swap_map,
        }


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def compare_tsp(n_cities: int = 51, n_tests: int = 20, seed_start: int = 0):
    from scipy.spatial.distance import cdist

    baseline_wins = 0; deerskin_wins = 0; ties = 0
    improvements = []

    print(f"\n{'='*55}")
    print(f"  TSP Comparison  ({n_tests} instances, {n_cities} cities each)")
    print(f"{'='*55}")

    for seed in range(seed_start, seed_start + n_tests):
        rng = np.random.RandomState(seed)
        coords = rng.rand(n_cities, 2) * 100.0
        dist   = cdist(coords, coords)

        # Baseline: NN + plain 2-opt
        n = n_cities; visited = [False]*n; path = [0]; visited[0] = True; cur = 0
        for _ in range(n-1):
            choices = [j for j in range(n) if not visited[j]]
            nxt = min(choices, key=lambda j: dist[cur,j])
            path.append(nxt); visited[nxt] = True; cur = nxt
        # 2-opt
        best = path[:]; best_len = sum(dist[best[i],best[(i+1)%n]] for i in range(n))
        improved = True
        while improved:
            improved = False
            for i in range(1,n-2):
                for j in range(i+1,n):
                    if j-i==1: continue
                    np2 = best[:i]+best[i:j+1][::-1]+best[j+1:]
                    nl  = sum(dist[np2[k],np2[(k+1)%n]] for k in range(n))
                    if nl < best_len - 1e-10:
                        best=np2; best_len=nl; improved=True; break
                if improved: break
        baseline_len = best_len

        # Deerskin-TSP
        solver = DeerskinTSP(coords, n_rounds=3, melt_rate=0.015)
        _, dk_len, _ = solver.solve()

        delta = (baseline_len - dk_len) / baseline_len * 100
        if delta > 0.1:  deerskin_wins += 1; improvements.append(delta)
        elif delta < -0.1: baseline_wins += 1
        else: ties += 1

    print(f"  Deerskin wins: {deerskin_wins}/{n_tests}  "
          f"(avg improvement {np.mean(improvements):.2f}% when winning)")
    print(f"  Baseline wins: {baseline_wins}/{n_tests}")
    print(f"  Ties:          {ties}/{n_tests}")

    return deerskin_wins, baseline_wins, ties, improvements


if __name__ == "__main__":
    compare_tsp(n_cities=51, n_tests=20)
