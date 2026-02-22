"""
benchmarks/maxcut_benchmark.py
================================
Head-to-head MaxCut benchmark reproducing and extending the q4.png result.

Solvers compared:
  1. Classical SA            (gold standard)
  2. Original P-KAS          (the broken version — reproduces q4.png collapse)
  3. Quantum-Inspired        (tanh relaxation)
  4. Deerskin-PKAS           (our contribution)

Output:
  - Console summary table
  - benchmarks/results_maxcut.png  (convergence plot + graph visualisation)
  - benchmarks/results_multi.png   (multi-instance statistics)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

from solvers.baselines import SimulatedAnnealing, OriginalPKAS, QuantumInspired
from solvers.deerskin_pkas import DeerskinPKAS, SolverConfig


# ---------------------------------------------------------------------------
# Graph factories
# ---------------------------------------------------------------------------

def make_regular_graph(n: int = 50, d: int = 3, seed: int = 7) -> np.ndarray:
    G = nx.random_regular_graph(d, n, seed=seed)
    return nx.to_numpy_array(G), G

def make_erdos_renyi(n: int = 50, p: float = 0.15, seed: int = 7) -> np.ndarray:
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return nx.to_numpy_array(G), G

def make_planted_partition(k: int = 5, size: int = 10, p_in: float = 0.1,
                           p_out: float = 0.5, seed: int = 7):
    """Planted partition — has known near-optimal cut structure."""
    G = nx.planted_partition_graph(k, size, p_in, p_out, seed=seed)
    return nx.to_numpy_array(G), G


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(adj: np.ndarray, G: nx.Graph, label: str, max_iter: int = 5000):
    n = adj.shape[0]
    n_edges = int(adj.sum()) // 2
    print(f"\n{'='*65}")
    print(f"  {label}  |  {n} nodes, {n_edges} edges")
    print(f"{'='*65}")

    results = {}

    # 1. Classical SA
    t0 = time.time()
    sa = SimulatedAnnealing(adj, max_iter=max_iter)
    spins_sa, cut_sa, hist_sa = sa.solve()
    t_sa = time.time() - t0
    results["Classical SA"] = dict(cut=cut_sa, time=t_sa, history=hist_sa, spins=spins_sa)
    print(f"  Classical SA:      {cut_sa}/{n_edges} ({100*cut_sa/n_edges:.1f}%)  in {t_sa:.2f}s")

    # 2. Original P-KAS
    t0 = time.time()
    pkas = OriginalPKAS(adj, max_iter=max_iter)
    spins_pk, cut_pk, hist_pk = pkas.solve()
    t_pk = time.time() - t0
    results["Original P-KAS"] = dict(cut=cut_pk, time=t_pk, history=hist_pk, spins=spins_pk)
    print(f"  Original P-KAS:    {cut_pk}/{n_edges} ({100*cut_pk/n_edges:.1f}%)  in {t_pk:.2f}s")

    # 3. Quantum-Inspired
    t0 = time.time()
    qi = QuantumInspired(adj, max_iter=max_iter)
    spins_qi, cut_qi, hist_qi = qi.solve()
    t_qi = time.time() - t0
    results["Quantum-Inspired"] = dict(cut=cut_qi, time=t_qi, history=hist_qi, spins=spins_qi)
    print(f"  Quantum-Inspired:  {cut_qi}/{n_edges} ({100*cut_qi/n_edges:.1f}%)  in {t_qi:.2f}s")

    # 4. Deerskin-PKAS
    t0 = time.time()
    cfg = SolverConfig(max_iter=max_iter)
    dk = DeerskinPKAS(adj, cfg)
    spins_dk, cut_dk, hist_dk = dk.solve()
    t_dk = time.time() - t0
    results["Deerskin-PKAS"] = dict(
        cut=cut_dk, time=t_dk, history=hist_dk["cut"], spins=spins_dk,
        stress=hist_dk["stress"], melt=hist_dk["melt"]
    )
    print(f"  Deerskin-PKAS:     {cut_dk}/{n_edges} ({100*cut_dk/n_edges:.1f}%)  in {t_dk:.2f}s")

    # Winner
    best_name = max(results, key=lambda k: results[k]["cut"])
    print(f"\n  🏆  {best_name}")

    return results, n_edges


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Classical SA":     "#4C72B0",
    "Original P-KAS":   "#DD8452",
    "Quantum-Inspired": "#55A868",
    "Deerskin-PKAS":    "#C44E52",
}

def plot_benchmark(results: dict, n_edges: int, G: nx.Graph, save_path: str):
    fig = plt.figure(figsize=(20, 14), facecolor="#1a1a2e")

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45, wspace=0.35,
        height_ratios=[2.5, 1, 1.5]
    )

    ax_main  = fig.add_subplot(gs[0, :])   # convergence
    ax_s     = fig.add_subplot(gs[1, :2])  # stress
    ax_m     = fig.add_subplot(gs[1, 2:])  # melt rate
    ax_sa    = fig.add_subplot(gs[2, 0])
    ax_pk    = fig.add_subplot(gs[2, 1])
    ax_qi    = fig.add_subplot(gs[2, 2])
    ax_dk    = fig.add_subplot(gs[2, 3])

    text_kw  = dict(color="#e0e0e0")
    spine_c  = "#444"

    def style_ax(ax, title=""):
        ax.set_facecolor("#111122")
        for sp in ax.spines.values(): sp.set_edgecolor(spine_c)
        ax.tick_params(colors="#aaa", labelsize=8)
        if title: ax.set_title(title, color="#e0e0e0", fontsize=10, pad=6)

    # --- Convergence ---
    style_ax(ax_main)
    ax_main.set_title("Convergence: MaxCut Cut Value vs Iteration",
                       color="#00ffdd", fontsize=13, pad=8)
    ax_main.axhline(n_edges, color="#ff6b6b", lw=1.5, ls="--",
                    label=f"Theoretical max ({n_edges})", alpha=0.7)

    for name, res in results.items():
        h = res["history"]
        x = np.linspace(0, len(h), len(h))
        ax_main.plot(x, h, color=COLORS[name], lw=1.5,
                     label=f"{name}  [{res['cut']}/{n_edges}  {100*res['cut']/n_edges:.1f}%]",
                     alpha=0.85)

    ax_main.set_xlabel("Iteration", **text_kw)
    ax_main.set_ylabel("Cut Value", **text_kw)
    leg = ax_main.legend(loc="lower right", fontsize=9,
                          facecolor="#222233", edgecolor="#444")
    for t in leg.get_texts(): t.set_color("#e0e0e0")

    # --- Stress & melt (Deerskin internals) ---
    style_ax(ax_s, "Deerskin: Geometric Stress over Time")
    dk_stress = results["Deerskin-PKAS"].get("stress", [])
    if dk_stress:
        ax_s.plot(dk_stress, color="#e879f9", lw=1, alpha=0.8)
        ax_s.set_ylabel("Free Energy", color="#aaa", fontsize=8)
        ax_s.set_xlabel("Iteration", color="#aaa", fontsize=8)

    style_ax(ax_m, "Deerskin: Viscosity (Melt Rate)")
    dk_melt = results["Deerskin-PKAS"].get("melt", [])
    if dk_melt:
        ax_m.plot(dk_melt, color="#34d399", lw=1.5, alpha=0.9)
        ax_m.set_ylabel("Melt Rate", color="#aaa", fontsize=8)
        ax_m.set_xlabel("Iteration", color="#aaa", fontsize=8)

    # --- Graph panels ---
    pos = nx.spring_layout(G, seed=42)

    def draw_graph(ax, spins, title, cut, total):
        style_ax(ax)
        colors = ["#ff6b6b" if s == 1 else "#4ecdc4" for s in spins]
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=40)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#666", width=0.5, alpha=0.5)
        ax.set_title(f"{title}\n{cut}/{total} ({100*cut/total:.1f}%)",
                     color="#e0e0e0", fontsize=8)
        ax.axis("off")

    for ax, (name, res) in zip(
        [ax_sa, ax_pk, ax_qi, ax_dk],
        results.items()
    ):
        draw_graph(ax, res["spins"], name, res["cut"], n_edges)

    plt.suptitle(
        "Deerskin-PKAS  vs  Classical Methods\nGeometric Prior replaces Temperature Schedule",
        color="#00ffdd", fontsize=15, y=0.98
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Multi-instance statistics
# ---------------------------------------------------------------------------

def run_multi_instance(n_tests: int = 20, n_nodes: int = 50, max_iter: int = 3000):
    """Run on multiple random graphs and report aggregate statistics."""
    print(f"\n{'='*65}")
    print(f"  Multi-instance benchmark  ({n_tests} graphs, {n_nodes} nodes each)")
    print(f"{'='*65}")

    records = {k: [] for k in ["Classical SA", "Original P-KAS", "Quantum-Inspired", "Deerskin-PKAS"]}

    for seed in range(n_tests):
        G = nx.random_regular_graph(3, n_nodes, seed=seed)
        adj = nx.to_numpy_array(G)
        n_edges = int(adj.sum()) // 2

        sa_spins, cut_sa, _ = SimulatedAnnealing(adj, max_iter=max_iter).solve()
        pk_spins, cut_pk, _ = OriginalPKAS(adj, max_iter=max_iter).solve()
        qi_spins, cut_qi, _ = QuantumInspired(adj, max_iter=max_iter).solve()
        cfg = SolverConfig(max_iter=max_iter)
        dk_spins, cut_dk, _ = DeerskinPKAS(adj, cfg).solve()

        for name, cut in [
            ("Classical SA", cut_sa),
            ("Original P-KAS", cut_pk),
            ("Quantum-Inspired", cut_qi),
            ("Deerskin-PKAS", cut_dk),
        ]:
            records[name].append(100 * cut / n_edges)

        if (seed + 1) % 5 == 0:
            print(f"  Completed {seed+1}/{n_tests}...")

    print(f"\n  {'Solver':<22} {'Mean %':>8}  {'Std':>6}  {'Min':>6}  {'Max':>6}")
    print(f"  {'-'*55}")
    for name, vals in records.items():
        arr = np.array(vals)
        print(f"  {name:<22} {arr.mean():>7.1f}%  {arr.std():>5.1f}  {arr.min():>5.1f}  {arr.max():>5.1f}")

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",    type=int, default=50)
    parser.add_argument("--iter",     type=int, default=5000)
    parser.add_argument("--multi",    action="store_true")
    parser.add_argument("--n-tests",  type=int, default=20)
    parser.add_argument("--graph",    choices=["regular", "erdos", "planted"],
                        default="regular")
    args = parser.parse_args()

    if args.graph == "regular":
        adj, G = make_regular_graph(args.nodes, seed=7)
        label  = f"3-Regular Graph"
    elif args.graph == "erdos":
        adj, G = make_erdos_renyi(args.nodes, seed=7)
        label  = f"Erdős–Rényi Graph"
    else:
        adj, G = make_planted_partition(seed=7)
        label  = f"Planted Partition Graph"

    results, n_edges = run_benchmark(adj, G, label, max_iter=args.iter)
    plot_benchmark(results, n_edges, G, "benchmarks/results_maxcut.png")

    if args.multi:
        records = run_multi_instance(n_tests=args.n_tests, max_iter=args.iter // 2)
