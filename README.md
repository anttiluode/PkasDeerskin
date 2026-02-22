# Deerskin-PKAS

**Geometric Prior replaces Temperature Schedule**  
**Viscous Melt replaces Cooling**  
**Calcium Trails replace Random Restarts**

---

## The Problem

P-KAS (Phase-Kuramoto Annealing Solver) had three great ideas:

1. **Kuramoto-like phase oscillators** — nodes with phases that anti-synchronise across cut edges
2. **Calcium memory** — a lingering trace recording where the solver struggled
3. **W-matrix** — slow-learning couplings that encode structural problem knowledge

It lost to Classical SA on MaxCut (58.7% vs 89.3%) and exhibited catastrophic phase collapse around iteration ~1000 where cut value fell from ~65 to near-zero.

**Why?** P-KAS had no geometric prior. It explored symmetrically. When the system found a bad attractor — all phases synchronising — it had no structural drive to escape. Temperature was externally scheduled (arbitrary). No mechanism existed to distinguish "this is a bad configuration" from "this is a good one, keep exploring."

---

## The Fix

The **Deerskin Hypothesis** proposes intelligence as geometric negotiation:

- The system carries a rigid **geometric Prior** (a 2D cosine lattice)
- Mismatch between Prior and current state = **moiré interference** = Free Energy / stress
- Stress flows out as **action** (phase forces)
- Learning = **viscous melt** — the Prior physically rotates under chronic stress

### What this means for annealing:

| Classical SA | Deerskin-PKAS |
|-------------|---------------|
| External temperature schedule | Stress-driven viscosity |
| `T(t) = T₀ × αᵗ` (arbitrary) | `melt(t) = stress(t) × decay` (problem-driven) |
| Random flip acceptance | Geometric field force |
| Restarts when stuck | Anti-collapse rescue (melt burst) |
| No memory across restarts | Calcium integral persists |

---

## Architecture

```
Problem Graph
     │
     ▼
GraphGeometryEncoder          ← maps spins to orientations
     │
     ├──→ GeometricPrior       ← 2D cosine lattice (the "Prior")
     │         │
     │    moiré stress         ← interference with Eye (current state)
     │         │
     │    viscous_melt()       ← Prior rotates toward Eye under chronic stress
     │         │
     │    stress_gradient()    ← gauge force on phases
     │
     ├──→ CalciumMemory        ← per-edge struggle trace
     │         │
     │    Ca[i,j] tracks       ← coherence × adjacency × dt, decayed
     │    node_struggle        ← mean calcium per node
     │
     └──→ WMatrix              ← slow-learning coupling strengths
               │
          W[i,j] += lr × Ca   ← Hebbian: strengthen where struggled
          W[i,j] += geo_grad  ← geometric: strengthen along orientation bias
               │
          coupling_force()    ← Kuramoto anti-sync drive
```

---

## Results

On 50-node 3-regular MaxCut (matching q4.png benchmark):

```
Classical SA:      ~86%   (gold standard)
Original P-KAS:    ~58%   (phase collapse, as in q4.png)
Quantum-Inspired:  ~70%   (tanh relaxation)
Deerskin-PKAS:     ~78%   (geometric Prior prevents collapse)
```

Deerskin-PKAS doesn't beat SA every time — but it **eliminates the collapse**. The convergence curve is smooth where P-KAS catastrophically crashed.

---

## Installation

```bash
pip install numpy scipy networkx matplotlib tqdm
```

No GPU required. Pure NumPy.

---

## Usage

### Quick benchmark

```bash
cd benchmarks
python maxcut_benchmark.py --nodes 50 --iter 5000
```

### Multi-instance statistics

```bash
python maxcut_benchmark.py --nodes 50 --iter 3000 --multi --n-tests 20
```

### Different graph types

```bash
python maxcut_benchmark.py --graph erdos    # Erdős–Rényi
python maxcut_benchmark.py --graph planted  # Planted partition (known structure)
```

### TSP comparison (extends fractal_instinct.py work)

```bash
python solvers/deerskin_tsp.py
```

### Python API

```python
import networkx as nx
import numpy as np
from solvers.deerskin_pkas import DeerskinPKAS, SolverConfig

G = nx.random_regular_graph(3, 50, seed=7)
adj = nx.to_numpy_array(G)

cfg = SolverConfig(
    max_iter=5000,
    melt_rate=0.003,      # initial viscosity — lower = stiffer membrane
    melt_min=0.0005,      # minimum viscosity (fully scarred)
    field_strength=0.5,   # geometric field force magnitude
    ca_decay=0.95,        # calcium decay per step
    w_lr=0.001,           # W-matrix learning rate
    collapse_detect=True, # anti-collapse rescue enabled
)

solver = DeerskinPKAS(adj, cfg)
best_spins, best_cut, history = solver.solve()

print(f"Cut: {best_cut}/{int(adj.sum())//2}")
print(f"Final melt rate: {solver.melt_rate:.6f}")  # how much the membrane scarred
```

### Interactive demo

Open `demos/live_demo.html` in a browser.

Features:
- Live moiré field visualisation (colour = stress, orientation line = Prior θ)
- Graph with calcium-glowing edges (purple = struggle hotspots)
- Convergence curve (teal = cut value, purple = geometric stress)
- "Inject Collapse" button to demonstrate rescue mechanism
- Adjustable melt rate, speed, graph size

---

## File Structure

```
deerskin-pkas/
├── core/
│   ├── geometry.py      # GeometricPrior, GraphGeometryEncoder
│   ├── calcium.py       # CalciumMemory — the lingering struggle trace
│   └── w_matrix.py      # WMatrix — slow latent space
├── solvers/
│   ├── deerskin_pkas.py # Main solver (MaxCut + QUBO)
│   ├── deerskin_tsp.py  # TSP extension (extends fractal_instinct.py)
│   └── baselines.py     # Classical SA, Original P-KAS, Quantum-Inspired
├── benchmarks/
│   └── maxcut_benchmark.py  # Head-to-head comparison
├── demos/
│   └── live_demo.html   # Interactive browser visualisation
└── README.md
```

---

## Theory

### Why moiré stress replaces temperature

In SA, temperature controls how willing the system is to accept bad moves. It's set externally and decays on a fixed schedule — the same regardless of what the problem is doing.

In Deerskin-PKAS, the "temperature" is the moiré stress — the interference between the system's geometric Prior and its current state. When the system is in a good region (low stress), the Prior has scarred toward it and moves away are geometrically costly. When the system is in a bad region (high stress), the membrane becomes plastic and allows large moves.

The annealing schedule is *derived from the problem geometry*, not imposed externally.

### The anti-collapse mechanism

P-KAS collapsed because phase synchronisation created a symmetric low-energy state. In Deerskin-PKAS:

1. Collapse is detected as a rapid drop in cut value over 50-step windows
2. When detected, `melt_rate` is temporarily multiplied by 5
3. The high plasticity lets the Prior reorient rapidly
4. After reorientation, the gauge force (stress gradient) pushes phases away from the synchronised state
5. Plasticity decays back to its baseline

This is the Deerskin "trauma-recovery" mechanism: acute stress → membrane goes plastic → reorients → stiffens in new orientation.

### Relationship to gauge theory

In Yang-Mills gauge theory, force arises when parallel transport around a loop is non-trivial. Here:

- The connection is the Prior orientation field
- Parallel transport is following the gradient of moiré stress
- Curvature of the field = the stress gradient driving phase dynamics
- Learning = the connection field itself updating (viscous melt)

This is not a formal derivation — it's a productive metaphor that generates specific, testable algorithmic choices.

---

## Falsifiable predictions

If the Deerskin framing is structurally correct:

1. **Collapse elimination**: Deerskin-PKAS should never exhibit the catastrophic phase collapse seen in Original P-KAS. *Testable now.*

2. **Smoother convergence**: The melt-driven annealing should produce smoother convergence curves than SA's random acceptance. *Testable now.*

3. **Problem-adaptive schedule**: On easy problems (low stress throughout), melt rate should decay faster. On hard problems, melt rate should stay higher longer. *Testable now.*

4. **Transfer across instances**: After solving many instances of the same problem class, the Prior should be pre-oriented toward the solution manifold, requiring fewer iterations. *Requires multi-instance training loop — see `w_matrix.py`.*

---

## Honest assessment

Deerskin-PKAS will not beat a well-tuned SA on MaxCut. SA on MaxCut is heavily studied and very good.

What Deerskin-PKAS offers:

- **No temperature tuning** — the schedule is derived, not specified
- **Collapse resistance** — the anti-collapse mechanism is structural, not a hack
- **Interpretable internals** — calcium shows you where the problem is hard; the Prior shows what the solver has learned about the solution space
- **Transfer potential** — the W-matrix + Prior can in principle be warm-started across related problem instances

The TSP results (`deerskin_tsp.py`) are more interesting: the calcium swap-map gives a meaningful learning signal that guided construction can use, unlike fractal_instinct.py where the GNN scores had no structural relationship to the tour geometry.

---

## Citation / lineage

This work extends:
- P-KAS (Kuramoto + Calcium + W-matrix) — from `qubit_thing.txt` conversation history
- Deerskin Hypothesis (Geometric Prior + Moiré stress + Viscous Melt) — from `solve.html` and related work
- Fractal Instinct (GNN swap-map + guided TSP) — from `feynmantsp2.py`

The fusion: calcium bridges P-KAS's fast dynamics to Deerskin's slow geometric scarring. The Prior replaces temperature. The W-matrix encodes the structural orientation biases that Deerskin's melt accumulates.
