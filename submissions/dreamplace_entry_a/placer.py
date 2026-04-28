"""
dreamplace_entry_a — DREAMPlace port + post-optimization SA refinement (experimental).

Status (2026-04-29): scaffolding works, but the SA pass has not yet found
improvements over the DP output on tested IBMs. See "Honest assessment"
below. Final candidate selection always falls back to one of {DP_output,
initial}, so this placer is a strict superset of dreamplace_port — it can
only match or beat, never lose. Runtime overhead ~30-60s per bench.

Pipeline:
1. Run dreamplace_port (analytical ePlace-style placer) to convergence.
2. Apply SA refinement on (a) the DP output, (b) the bench's initial
   placement. SA inner cost is HPWL only (vectorized numpy, ms/eval).
3. Re-evaluate all 4 candidates with the slow-but-correct
   compute_proxy_cost; pick the min.

Why a hybrid analytical + SA approach is interesting (the *idea*):
- DP gradient only sees wirelength + density. It is BLIND to the
  contest's congestion term (weight 0.5 in proxy).
- DP can't reach SA-style local minima — empirically true on ibm01/06/08
  and on ariane133 where the bench's provided (SA-derived) initial beats
  anything DP finds.
- SA on its own lacks global-structure-finding power.
- Hybrid: DP for global, SA for local refinement.

Honest assessment:
- The fast-HPWL SA inner cost is too coarse: HPWL improvements don't
  reliably correlate with proxy improvements when density/congestion are
  the bottleneck.
- Random-Gaussian displacements on dense IBM canvases (ibm10: 78x78µm,
  387 macros) hit neighbors ~80% of the time and get rejected.
- Pair-swaps work better on uniform-size macros (most of NG45) but
  rarely help on IBMs where macros vary 33× in area.
- Real proxy_cost takes ~10-30s per call (verified) so we cannot use it
  as the SA inner cost.

Promising directions (not implemented):
- **Differentiable congestion via RUDY-like routing-demand model** —
  add a third gradient term for congestion to the port's loss. Real
  algorithmic addition; the port currently only optimizes WL+density.
- **Smarter move proposals**: target empty regions (Hannan grid free
  slots) rather than Gaussian. Reduces overlap rejection.
- **Pair-swap with HPWL-delta computed BEFORE swap** (only commit if
  predicted to improve). Saves the full HPWL-recomputation cost.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch

# Import the dreamplace_port placer as the analytical core.
# Both placer files have module-name `placer`, so use importlib to load the
# port file by absolute path to avoid name collision.
import importlib.util as _ilu
_HERE = os.path.dirname(os.path.abspath(__file__))
_PORT_PATH = os.path.join(os.path.dirname(_HERE), 'dreamplace_port', 'placer.py')
_spec = _ilu.spec_from_file_location('dreamplace_port_placer', _PORT_PATH)
port_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(port_mod)
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost


def _has_overlap(centers: np.ndarray, sizes: np.ndarray, n_hard: int, gap: float = 0.0) -> bool:
    """Check if any pair of hard macros overlaps (within `gap` slack)."""
    hp = centers[:n_hard]
    s = sizes[:n_hard]
    x_ll = hp[:, 0] - s[:, 0] / 2
    y_ll = hp[:, 1] - s[:, 1] / 2
    x_ur = x_ll + s[:, 0]
    y_ur = y_ll + s[:, 1]
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            gx = max(x_ll[j] - x_ur[i], x_ll[i] - x_ur[j], 0.0) - gap
            gy = max(y_ll[j] - y_ur[i], y_ll[i] - y_ur[j], 0.0) - gap
            if gx <= 0 and gy <= 0:
                return True
    return False


def _move_violates(centers: np.ndarray, sizes: np.ndarray, n_hard: int, idx: int,
                   gap: float = 0.0, canvas_w: float = 0.0, canvas_h: float = 0.0) -> bool:
    """Check whether moving macro `idx` causes a new overlap or off-canvas.

    Faster than a full O(n^2) check — only checks idx vs all others.
    """
    s_i = sizes[idx]
    x_i_ll = centers[idx, 0] - s_i[0] / 2
    y_i_ll = centers[idx, 1] - s_i[1] / 2
    x_i_ur = x_i_ll + s_i[0]
    y_i_ur = y_i_ll + s_i[1]
    if canvas_w > 0 and (x_i_ll < 0 or x_i_ur > canvas_w):
        return True
    if canvas_h > 0 and (y_i_ll < 0 or y_i_ur > canvas_h):
        return True
    for j in range(n_hard):
        if j == idx:
            continue
        s_j = sizes[j]
        x_j_ll = centers[j, 0] - s_j[0] / 2
        y_j_ll = centers[j, 1] - s_j[1] / 2
        x_j_ur = x_j_ll + s_j[0]
        y_j_ur = y_j_ll + s_j[1]
        gx = max(x_j_ll - x_i_ur, x_i_ll - x_j_ur, 0.0) - gap
        gy = max(y_j_ll - y_i_ur, y_i_ll - y_j_ur, 0.0) - gap
        if gx <= 0 and gy <= 0:
            return True
    return False


def _build_net_index(bench: Benchmark) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten net_nodes into parallel (pin_node_id, pin_net_id) arrays plus
    a per-node-to-net inverted index for fast incremental HPWL updates.

    Returns:
        pin_node:  [P] int — node id per pin
        pin_net:   [P] int — net id per pin
        node2nets: dict-like list-of-lists (numpy ragged via list)
    """
    pin_node = []
    pin_net = []
    node2nets: list[list[int]] = [[] for _ in range(bench.num_macros + bench.port_positions.shape[0] + 1)]
    for net_id, net in enumerate(bench.net_nodes):
        for nid in net.tolist():
            pin_node.append(nid)
            pin_net.append(net_id)
            if nid < len(node2nets):
                node2nets[nid].append(net_id)
    return np.asarray(pin_node, dtype=np.int64), np.asarray(pin_net, dtype=np.int64), node2nets


def _hpwl_total(pos_full: np.ndarray, pin_node: np.ndarray, pin_net: np.ndarray, num_nets: int) -> float:
    """Vectorized HPWL = sum over nets of (max_x - min_x + max_y - min_y)."""
    px = pos_full[pin_node, 0]
    py = pos_full[pin_node, 1]
    # bincount-style reductions
    inf = np.full(num_nets, np.inf)
    ninf = np.full(num_nets, -np.inf)
    x_min = np.minimum.reduceat(np.r_[inf,  px], np.arange(num_nets))  # not quite right; use np.minimum.at
    # Simpler vectorization via np.minimum.at / maximum.at
    xmin = np.full(num_nets, np.inf)
    xmax = np.full(num_nets, -np.inf)
    ymin = np.full(num_nets, np.inf)
    ymax = np.full(num_nets, -np.inf)
    np.minimum.at(xmin, pin_net, px)
    np.maximum.at(xmax, pin_net, px)
    np.minimum.at(ymin, pin_net, py)
    np.maximum.at(ymax, pin_net, py)
    return float((xmax - xmin).sum() + (ymax - ymin).sum())


def _hpwl_for_nets(pos_full: np.ndarray, net_pins: list, net_ids) -> float:
    """HPWL summed over a specific subset of nets (for incremental updates)."""
    total = 0.0
    for nid in net_ids:
        pins = net_pins[nid]
        if len(pins) == 0:
            continue
        xs = pos_full[pins, 0]
        ys = pos_full[pins, 1]
        total += (xs.max() - xs.min()) + (ys.max() - ys.min())
    return total


def sa_refine(
    starting_pos: np.ndarray,
    bench: Benchmark,
    plc,
    sizes: np.ndarray,
    n_hard: int,
    *,
    moves: int = 1500,
    initial_temp_frac: float = 0.02,
    cool_rate: float = 0.997,
    gap: float = 0.0,
    seed: int = 1000,
    verbose: bool = False,
) -> tuple[np.ndarray, float]:
    """SA refinement using fast HPWL cost (vectorized numpy, no plc round-trip).

    Real proxy_cost evaluation is too slow (~10-30s per call for ibm/ariane).
    We use HPWL-only as the SA inner cost — it's the dominant term in the
    proxy (1.0× weight) and is locally well-correlated with proxy. Final
    candidate selection re-evaluates with actual proxy.

    Args:
        starting_pos: [num_macros + port + ...] starting placement
        bench, plc:   for canvas dims and net structure
        sizes:        [num_macros, 2] macro sizes
        n_hard:       hard (movable) macro count
        moves:        SA move attempts
        initial_temp_frac, cool_rate, gap, seed: standard SA knobs

    Returns:
        (best_pos, best_hpwl_cost)
    """
    rng = np.random.default_rng(seed)
    pos = starting_pos.copy().astype(np.float64)
    canvas_w, canvas_h = plc.get_canvas_width_height()

    # Move statistics for diagnosis
    rejected_overlap = 0
    rejected_metro = 0

    # Build full position array including ports (ports are fixed, but contribute
    # to HPWL when nets cross them).
    n_total = pos.shape[0]
    n_ports = bench.port_positions.shape[0]
    pos_full = np.zeros((n_total + n_ports, 2), dtype=np.float64)
    pos_full[:n_total] = pos
    if n_ports > 0:
        pos_full[n_total:n_total + n_ports] = bench.port_positions.numpy()

    # Net indexing
    num_nets = bench.num_nets
    net_pins = [net.numpy().astype(np.int64) for net in bench.net_nodes]  # list per net
    node2nets = [[] for _ in range(n_total + n_ports + 1)]
    for net_id, pins in enumerate(net_pins):
        for nid in pins:
            if 0 <= nid < len(node2nets):
                node2nets[nid].append(net_id)

    # Initial HPWL
    pin_node = np.concatenate(net_pins) if net_pins else np.zeros(0, dtype=np.int64)
    pin_net = np.repeat(np.arange(num_nets), [len(p) for p in net_pins])
    cur_hpwl = _hpwl_total(pos_full, pin_node, pin_net, num_nets)
    best_pos = pos.copy()
    best_hpwl = cur_hpwl

    T = initial_temp_frac * abs(cur_hpwl)
    if verbose:
        print(f"[entry_a SA] initial HPWL={cur_hpwl:.3e}  T0={T:.3e}  moves={moves}")

    accepted = 0
    improved = 0
    for it in range(moves):
        move_type = 'displace' if rng.random() < 0.7 else 'swap'

        if move_type == 'displace':
            idx = int(rng.integers(0, n_hard))
            old_xy = pos[idx].copy()
            # Larger moves early (canvas-scale scatter), shrink as T cools
            T_frac = T / max(initial_temp_frac * abs(cur_hpwl), 1e-9)
            scale_frac = max(0.01, 0.15 * T_frac)
            sigma_x = canvas_w * scale_frac
            sigma_y = canvas_h * scale_frac
            pos[idx, 0] += rng.normal(0, sigma_x)
            pos[idx, 1] += rng.normal(0, sigma_y)
            # clamp into canvas
            s = sizes[idx]
            pos[idx, 0] = np.clip(pos[idx, 0], s[0]/2, canvas_w - s[0]/2)
            pos[idx, 1] = np.clip(pos[idx, 1], s[1]/2, canvas_h - s[1]/2)
            if _move_violates(pos, sizes, n_hard, idx, gap, canvas_w, canvas_h):
                pos[idx] = old_xy
                rejected_overlap += 1
                continue
            affected_nets = node2nets[idx]
            pos_full[idx] = old_xy
            old_partial = _hpwl_for_nets(pos_full, net_pins, affected_nets)
            pos_full[idx] = pos[idx]
            new_partial = _hpwl_for_nets(pos_full, net_pins, affected_nets)
            new_hpwl = cur_hpwl + (new_partial - old_partial)
        else:  # swap
            i = int(rng.integers(0, n_hard))
            j = int(rng.integers(0, n_hard))
            if i == j:
                continue
            old_i = pos[i].copy()
            old_j = pos[j].copy()
            pos[i], pos[j] = old_j, old_i
            if (_move_violates(pos, sizes, n_hard, i, gap, canvas_w, canvas_h) or
                _move_violates(pos, sizes, n_hard, j, gap, canvas_w, canvas_h)):
                pos[i] = old_i
                pos[j] = old_j
                rejected_overlap += 1
                continue
            affected = list(set(node2nets[i] + node2nets[j]))
            # old: pos_full has old_i at i and old_j at j (state before swap)
            pos_full[i] = old_i
            pos_full[j] = old_j
            old_partial = _hpwl_for_nets(pos_full, net_pins, affected)
            pos_full[i] = pos[i]
            pos_full[j] = pos[j]
            new_partial = _hpwl_for_nets(pos_full, net_pins, affected)
            new_hpwl = cur_hpwl + (new_partial - old_partial)

        delta = new_hpwl - cur_hpwl
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            cur_hpwl = new_hpwl
            accepted += 1
            if new_hpwl < best_hpwl:
                best_hpwl = new_hpwl
                best_pos = pos.copy()
                improved += 1
            T *= cool_rate
        else:
            rejected_metro += 1
            # revert
            if move_type == 'displace':
                pos[idx] = old_xy
                pos_full[idx] = old_xy
            else:
                pos[i] = old_i
                pos[j] = old_j
                pos_full[i] = old_i
                pos_full[j] = old_j

        if verbose and (it + 1) % 200 == 0:
            print(f"[entry_a SA] move {it+1:5d}  cur HPWL={cur_hpwl:.3e}  best={best_hpwl:.3e}  T={T:.3e}  accepted={accepted}  improved={improved}  rej_overlap={rejected_overlap}  rej_metro={rejected_metro}")

    if verbose:
        print(f"[entry_a SA] done: best HPWL={best_hpwl:.3e}  accepted={accepted}/{moves}  improved={improved}  rej_overlap={rejected_overlap}  rej_metro={rejected_metro}")
    return best_pos.astype(np.float32), best_hpwl


class DreamPlaceEntryA(port_mod.DreamPlacePort):
    """DreamPlace port + simulated-annealing proxy-cost refinement.

    Public-API drop-in replacement for ``DreamPlacePort``. Adds two kwargs:

    Args:
        sa_moves: SA moves per starting point (0 disables SA refinement,
                  reverts to plain dreamplace_port behavior).
        sa_seed: SA RNG seed.
    """

    def __init__(
        self,
        *args,
        sa_moves: int = 1500,
        sa_seed: Optional[int] = None,
        sa_initial_temp_frac: float = 0.02,
        sa_cool_rate: float = 0.99,
        sa_gap: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sa_moves = sa_moves
        self.sa_seed = sa_seed if sa_seed is not None else self.seed + 1
        self.sa_initial_temp_frac = sa_initial_temp_frac
        self.sa_cool_rate = sa_cool_rate
        self.sa_gap = sa_gap

    def place(self, bench: Benchmark, plc=None, proxy_check_every: int = 50) -> torch.Tensor:
        # Run analytical placement first
        port_pos = super().place(bench, plc=plc, proxy_check_every=proxy_check_every)

        if plc is None or self.sa_moves <= 0:
            return port_pos

        # Build size array
        n_hard = len(bench.hard_macro_indices)
        sizes = np.zeros((bench.num_macros, 2), dtype=np.float64)
        for i, idx in enumerate(bench.hard_macro_indices):
            node = plc.modules_w_pins[idx]
            sizes[i, 0] = node.get_width()
            sizes[i, 1] = node.get_height()

        port_np = port_pos.cpu().numpy()
        port_proxy = float(compute_proxy_cost(port_pos, bench, plc)['proxy_cost'])
        if self.verbose:
            print(f"[entry_a] DP output proxy={port_proxy:.4f}")

        # SA refine the DP output (cost = HPWL only, fast)
        sa_pos, sa_hpwl = sa_refine(
            port_np, bench, plc, sizes, n_hard,
            moves=self.sa_moves,
            initial_temp_frac=self.sa_initial_temp_frac,
            cool_rate=self.sa_cool_rate,
            gap=self.sa_gap,
            seed=self.sa_seed,
            verbose=self.verbose,
        )

        # Also SA-refine the bench's initial placement
        init_np = bench.macro_positions.cpu().numpy().copy()
        if init_np.shape[0] < port_np.shape[0]:
            full = port_np.copy()
            full[:init_np.shape[0]] = init_np
            init_np = full
        sa_init_pos, _ = sa_refine(
            init_np, bench, plc, sizes, n_hard,
            moves=self.sa_moves,
            initial_temp_frac=self.sa_initial_temp_frac,
            cool_rate=self.sa_cool_rate,
            gap=self.sa_gap,
            seed=self.sa_seed + 17,
            verbose=False,
        )

        # Final selection: re-evaluate all 4 candidates with REAL proxy
        # (slow: ~10-30s × 4 ≈ 1-2 min per bench)
        if self.verbose:
            print(f"[entry_a] evaluating 4 candidates with real proxy_cost...")
        cands = [
            ('DP_output', port_np),
            ('SA(DP)',    sa_pos),
            ('initial',   init_np),
            ('SA(init)',  sa_init_pos),
        ]
        scored = []
        for name, p in cands:
            pr = float(compute_proxy_cost(torch.tensor(p, dtype=torch.float32), bench, plc)['proxy_cost'])
            scored.append((name, p, pr))
            if self.verbose:
                print(f"[entry_a] cand {name}: proxy={pr:.4f}")
        scored.sort(key=lambda x: x[2])
        if self.verbose:
            print(f"[entry_a] picked: {scored[0][0]} at proxy={scored[0][2]:.4f}")
        return torch.tensor(scored[0][1], dtype=torch.float32)
