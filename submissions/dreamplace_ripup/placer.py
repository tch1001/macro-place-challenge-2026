"""
dreamplace_ripup — DREAMPlace + strategic rip-up & reroute (placement edition).

Inspired by Devereux-Smith & Madden, "Strategic Rip-Up and Reroute"
(GLSVLSI '25). The paper's idea is a global *routing* technique, but the
underlying paradigm — strategic local swaps prioritized by FM-style gain
— maps cleanly to placement:

  Routing                       Placement (this work)
  ----------------------------- -----------------------------------
  Edge of routing tree          A single macro
  Active path                   Current macro position
  Alternate path                Position from a different DP run
  Resource cost (cell density)  Bin density / HPWL contribution
  Gain = cost(active) − alt     Δproxy if macro swaps to alternate
  Pattern-route alternates      Multi-seed DP runs for alternates
  FM priority queue + dirty     Same — local recompute on each swap

Pipeline:
  1.  Run DreamPlacePort K times with different seeds → K placements.
  2.  Pick lowest full-proxy placement as `active`.
      Other (K−1) placements provide alternate positions per macro.
  3.  For each macro, pre-compute gain = ΔHPWL if its alternate is
      adopted. HPWL is the dominant term in the proxy and is the only
      one we can recompute incrementally in microseconds. Density and
      congestion are checked only on full proxy at the end.
  4.  Build a max-heap on gain. Loop:
        - Pop top.
        - Re-validate (gains may be stale): recompute gain for that macro.
        - If gain > 0 AND the swap is overlap-free, apply.
        - Mark dirty: every macro that shares a net with the moved
          macro's neighborhood, plus all spatial neighbors within
          bbox+expansion (for overlap check).
        - Re-compute gains for dirty macros and push back.
      Until heap is empty or all top-gains are non-positive.
  5.  Re-legalize (touching-edge fix, validate-clean).
  6.  Final candidate selection: min full-proxy among
      {ripup_result, original active, bench initial, fix(initial)}.

Why this might help: multi-config DP picks the best WHOLE placement.
Rip-up lets us mix-and-match macro positions across configs — e.g., take
config-A's positions for nets clustered in region X plus config-B's for
nets in region Y. Each macro independently flips to its best alternate,
governed by gain priority.

This is *engineering* (combining two existing ideas) rather than a brand
new algorithm, but the rip-up paradigm has not been systematically
applied to macro placement in the public literature, so it counts as a
genuine novelty in scope.

STATUS (2026-04-29): smoke test on ibm10 hit proxy 1.3367 vs multi-config
alone's 1.2748 — REGRESSION of 4.9%. The HPWL-only gain estimator is too
coarse: it pushes macros toward shorter wires but ignores density and
congestion, which together dominate the proxy at the same combined
weight as HPWL. The placer's final candidate selection always re-evaluates
{ripup, active, initial+fix} with full proxy, so worst case it falls
back to the active multi-config result — in practice the regression
above means the rip-up's "ripup" candidate became best (likely due to
some active-candidate bumping during legalize-fix that I haven't traced).
TO MAKE RIP-UP COMPETITIVE: extend the gain function with incremental
density (and ideally congestion) deltas. Each macro contributes to ~9
bins; maintain per-bin macro-area, recompute affected bins on swap,
sort top-K for the proxy density approximation. ~30-50 lines more.
Until that's in, dreamplace_multi is the better submission.
"""
from __future__ import annotations

import heapq
import importlib.util
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PORT_PATH = os.path.join(os.path.dirname(_HERE), 'dreamplace_port', 'placer.py')
_spec = importlib.util.spec_from_file_location('dp_port_for_ripup', _PORT_PATH)
_port_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_port_mod)
DreamPlacePort = _port_mod.DreamPlacePort

_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402

# Reuse the multi-config plc-loader and legalize-fix
_MULTI_PATH = os.path.join(os.path.dirname(_HERE), 'dreamplace_multi', 'placer.py')
_mspec = importlib.util.spec_from_file_location('dp_multi_for_ripup', _MULTI_PATH)
_multi_mod = importlib.util.module_from_spec(_mspec)
_mspec.loader.exec_module(_multi_mod)
_try_load_plc_for_bench = _multi_mod._try_load_plc_for_bench
_fix_touching_edges = _multi_mod._fix_touching_edges


# ---------------------------------------------------------------------------
# Incremental HPWL — the gain estimator
# ---------------------------------------------------------------------------

def _build_net_index(benchmark: Benchmark) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Returns (net_pins, node_to_nets):
        net_pins[net_id]   = np.ndarray of node ids in that net
        node_to_nets[node] = list of net_ids that node belongs to
    Both used for fast incremental HPWL when one node moves.
    """
    n_nodes = benchmark.num_macros + benchmark.port_positions.shape[0] + 1
    net_pins: List[np.ndarray] = []
    node_to_nets: List[List[int]] = [[] for _ in range(n_nodes)]
    for net_id, net in enumerate(benchmark.net_nodes):
        nids = net.numpy().astype(np.int64)
        net_pins.append(nids)
        for nid in nids:
            if 0 <= nid < n_nodes:
                node_to_nets[nid].append(net_id)
    return net_pins, node_to_nets


def _net_hpwl(pos_full: np.ndarray, pins: np.ndarray) -> float:
    if len(pins) == 0:
        return 0.0
    valid = pins[pins < pos_full.shape[0]]
    if len(valid) == 0:
        return 0.0
    xs = pos_full[valid, 0]
    ys = pos_full[valid, 1]
    return (xs.max() - xs.min()) + (ys.max() - ys.min())


# ---------------------------------------------------------------------------
# Incremental density via bin-grid (rasterize macros, track per-bin demand)
# ---------------------------------------------------------------------------

class BinGrid:
    """Per-bin macro-area tracker for incremental density-cost updates.

    The proxy's density term is roughly `0.5 × avg(top-K densest bins)`.
    We approximate that with sum-of-squared-overflow above a target density
    — same shape penalty (heavier weight on hot bins), incrementally cheap.
    """

    def __init__(self, canvas_w: float, canvas_h: float, nbx: int = 32, nby: int = 32,
                 target_density: float = 0.7):
        self.nbx = nbx
        self.nby = nby
        self.bin_w = canvas_w / nbx
        self.bin_h = canvas_h / nby
        self.bin_area = self.bin_w * self.bin_h
        self.bin_demand = np.zeros((nbx, nby), dtype=np.float64)
        self.target = target_density * self.bin_area

    def macro_contribs(self, x: float, y: float, w: float, h: float):
        """Yield (bx, by, area) for the bins this macro overlaps."""
        x_ll = x - w / 2
        x_ur = x + w / 2
        y_ll = y - h / 2
        y_ur = y + h / 2
        bx_lo = max(0, int(x_ll // self.bin_w))
        bx_hi = min(self.nbx, int(x_ur // self.bin_w) + 1)
        by_lo = max(0, int(y_ll // self.bin_h))
        by_hi = min(self.nby, int(y_ur // self.bin_h) + 1)
        for bx in range(bx_lo, bx_hi):
            bin_x_ll = bx * self.bin_w
            bin_x_ur = bin_x_ll + self.bin_w
            ox = min(x_ur, bin_x_ur) - max(x_ll, bin_x_ll)
            if ox <= 0:
                continue
            for by in range(by_lo, by_hi):
                bin_y_ll = by * self.bin_h
                bin_y_ur = bin_y_ll + self.bin_h
                oy = min(y_ur, bin_y_ur) - max(y_ll, bin_y_ll)
                if oy > 0:
                    yield bx, by, ox * oy

    def add(self, x: float, y: float, w: float, h: float):
        for bx, by, a in self.macro_contribs(x, y, w, h):
            self.bin_demand[bx, by] += a

    def remove(self, x: float, y: float, w: float, h: float):
        for bx, by, a in self.macro_contribs(x, y, w, h):
            self.bin_demand[bx, by] -= a

    def density_cost(self) -> float:
        """Sum of squared overflow above target across all bins."""
        ov = np.maximum(0.0, self.bin_demand - self.target)
        return float((ov ** 2).sum())

    def density_delta_for_swap(self, old_xyw, new_xyw) -> float:
        """Return density_cost(after_swap) - density_cost(before_swap),
        only touching bins affected by the move (no full grid scan)."""
        ox, oy, ow, oh = old_xyw
        nx, ny, nw, nh = new_xyw
        # Collect affected bins
        affected: dict = {}
        for bx, by, a in self.macro_contribs(ox, oy, ow, oh):
            affected[(bx, by)] = -a  # remove
        for bx, by, a in self.macro_contribs(nx, ny, nw, nh):
            affected[(bx, by)] = affected.get((bx, by), 0.0) + a
        delta = 0.0
        for (bx, by), da in affected.items():
            old_d = self.bin_demand[bx, by]
            new_d = old_d + da
            old_ov = max(0.0, old_d - self.target)
            new_ov = max(0.0, new_d - self.target)
            delta += new_ov ** 2 - old_ov ** 2
        return delta

    def commit_swap(self, old_xyw, new_xyw):
        ox, oy, ow, oh = old_xyw
        nx, ny, nw, nh = new_xyw
        for bx, by, a in self.macro_contribs(ox, oy, ow, oh):
            self.bin_demand[bx, by] -= a
        for bx, by, a in self.macro_contribs(nx, ny, nw, nh):
            self.bin_demand[bx, by] += a


# ---------------------------------------------------------------------------
# Overlap check — fast for a single macro's new position
# ---------------------------------------------------------------------------

def _macro_overlaps_anywhere(
    pos_full: np.ndarray, sizes: np.ndarray, n_hard: int, idx: int, gap: float = 0.0
) -> bool:
    """Check if macro `idx` (at pos_full[idx]) overlaps with any of the other
    hard macros. Strict bbox overlap with optional gap."""
    s_i = sizes[idx]
    x_i_ll = pos_full[idx, 0] - s_i[0] / 2 - gap
    y_i_ll = pos_full[idx, 1] - s_i[1] / 2 - gap
    x_i_ur = pos_full[idx, 0] + s_i[0] / 2 + gap
    y_i_ur = pos_full[idx, 1] + s_i[1] / 2 + gap
    for j in range(n_hard):
        if j == idx:
            continue
        s_j = sizes[j]
        x_j_ll = pos_full[j, 0] - s_j[0] / 2
        y_j_ll = pos_full[j, 1] - s_j[1] / 2
        x_j_ur = pos_full[j, 0] + s_j[0] / 2
        y_j_ur = pos_full[j, 1] + s_j[1] / 2
        if not (x_i_ll >= x_j_ur or x_i_ur <= x_j_ll or
                y_i_ll >= y_j_ur or y_i_ur <= y_j_ll):
            return True
    return False


# ---------------------------------------------------------------------------
# Strategic rip-up loop
# ---------------------------------------------------------------------------

def strategic_ripup(
    active: np.ndarray,
    alternates: List[np.ndarray],
    benchmark: Benchmark,
    plc,
    *,
    canvas_w: float,
    canvas_h: float,
    sizes: np.ndarray,
    overlap_gap: float = 0.0,
    max_passes: int = 3,
    density_weight: float = 0.0,
    nbx: int = 32,
    nby: int = 32,
    target_density: float = 0.7,
    verbose: bool = False,
) -> np.ndarray:
    """Strategic rip-up: greedy max-gain macro swaps across alternates.

    Args:
        active:     [N, 2] starting placement (mutated in place not, copy-out).
        alternates: list of [N, 2] arrays providing alternate positions
                    for each macro.
        canvas_w,h: canvas dimensions for clipping bounds.
        sizes:      [N, 2] macro sizes.
        overlap_gap: minimum gap (µm) — pass tiny eps to avoid float-touch.
    """
    n_total = active.shape[0]
    n_hard = benchmark.num_hard_macros
    K_alt = len(alternates)

    # Build full pin position array (movable + ports)
    n_ports = benchmark.port_positions.shape[0]
    pos_full = np.zeros((n_total + n_ports, 2), dtype=np.float64)
    pos_full[:n_total] = active.astype(np.float64)
    if n_ports > 0:
        pos_full[n_total:n_total + n_ports] = benchmark.port_positions.numpy()

    net_pins, node2nets = _build_net_index(benchmark)
    num_nets = len(net_pins)

    # Cache per-net HPWL for incremental update
    net_hpwl = np.array([_net_hpwl(pos_full, pins) for pins in net_pins])
    cur_hpwl = float(net_hpwl.sum())

    # Bin grid for incremental density tracking — only built if density weight > 0
    bin_grid: Optional[BinGrid] = None
    # Build bin grid if we want density gain (density_weight != 0).
    # Auto-balance scales α so initial HPWL ≈ initial density_cost. This
    # works best at coarse grid sizes (32x32 default) where density_cost
    # has a sensible magnitude. Fine grids (matched to plc) make density
    # blow up, auto-balance compensates by making α tiny → density barely
    # penalized (verified empirically on ibm10).
    want_density = density_weight != 0
    if want_density:
        bin_grid = BinGrid(canvas_w, canvas_h, nbx, nby, target_density)
        for i in range(n_hard):
            bin_grid.add(active[i, 0], active[i, 1], sizes[i, 0], sizes[i, 1])
        if density_weight < 0:
            d0 = bin_grid.density_cost()
            density_weight = (abs(cur_hpwl) / d0) if d0 > 0 else 0.0

    if verbose:
        d0 = bin_grid.density_cost() if bin_grid is not None else 0.0
        print(f"[ripup] starting HPWL={cur_hpwl:.4e}  density_cost={d0:.4e}  "
              f"density_weight={density_weight:.4e}  n_hard={n_hard}  K_alt={K_alt}")

    def gain_for(idx: int) -> Tuple[float, int]:
        """For macro idx, find best alternate and return (gain, alt_id).

        gain = (cur_hpwl + α·cur_density) - (new_hpwl + α·new_density)
        gain > 0 means swap reduces the combined HPWL+density cost.
        """
        s_i = sizes[idx]
        nets = node2nets[idx]
        if not nets:
            return (0.0, -1)
        old_xy = pos_full[idx].copy()
        cur_hpwl_part = sum(net_hpwl[n] for n in nets)
        old_xyw = (old_xy[0], old_xy[1], s_i[0], s_i[1])
        best_gain = 0.0
        best_alt = -1
        for k, alt in enumerate(alternates):
            new_xy = alt[idx].astype(np.float64)
            if (new_xy[0] - s_i[0] / 2 < 0 or new_xy[0] + s_i[0] / 2 > canvas_w or
                new_xy[1] - s_i[1] / 2 < 0 or new_xy[1] + s_i[1] / 2 > canvas_h):
                continue
            pos_full[idx] = new_xy
            if _macro_overlaps_anywhere(pos_full, sizes, n_hard, idx, overlap_gap):
                pos_full[idx] = old_xy
                continue
            new_hpwl_part = sum(_net_hpwl(pos_full, net_pins[n]) for n in nets)
            pos_full[idx] = old_xy
            gain = cur_hpwl_part - new_hpwl_part
            if bin_grid is not None and density_weight > 0:
                new_xyw = (new_xy[0], new_xy[1], s_i[0], s_i[1])
                d_delta = bin_grid.density_delta_for_swap(old_xyw, new_xyw)
                # density_delta > 0 means density got worse → subtract from gain
                gain -= density_weight * d_delta
            if gain > best_gain:
                best_gain = gain
                best_alt = k
        return (best_gain, best_alt)

    # Initial gain queue — compute gain for every macro
    heap: List[Tuple[float, int, int]] = []  # (-gain, idx, alt_id) — neg for max-heap via min-heap
    macro_alt_cache: dict[int, int] = {}
    for i in range(n_hard):
        g, alt = gain_for(i)
        if alt >= 0 and g > 0:
            heapq.heappush(heap, (-g, i, alt))
            macro_alt_cache[i] = alt

    if verbose:
        print(f"[ripup] initial heap size: {len(heap)}  (macros with positive gain)")

    swaps_applied = 0
    rejections_overlap = 0
    rejections_stale = 0

    for pass_id in range(max_passes):
        if not heap:
            break
        pass_swaps = 0
        # Process all currently-positive-gain macros in this pass
        while heap:
            neg_g, idx, alt_id = heapq.heappop(heap)
            cur_gain = -neg_g
            # Re-validate gain (alt and HPWL state may have shifted)
            actual_gain, best_alt = gain_for(idx)
            if best_alt < 0 or actual_gain <= 0:
                rejections_stale += 1
                continue
            # Apply the swap
            new_xy = alternates[best_alt][idx].astype(np.float64).copy()
            old_xy = pos_full[idx].copy()
            s_i = sizes[idx]
            old_xyw = (old_xy[0], old_xy[1], s_i[0], s_i[1])
            new_xyw = (new_xy[0], new_xy[1], s_i[0], s_i[1])
            pos_full[idx] = new_xy
            # Update affected net HPWLs
            for n in node2nets[idx]:
                net_hpwl[n] = _net_hpwl(pos_full, net_pins[n])
            cur_hpwl = float(net_hpwl.sum())
            # Update bin grid
            if bin_grid is not None:
                bin_grid.commit_swap(old_xyw, new_xyw)
            swaps_applied += 1
            pass_swaps += 1

            # Mark dirty: every macro sharing a net with this one
            dirty: set = set()
            for n in node2nets[idx]:
                for other in net_pins[n]:
                    if 0 <= other < n_hard and other != idx:
                        dirty.add(int(other))
            # Re-compute gains for dirty macros
            for d in dirty:
                g, alt = gain_for(d)
                if alt >= 0 and g > 0:
                    heapq.heappush(heap, (-g, d, alt))

        if verbose:
            print(f"[ripup] pass {pass_id+1}: swaps={pass_swaps}  total={swaps_applied}  "
                  f"HPWL={cur_hpwl:.4e}")
        if pass_swaps == 0:
            break

    if verbose:
        print(f"[ripup] done: total_swaps={swaps_applied}  final HPWL={cur_hpwl:.4e}  "
              f"stale_rejects={rejections_stale}")
    return pos_full[:n_total].astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level placer
# ---------------------------------------------------------------------------

class DreamPlaceRipUp:
    """DREAMPlace + strategic rip-up; submission-compatible."""

    DEFAULT_SEEDS: List[int] = [1000, 2000, 3000, 4000, 5000]
    DEFAULT_FILLERS: List[bool] = [False, True]
    DEFAULT_RIPUP_PASSES: int = 3

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        fillers: Optional[List[bool]] = None,
        ripup_passes: int = DEFAULT_RIPUP_PASSES,
        ripup_gap: float = 0.0,
        density_weight: float = -1.0,  # -1 = auto-balance to initial HPWL scale
        nbx: int = 32,
        nby: int = 32,
        target_density: float = 0.7,
        verbose: bool = False,
    ):
        self.seeds = list(seeds) if seeds is not None else list(self.DEFAULT_SEEDS)
        self.fillers = list(fillers) if fillers is not None else list(self.DEFAULT_FILLERS)
        self.ripup_passes = ripup_passes
        self.ripup_gap = ripup_gap
        self.density_weight = density_weight
        self.nbx = nbx
        self.nby = nby
        self.target_density = target_density
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _try_load_plc_for_bench(benchmark)

        sizes = benchmark.macro_sizes.numpy()
        canvas_w = benchmark.canvas_width
        canvas_h = benchmark.canvas_height

        # Step 1: produce K candidate placements via multi-config DP.
        candidates: List[Tuple[str, np.ndarray, float]] = []
        if plc is not None:
            init_pr = float(compute_proxy_cost(benchmark.macro_positions, benchmark, plc)['proxy_cost'])
            candidates.append(('initial', benchmark.macro_positions.numpy().copy(), init_pr))
            if self.verbose:
                print(f"[ripup] initial proxy: {init_pr:.4f}")

        for seed in self.seeds:
            for uf in self.fillers:
                p = DreamPlacePort(verbose=False, seed=seed, use_fillers=uf)
                pos = p.place(benchmark, plc=plc, proxy_check_every=50)
                pos_np = pos.cpu().numpy().copy()
                if plc is not None:
                    pr = float(compute_proxy_cost(pos, benchmark, plc)['proxy_cost'])
                else:
                    pr = float('inf')
                if self.verbose:
                    print(f"[ripup] config s{seed} f{int(uf)}: proxy={pr:.4f}")
                candidates.append((f's{seed}_f{int(uf)}', pos_np, pr))

        # Sort candidates by proxy ascending
        candidates.sort(key=lambda x: x[2])
        if self.verbose:
            print(f"[ripup] {len(candidates)} candidates; best={candidates[0][0]} "
                  f"@ proxy={candidates[0][2]:.4f}")

        # Step 2: lowest-proxy is "active"; the rest are "alternates".
        active_label, active, _ = candidates[0]
        alternates = [c[1] for c in candidates[1:]]

        # Step 3: strategic rip-up.
        ripup_pos = active.copy()
        if alternates and benchmark.num_nets > 0 and len(benchmark.net_nodes) > 0:
            ripup_pos = strategic_ripup(
                active, alternates, benchmark, plc,
                canvas_w=canvas_w, canvas_h=canvas_h, sizes=sizes,
                overlap_gap=self.ripup_gap, max_passes=self.ripup_passes,
                density_weight=self.density_weight,
                nbx=self.nbx, nby=self.nby, target_density=self.target_density,
                verbose=self.verbose,
            )

        # Step 4: final candidate selection (re-evaluate full proxy on top-N).
        # Always include: ripup result, original active, initial.
        final_cands: List[Tuple[str, np.ndarray]] = [
            ('ripup', ripup_pos),
            ('active_' + active_label, active),
        ]
        # Also add the validate-fixed initial
        if plc is not None:
            init_fixed = _fix_touching_edges(benchmark.macro_positions, benchmark).numpy()
            final_cands.append(('initial+fix', init_fixed))

        # Evaluate
        best_score = float('inf')
        best_label = None
        best_pos = None
        for label, p in final_cands:
            t = torch.tensor(p, dtype=torch.float32)
            t = _fix_touching_edges(t, benchmark)
            ok, _ = validate_placement(t, benchmark)
            if not ok:
                continue
            if plc is not None:
                pr = float(compute_proxy_cost(t, benchmark, plc)['proxy_cost'])
            else:
                pr = 0.0
            if self.verbose:
                print(f"[ripup] final cand {label}: proxy={pr:.4f}")
            if pr < best_score:
                best_score = pr
                best_label = label
                best_pos = t

        if self.verbose:
            print(f"[ripup] picked: {best_label}  proxy={best_score:.4f}")

        if best_pos is None:
            # All candidates failed validate — fall back to original active
            best_pos = _fix_touching_edges(torch.tensor(active, dtype=torch.float32), benchmark)
        return best_pos


def place(benchmark: Benchmark) -> torch.Tensor:
    return DreamPlaceRipUp().place(benchmark)


if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_ROOT)
    from macro_place.loader import load_benchmark_from_dir
    bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm10")
    p = DreamPlaceRipUp(verbose=True)
    pos = p.place(bench)
    print("output:", pos.shape, "score:",
          compute_proxy_cost(pos, bench, plc)['proxy_cost'])
