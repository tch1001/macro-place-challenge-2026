"""
exp_dcgp — DREAMPlace port + DCGP-style virtual-cell net-moving congestion gradient.

Adapts DCGP (DAC 2025: "Differentiable Net-Moving and Local Congestion
Mitigation for Routability-Driven Global Placement", Wenchao Li et al., Fuzhou U)
for IBM ICCAD04 macro placement.

Novel piece (Stage B of DCGP): instead of penalizing every cell inside a net's
congested bbox uniformly, we place a *virtual cell* at the maximum-congestion
sample-point along the segment of a two-pin net, compute the congestion
gradient there, project that gradient perpendicular to the pin-to-pin
direction, then propagate it to the two endpoint cells weighted by inverse
distance. This MOVES the entire net's pivot away from a hotspot while
preserving the topological pin-to-pin relationship — congested two-pin nets
get pushed sideways as a unit, instead of one endpoint being yanked into a
non-congested region while the other stays put.

Stages mapped to this code:
  - Stage A: differentiable congestion via the same RUDY-style demand map as
    exp_congestion_dp (consistent with proxy oracle). The full Poisson/DCT2
    machinery is overkill for IBMs without routing-layer info; RUDY is the
    "simple Stage A" the prompt allowed.
  - Stage B: the virtual-cell net-moving routine (this file's novel piece).
    Implemented in `_apply_net_moving_grad` — runs AFTER the standard
    backward() pass and ADDS its term to x.grad in-place.
  - Stage C: cell inflation via the historical-max momentum (cumulative
    `inflation_history` array; we use it to upweight a cell's contribution to
    the density map, which has the same effect as inflation in DREAMPlace).
  - Stage D: total loss = WL + λ_d·D + λ_c·C, with the additional Stage-B
    gradient added on top of dC/dx flowing through λ_c·C. λ_c uses a
    schedule similar to v2's `late_ramp` / `ovf_gated`.

Adaptations for IBMs:
  - "Cells" = macros (movable + soft).
  - Congestion grid matches proxy oracle's grid (bench.grid_cols × grid_rows).
  - Skip pin-accessibility (Stage III-C in paper) — IBMs have no pin info.
  - For multi-pin nets, we apply Stage B only to two-pin nets (clean novel
    contribution); the standard congestion gradient handles multi-pin nets.

CRITICAL: positions are clamped to canvas with eps=1e-3 at every iteration
(float32 boundary bug seen in v3 placer pushed macro 36 past canvas).

Usage:
    python submissions/exp_dcgp/run_smoke.py
    python -m macro_place.evaluate submissions/exp_dcgp/placer.py --bench ibm10
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

_TT = int(os.environ.get('EXP_DCGP_TORCH_THREADS', '1')) if 'EXP_DCGP_TORCH_THREADS' in os.environ else 1
try:
    torch.set_num_threads(_TT)
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Reuse the underlying DP port helpers (load by file path to avoid name collision).
_PORT_PATH = os.path.join(os.path.dirname(_HERE), 'dreamplace_port', 'placer.py')
_spec = importlib.util.spec_from_file_location('dreamplace_port_placer_dcgp', _PORT_PATH)
_port_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_port_mod)
_build_pin_arrays = _port_mod._build_pin_arrays
wa_wirelength = _port_mod.wa_wirelength
bin_density_map = _port_mod.bin_density_map
electric_potential_energy = _port_mod.electric_potential_energy
compute_overflow = _port_mod.compute_overflow
_legalize_hard = _port_mod._legalize_hard
_has_hard_overlap = _port_mod._has_hard_overlap
_greedy_slot = _port_mod._greedy_slot

# Reuse v1's differentiable RUDY congestion surrogate.
_V1_PATH = os.path.join(os.path.dirname(_HERE), 'exp_congestion_dp', 'placer.py')
_v1_spec = importlib.util.spec_from_file_location('exp_cong_v1_for_dcgp', _V1_PATH)
_v1_mod = importlib.util.module_from_spec(_v1_spec)
_v1_spec.loader.exec_module(_v1_mod)
_congestion_demand = _v1_mod._congestion_demand
congestion_loss = _v1_mod.congestion_loss

from macro_place.benchmark import Benchmark  # noqa: E402
from macro_place.objective import compute_proxy_cost  # noqa: E402
from macro_place.utils import validate_placement  # noqa: E402


# ---------------------------------------------------------------------------
# Two-pin net pre-processing
# ---------------------------------------------------------------------------

def _collect_two_pin_nets(bench: Benchmark) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (net_idx_2p, endpoint_pairs) tensors.

    net_idx_2p[k] = original net index for the k-th 2-pin net.
    endpoint_pairs[k] = (node_id_a, node_id_b) — the two endpoint pin-nodes.

    Skips nets with <2 distinct nodes (single-pin / collapsed).
    """
    pairs = []
    net_ids = []
    for i, net in enumerate(bench.net_nodes):
        nids = net.tolist()
        if len(nids) != 2:
            continue
        a, b = int(nids[0]), int(nids[1])
        if a == b:
            continue
        pairs.append((a, b))
        net_ids.append(i)
    if not pairs:
        return (
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0, 2, dtype=torch.long),
        )
    return (
        torch.tensor(net_ids, dtype=torch.long),
        torch.tensor(pairs, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Stage B: virtual-cell net-moving gradient
# ---------------------------------------------------------------------------

def _net_moving_grad(
    all_pos: torch.Tensor,
    pin_node: torch.Tensor,
    pin_net: torch.Tensor,
    num_nets: int,
    canvas_w: float,
    canvas_h: float,
    grid_col: int,
    grid_row: int,
    grid_v_routes: float,
    grid_h_routes: float,
    net_weights: Optional[torch.Tensor],
    twop_endpoints: torch.Tensor,    # [K, 2] node ids per 2-pin net
    cong_thresh: float = 0.7,
    n_samples: int = 5,
) -> torch.Tensor:
    """Compute the Stage-B per-node congestion gradient for all 2-pin nets.

    Returns: tensor of shape `all_pos.shape`. Add this directly to x.grad
    (it carries gradient w.r.t. `all_pos`).

    Algorithm per 2-pin net (p1, p2):
      1. Sample (n_samples+1) points along segment p1->p2 (including
         endpoints).
      2. Find the sample with the highest congestion `C_{m,n}` (RUDY demand).
         Skip if max < cong_thresh.
      3. Compute ∇C w.r.t. the virtual-cell position (xv, yv) by autograd
         on a tiny re-evaluation of the demand map at (xv, yv) (we use the
         analytic gradient: each net's contribution to bin (m,n) is
         linear in inv_h * inv_w * overlap_areas, but we approximate via
         simple finite differences for tractability — see implementation).
      4. Project ∇C onto unit normal n̂ ⟂ segment, oriented to be acute with
         ∇C.
      5. Add (L/(2·d_iv)) · ∇C_⊥ to each endpoint's gradient.

    For tractability and to avoid an autograd inside this function, we
    compute ∇C at the sample point by a 4-point CENTERED FINITE-DIFFERENCE on
    the precomputed RUDY demand map (which is already differentiable in
    practice but we treat it as a static field for this routine). The
    'static field' approximation is cheap and the gradient direction is
    what matters here, not the magnitude — the magnitude is rescaled by the
    cong-loss alpha anyway.
    """
    device = all_pos.device
    dtype = all_pos.dtype
    K = twop_endpoints.shape[0]
    grad_out = torch.zeros_like(all_pos)
    if K == 0:
        return grad_out

    # ---- Compute RUDY demand map (no grad — this is just the *field* we
    #      sample; differentiable demand for the loss term is computed
    #      separately and adds its own gradient.) ----
    with torch.no_grad():
        demand_map = _congestion_demand(
            all_pos.detach(), pin_node, pin_net, num_nets,
            canvas_w, canvas_h, grid_col, grid_row,
            grid_v_routes, grid_h_routes, net_weights,
        )  # [grid_row, grid_col]

    bin_w = canvas_w / grid_col
    bin_h = canvas_h / grid_row

    # 2-pin endpoint positions
    p1 = all_pos[twop_endpoints[:, 0]].detach()  # [K, 2]
    p2 = all_pos[twop_endpoints[:, 1]].detach()  # [K, 2]
    seg = p2 - p1                                # [K, 2]
    L = seg.norm(dim=-1).clamp(min=1e-9)         # [K]

    # Sample n_samples+1 points along segment in parallel
    ts = torch.linspace(0.0, 1.0, steps=n_samples + 1, device=device, dtype=dtype)  # [S]
    # samples[k, s, :] = p1[k] + ts[s] * seg[k]
    samples = p1[:, None, :] + ts[None, :, None] * seg[:, None, :]  # [K, S, 2]

    # Bin indices for each sample
    sx = samples[..., 0]  # [K, S]
    sy = samples[..., 1]  # [K, S]
    col = torch.clamp((sx / bin_w).floor().long(), 0, grid_col - 1)
    row = torch.clamp((sy / bin_h).floor().long(), 0, grid_row - 1)
    # Look up demand at each sample
    dmap_flat = demand_map  # [grid_row, grid_col]
    sample_cong = dmap_flat[row, col]  # [K, S]

    # Pick max-congestion sample per net
    max_vals, max_idx = sample_cong.max(dim=1)  # [K], [K]
    # Active: max congestion exceeds threshold
    active = max_vals > cong_thresh  # [K] bool
    if not bool(active.any()):
        return grad_out

    # Virtual cell positions
    K_idx = torch.arange(K, device=device)
    xv = samples[K_idx, max_idx, 0]   # [K]
    yv = samples[K_idx, max_idx, 1]   # [K]

    # ---- Compute ∇C at virtual cell via finite differences on demand_map ----
    # 4-point centered FD on grid:
    #   dC/dx ≈ (C(col+1, row) - C(col-1, row)) / (2 * bin_w)
    #   dC/dy ≈ (C(col, row+1) - C(col, row-1)) / (2 * bin_h)
    # with replicate-padding boundary.
    cv_col = torch.clamp((xv / bin_w).floor().long(), 0, grid_col - 1)
    cv_row = torch.clamp((yv / bin_h).floor().long(), 0, grid_row - 1)
    col_p = torch.clamp(cv_col + 1, 0, grid_col - 1)
    col_m = torch.clamp(cv_col - 1, 0, grid_col - 1)
    row_p = torch.clamp(cv_row + 1, 0, grid_row - 1)
    row_m = torch.clamp(cv_row - 1, 0, grid_row - 1)
    dC_dx = (dmap_flat[cv_row, col_p] - dmap_flat[cv_row, col_m]) / (2.0 * bin_w)  # [K]
    dC_dy = (dmap_flat[row_p, cv_col] - dmap_flat[row_m, cv_col]) / (2.0 * bin_h)  # [K]
    grad_C = torch.stack([dC_dx, dC_dy], dim=-1)  # [K, 2]

    # ---- Project ∇C onto unit normal perpendicular to segment ----
    # segment direction
    seg_norm = seg / L[:, None]                   # [K, 2]
    # normal: rotate by 90°: (a, b) -> (-b, a)
    n_hat = torch.stack([-seg_norm[:, 1], seg_norm[:, 0]], dim=-1)  # [K, 2]
    # orient n_hat to form ACUTE angle with grad_C: flip if dot < 0
    dot_n = (grad_C * n_hat).sum(dim=-1)          # [K]
    sign = torch.where(dot_n >= 0,
                       torch.ones_like(dot_n),
                       -torch.ones_like(dot_n))
    n_hat = n_hat * sign[:, None]
    dot_n = dot_n.abs()                            # always >= 0
    grad_perp = dot_n[:, None] * n_hat            # [K, 2] — projected gradient

    # ---- Distances from endpoints to virtual cell ----
    cv_pos = torch.stack([xv, yv], dim=-1)         # [K, 2]
    d1 = (cv_pos - p1).norm(dim=-1).clamp(min=1e-9)  # [K]
    d2 = (cv_pos - p2).norm(dim=-1).clamp(min=1e-9)  # [K]
    # Weighting: paper says (L / (2 * d_iv)). Drop sign — this is the
    # gradient that PUSHES the endpoint AWAY from congestion, so we apply
    # +grad_perp (which points up the congestion gradient) — but the optimizer
    # MINIMIZES, so adding this to grad pushes positions DOWN the gradient ≡
    # pushes endpoints AWAY from congestion. Correct.
    w1 = (L / (2.0 * d1)).clamp(max=10.0)         # cap weight (paper has no cap; we cap to avoid huge gradients on co-located samples)
    w2 = (L / (2.0 * d2)).clamp(max=10.0)

    # Mask inactive nets to zero
    active_f = active.to(dtype)
    w1 = w1 * active_f
    w2 = w2 * active_f

    # Add to grad_out (scatter via index_add_)
    add1 = w1[:, None] * grad_perp                 # [K, 2]
    add2 = w2[:, None] * grad_perp                 # [K, 2]
    grad_out.index_add_(0, twop_endpoints[:, 0], add1)
    grad_out.index_add_(0, twop_endpoints[:, 1], add2)

    return grad_out


# ---------------------------------------------------------------------------
# Schedule for λ_c
# ---------------------------------------------------------------------------

def _alpha_at(schedule: str, target: float, it: int, total_iters: int, ovf: float) -> float:
    if schedule == 'off':
        return 0.0
    if schedule == 'late_ramp':
        if it < int(total_iters * 0.8):
            return 0.0
        frac = (it - int(total_iters * 0.8)) / max(1, total_iters - int(total_iters * 0.8))
        return target * min(1.0, frac)
    if schedule == 'late_ramp90':
        if it < int(total_iters * 0.9):
            return 0.0
        frac = (it - int(total_iters * 0.9)) / max(1, total_iters - int(total_iters * 0.9))
        return target * min(1.0, frac)
    if schedule == 'late_const':
        return 0.0 if it < int(total_iters * 0.8) else target
    if schedule == 'ovf_gated':
        if ovf >= 0.3:
            return 0.0
        frac = max(0.0, min(1.0, (0.3 - ovf) / 0.2))
        return target * frac
    raise ValueError(f"unknown schedule {schedule}")


# ---------------------------------------------------------------------------
# DCGP placer (single config)
# ---------------------------------------------------------------------------

class DreamPlaceDCGP:
    def __init__(
        self,
        iterations: int = 1500,
        num_bins: int = 64,
        target_density: float = 0.9,
        gamma_factor: float = 4.0,
        lr_frac: float = 0.003,
        density_weight_init: float = 8e-5,
        density_weight_mu_up: float = 1.10,
        density_weight_mu_down: float = 0.99,
        stop_overflow: float = 0.08,
        use_fillers: bool = False,
        center_init: bool = False,
        center_init_std_frac: float = 0.001,
        warm_start: bool = False,
        hpwl_dw_update: bool = True,
        hpwl_ref: float = 350000.0,
        hpwl_upper_pcof: float = 1.05,
        hpwl_lower_pcof: float = 0.95,
        seed: int = 1000,
        device: str = "cpu",
        verbose: bool = True,
        # ----- congestion (Stage A loss) -----
        schedule: str = 'late_ramp',
        congestion_target: float = 0.1,
        congestion_cap: float = 1.0,
        congestion_grid_col: Optional[int] = None,
        congestion_grid_row: Optional[int] = None,
        grid_v_routes: Optional[float] = None,
        grid_h_routes: Optional[float] = None,
        # ----- net moving (Stage B) -----
        net_moving_target: float = 0.05,    # weight for Stage-B grad
        net_moving_thresh: float = 0.7,     # only nets with max-cong > thresh
        net_moving_samples: int = 5,        # samples along each 2-pin segment
        # ----- inflation (Stage C) -----
        inflation_momentum: float = 0.9,    # 0 = no history, 1 = full history
        inflation_max: float = 1.5,         # cap on cumulative inflation ratio
    ):
        self.iterations = iterations
        self.num_bins = num_bins
        self.target_density = target_density
        self.gamma_factor = gamma_factor
        self.lr_frac = lr_frac
        self.dw0 = density_weight_init
        self.mu_up = density_weight_mu_up
        self.mu_down = density_weight_mu_down
        self.stop_overflow = stop_overflow
        self.use_fillers = use_fillers
        self.center_init = center_init
        self.center_init_std_frac = center_init_std_frac
        self.warm_start = warm_start
        self.hpwl_dw_update = hpwl_dw_update
        self.hpwl_ref = hpwl_ref
        self.hpwl_upper_pcof = hpwl_upper_pcof
        self.hpwl_lower_pcof = hpwl_lower_pcof
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.schedule = schedule
        self.congestion_target = congestion_target
        self.congestion_cap = congestion_cap
        self.congestion_grid_col = congestion_grid_col
        self.congestion_grid_row = congestion_grid_row
        self.grid_v_routes = grid_v_routes
        self.grid_h_routes = grid_h_routes
        self.net_moving_target = net_moving_target
        self.net_moving_thresh = net_moving_thresh
        self.net_moving_samples = net_moving_samples
        self.inflation_momentum = inflation_momentum
        self.inflation_max = inflation_max

    def place(self, bench: Benchmark, plc=None, proxy_check_every: int = 50) -> torch.Tensor:
        torch.manual_seed(self.seed)
        dev = torch.device(self.device)
        dtype = torch.float32

        num_macros = bench.num_macros
        num_ports = int(bench.port_positions.shape[0])
        canvas_w = float(bench.canvas_width)
        canvas_h = float(bench.canvas_height)
        cx, cy = canvas_w / 2, canvas_h / 2

        fixed_mac = bench.macro_fixed.to(dev)
        movable_mac = ~fixed_mac

        macro_sizes = bench.macro_sizes.to(dtype=dtype, device=dev)
        init_pos = bench.macro_positions.to(dtype=dtype, device=dev).clone()
        n_mov = int(movable_mac.sum())

        if self.warm_start:
            pass
        elif self.center_init:
            std_x = canvas_w * self.center_init_std_frac
            std_y = canvas_h * self.center_init_std_frac
            rnd = torch.empty(n_mov, 2, device=dev, dtype=dtype)
            rnd[:, 0] = torch.randn(n_mov, device=dev, dtype=dtype) * std_x + cx
            rnd[:, 1] = torch.randn(n_mov, device=dev, dtype=dtype) * std_y + cy
            init_pos[movable_mac] = rnd
        else:
            rnd = torch.rand(n_mov, 2, device=dev, dtype=dtype)
            rnd[:, 0] = rnd[:, 0] * canvas_w * 0.9 + canvas_w * 0.05
            rnd[:, 1] = rnd[:, 1] * canvas_h * 0.9 + canvas_h * 0.05
            init_pos[movable_mac] = rnd

        port_pos = (bench.port_positions.to(dtype=dtype, device=dev)
                    if num_ports else torch.zeros(0, 2, device=dev, dtype=dtype))

        pin_node, pin_net = _build_pin_arrays(bench)
        pin_node = pin_node.to(dev)
        pin_net = pin_net.to(dev)
        num_nets = bench.num_nets
        net_weights = (bench.net_weights.to(dtype=dtype, device=dev)
                       if bench.net_weights.numel() else None)

        # Pre-compute 2-pin nets (Stage B input)
        twop_net_ids, twop_endpoints = _collect_two_pin_nets(bench)
        twop_endpoints = twop_endpoints.to(dev)

        nbx = nby = self.num_bins
        bw = canvas_w / nbx
        bh = canvas_h / nby
        bin_area = bw * bh
        base_gamma = self.gamma_factor * (bw + bh)
        gamma = base_gamma

        cgcol = self.congestion_grid_col or int(bench.grid_cols)
        cgrow = self.congestion_grid_row or int(bench.grid_rows)
        gvr = self.grid_v_routes
        ghr = self.grid_h_routes
        if gvr is None:
            gvr = (canvas_w / cgcol) * float(bench.vroutes_per_micron)
        if ghr is None:
            ghr = (canvas_h / cgrow) * float(bench.hroutes_per_micron)

        with torch.no_grad():
            fixed_map = torch.zeros(nby, nbx, dtype=dtype, device=dev)
            if fixed_mac.any():
                fixed_map = fixed_map + bin_density_map(
                    init_pos[fixed_mac], macro_sizes[fixed_mac],
                    canvas_w, canvas_h, nbx, nby,
                )
            if num_ports:
                port_size = torch.ones_like(port_pos) * min(bw, bh) * 0.01
                fixed_map = fixed_map + bin_density_map(
                    port_pos, port_size, canvas_w, canvas_h, nbx, nby,
                )
            fixed_map = fixed_map.detach()

        mov_idx = movable_mac.nonzero(as_tuple=True)[0]
        fix_idx = fixed_mac.nonzero(as_tuple=True)[0]
        mov_sizes = macro_sizes[mov_idx]
        movable_area = float((mov_sizes[:, 0] * mov_sizes[:, 1]).sum())

        mov_w_np = mov_sizes[:, 0].detach().cpu().numpy()
        if len(mov_w_np) > 0 and self.use_fillers:
            import numpy as _np
            lo = _np.percentile(mov_w_np, 5)
            hi = _np.percentile(mov_w_np, 95)
            trimmed = mov_w_np[(mov_w_np >= lo) & (mov_w_np <= hi)]
            filler_sx = float(trimmed.mean()) if len(trimmed) else float(mov_w_np.mean())
            filler_sy = canvas_h / max(1, int(bench.grid_rows))
            canvas_area = canvas_w * canvas_h
            fixed_area = 0.0
            if fixed_mac.any():
                fm = macro_sizes[fixed_mac]
                fixed_area = float((fm[:, 0] * fm[:, 1]).sum())
            placeable = max(canvas_area - fixed_area, 1e-6)
            total_filler_area = max(
                placeable * self.target_density - movable_area, 0.0,
            )
            n_filler = int(round(total_filler_area / max(filler_sx * filler_sy, 1e-9)))
        else:
            filler_sx = 0.0
            filler_sy = 0.0
            n_filler = 0

        if n_filler > 0:
            filler_sizes = torch.full(
                (n_filler, 2), fill_value=0.0, dtype=dtype, device=dev,
            )
            filler_sizes[:, 0] = filler_sx
            filler_sizes[:, 1] = filler_sy
            all_mov_sizes = torch.cat([mov_sizes, filler_sizes], dim=0)
        else:
            all_mov_sizes = mov_sizes

        if self.verbose:
            print(f"[exp_dcgp] sched={self.schedule} cong_t={self.congestion_target:.3e} "
                  f"netmove_t={self.net_moving_target:.3e} thresh={self.net_moving_thresh} "
                  f"twop_nets={twop_endpoints.shape[0]} fillers={n_filler} grid={cgcol}x{cgrow}")

        # Position constraint with EPS=1e-3 (paranoia for float32 boundary bug)
        EPS = 1e-3
        half_w = all_mov_sizes[:, 0] / 2
        half_h = all_mov_sizes[:, 1] / 2
        xlo = half_w + EPS
        xhi = canvas_w - half_w - EPS
        ylo = half_h + EPS
        yhi = canvas_h - half_h - EPS
        # Guard against degenerate canvas (xlo > xhi can occur if a macro is
        # bigger than the canvas; fall back to centering)
        bad_x = xlo > xhi
        bad_y = ylo > yhi
        if bool(bad_x.any()):
            xlo = torch.where(bad_x, torch.full_like(xlo, cx), xlo)
            xhi = torch.where(bad_x, torch.full_like(xhi, cx), xhi)
        if bool(bad_y.any()):
            ylo = torch.where(bad_y, torch.full_like(ylo, cy), ylo)
            yhi = torch.where(bad_y, torch.full_like(yhi, cy), yhi)

        def constraint(pos: torch.Tensor):
            with torch.no_grad():
                pos[:, 0] = torch.minimum(torch.maximum(pos[:, 0], xlo), xhi)
                pos[:, 1] = torch.minimum(torch.maximum(pos[:, 1], ylo), yhi)

        # Stage C: per-cell historical inflation ratio (movable cells only)
        # (Note: this affects density-map contribution; we pre-multiply
        #  effective sizes for the density term.)
        inflation_history = torch.ones(n_mov + n_filler, dtype=dtype, device=dev)

        def make_all_pos(mov_pos: torch.Tensor) -> torch.Tensor:
            real_mov_pos = mov_pos[:n_mov]
            all_pos = torch.empty(num_macros + num_ports, 2, dtype=dtype, device=dev)
            all_pos[mov_idx] = real_mov_pos
            if fix_idx.numel():
                all_pos[fix_idx] = init_pos[fix_idx]
            if num_ports:
                all_pos[num_macros:] = port_pos
            return all_pos

        target_area = self.target_density * bin_area

        def obj_and_grad(x_data: torch.Tensor, dw: float, alpha_cong: float,
                         alpha_netmove: float, inflation: torch.Tensor):
            x = x_data.detach().clone().requires_grad_(True)
            all_pos = make_all_pos(x)
            wl = wa_wirelength(all_pos, pin_node, pin_net, num_nets, gamma, net_weights)
            # Stage-C inflation: scale node sizes by sqrt(inflation) so that
            # area scales linearly with inflation (preserves total mass per
            # cell but spreads it over a larger footprint => stronger density
            # repulsion in congested regions).
            sqrt_infl = inflation.sqrt().unsqueeze(-1)
            eff_sizes = all_mov_sizes * sqrt_infl
            dmap = bin_density_map(x, eff_sizes, canvas_w, canvas_h, nbx, nby) + fixed_map
            d_elec = electric_potential_energy(dmap, bw, bh)
            d_quad = (F.relu(dmap - target_area) ** 2).sum() / (bin_area * bin_area)
            d = d_elec + d_quad
            total = wl + dw * d
            cong_val = None
            if alpha_cong > 0.0:
                demand = _congestion_demand(
                    all_pos, pin_node, pin_net, num_nets,
                    canvas_w, canvas_h, cgcol, cgrow, gvr, ghr,
                    net_weights,
                )
                cong = congestion_loss(demand, self.congestion_cap)
                total = total + alpha_cong * cong
                cong_val = cong.detach()
            obj_val = total.detach()
            total.backward()
            grad = x.grad.detach().clone()
            wl_val = wl.detach()
            d_val = d.detach()
            dmap_val = dmap.detach()

            # ---- Stage B: virtual-cell net-moving gradient ----
            # Compute on detached `all_pos` snapshot, then index back into x.
            if alpha_netmove > 0.0 and twop_endpoints.shape[0] > 0:
                with torch.no_grad():
                    all_pos_snap = make_all_pos(x.detach())
                    nm_grad_full = _net_moving_grad(
                        all_pos_snap, pin_node, pin_net, num_nets,
                        canvas_w, canvas_h, cgcol, cgrow, gvr, ghr,
                        net_weights, twop_endpoints,
                        cong_thresh=self.net_moving_thresh,
                        n_samples=self.net_moving_samples,
                    )
                    # nm_grad_full is shape [num_macros + num_ports, 2].
                    # Map back to x's index space (mov_idx slot, then filler
                    # slots get zero — fillers aren't real macros, ignored).
                    nm_grad_x = torch.zeros_like(x)
                    nm_grad_x[:n_mov] = nm_grad_full[mov_idx]
                    grad = grad + alpha_netmove * nm_grad_x
            return obj_val, wl_val, d_val, dmap_val, grad, cong_val

        lr0 = self.lr_frac * math.hypot(canvas_w, canvas_h)
        real_init = init_pos[mov_idx].clone()
        if n_filler > 0:
            filler_init = torch.empty(n_filler, 2, dtype=dtype, device=dev)
            filler_init[:, 0] = torch.rand(n_filler, device=dev, dtype=dtype) * canvas_w * 0.9 + canvas_w * 0.05
            filler_init[:, 1] = torch.rand(n_filler, device=dev, dtype=dtype) * canvas_h * 0.9 + canvas_h * 0.05
            v_k = torch.cat([real_init, filler_init], dim=0)
        else:
            v_k = real_init
        constraint(v_k)
        u_k = v_k.clone()
        a_k = 1.0

        density_weight = self.dw0
        alpha_cong = 0.0
        alpha_netmove = 0.0

        obj_k, wl_k, d_k, dmap_k, g_k, _ = obj_and_grad(
            v_k, density_weight, alpha_cong, alpha_netmove, inflation_history,
        )
        v_km1 = v_k - lr0 * g_k
        constraint(v_km1)
        _, _, _, _, g_km1, _ = obj_and_grad(
            v_km1, density_weight, alpha_cong, alpha_netmove, inflation_history,
        )
        alpha_step = (v_k - v_km1).norm(p=2) / (g_k - g_km1).norm(p=2).clamp(min=1e-30)

        prev_ovf = float("inf")
        prev_hpwl = None
        best_pos = v_k.clone()
        best_ovf = float("inf")
        best_proxy = float("inf")
        best_proxy_pos = None
        cur_ovf = 1.0

        t0 = time.time()
        for it in range(self.iterations):
            alpha_cong = _alpha_at(self.schedule, self.congestion_target,
                                   it, self.iterations, cur_ovf)
            alpha_netmove = _alpha_at(self.schedule, self.net_moving_target,
                                      it, self.iterations, cur_ovf)

            # Update inflation history (Stage C): based on dmap_k from prev iter
            if alpha_cong > 0.0:
                with torch.no_grad():
                    bin_w_d = canvas_w / nbx
                    bin_h_d = canvas_h / nby
                    # Look up local density per movable cell (using its bin)
                    pos_now = v_k[:n_mov + n_filler]
                    col_i = torch.clamp((pos_now[:, 0] / bin_w_d).floor().long(), 0, nbx - 1)
                    row_i = torch.clamp((pos_now[:, 1] / bin_h_d).floor().long(), 0, nby - 1)
                    local_dens = dmap_k[row_i, col_i] / max(target_area, 1e-9)
                    # Inflation increases monotonically (DCGP Stage C); momentum
                    # smooths the per-iter signal.
                    new_infl = (self.inflation_momentum * inflation_history
                                + (1.0 - self.inflation_momentum) * local_dens)
                    new_infl = torch.maximum(inflation_history, new_infl)
                    inflation_history = new_infl.clamp(min=1.0, max=self.inflation_max)

            s_k = v_k - v_km1
            y_k = g_k - g_km1
            dot_sy = (s_k * y_k).sum()
            dot_yy = (y_k * y_k).sum().clamp(min=1e-30)
            norm_s = s_k.norm(p=2)
            norm_y = y_k.norm(p=2).clamp(min=1e-30)
            bb_short = dot_sy / dot_yy
            lip = norm_s / norm_y
            if bool((bb_short > 0).item()):
                step = bb_short
            else:
                step = torch.minimum(lip, alpha_step)
            step = step.clamp(min=1e-8)

            a_kp1 = (1.0 + (4.0 * a_k * a_k + 1.0) ** 0.5) / 2.0
            coef = (a_k - 1.0) / a_kp1

            u_kp1 = v_k - step * g_k
            v_kp1 = u_kp1 + coef * (u_kp1 - u_k)
            constraint(v_kp1)

            v_km1 = v_k
            g_km1 = g_k
            u_k = u_kp1
            v_k = v_kp1
            a_k = a_kp1
            alpha_step = step

            obj_k, wl_k, d_k, dmap_k, g_k, cong_v = obj_and_grad(
                v_k, density_weight, alpha_cong, alpha_netmove, inflation_history,
            )

            if torch.isnan(g_k).any() or torch.isnan(v_k).any():
                if self.verbose:
                    print(f"[exp_dcgp] iter {it}: NaN; restoring best and stopping")
                v_k = best_pos.clone()
                break

            with torch.no_grad():
                ovf = compute_overflow(dmap_k, bin_area, self.target_density, movable_area)
            cur_ovf = float(ovf)

            coef_g = 10.0 ** ((ovf - 0.1) * (20.0 / 9.0) - 1.0)
            gamma = base_gamma * coef_g

            if ovf < best_ovf:
                best_ovf = ovf
                best_pos = v_k.clone()

            if plc is not None and ovf < 0.3 and (it % proxy_check_every == 0):
                with torch.no_grad():
                    probe = make_all_pos(v_k).detach().cpu()
                    import numpy as np
                    pos_np = probe[:num_macros].numpy().copy()
                    sizes_np = bench.macro_sizes.numpy()
                    movable_np = bench.get_movable_mask().numpy()
                    np.random.seed(self.seed)
                    pos_np = _legalize_hard(
                        pos_np, sizes_np, bench.num_hard_macros, movable_np,
                        canvas_w, canvas_h, gap=0.0, max_passes=100,
                    )
                    probe_leg = torch.from_numpy(pos_np).float()
                    m = compute_proxy_cost(probe_leg, bench, plc)
                    pr = float(m['proxy_cost'])
                    if pr < best_proxy:
                        best_proxy = pr
                        best_proxy_pos = v_k.clone()

            if ovf < self.stop_overflow:
                if self.verbose:
                    print(f"[exp_dcgp] iter {it}: overflow {ovf:.4f} <= stop; break")
                break

            if self.hpwl_dw_update:
                cur_hpwl = float(wl_k.item())
                if prev_hpwl is None:
                    mu = 1.0
                else:
                    delta = cur_hpwl - prev_hpwl
                    if delta < 0:
                        mu = self.hpwl_upper_pcof * max(0.9999 ** it, 0.98)
                    else:
                        pow_val = self.hpwl_upper_pcof ** (-delta / self.hpwl_ref)
                        pow_val = min(max(pow_val, self.hpwl_lower_pcof), self.hpwl_upper_pcof)
                        mu = self.hpwl_upper_pcof * pow_val
                density_weight *= mu
                prev_hpwl = cur_hpwl
            else:
                if ovf > prev_ovf:
                    density_weight *= self.mu_up
                else:
                    density_weight *= self.mu_down
            prev_ovf = ovf

            if self.verbose and (it % 100 == 0 or it == self.iterations - 1):
                cv = cong_v.item() if cong_v is not None else 0.0
                print(
                    f"[exp_dcgp] it {it:4d}: wl={wl_k.item():.3e} d={d_k.item():.3e} "
                    f"cong={cv:.3e} ac={alpha_cong:.3e} an={alpha_netmove:.3e} "
                    f"lam={density_weight:.3e} ovf={ovf:.4f} infl_max={float(inflation_history.max()):.2f}"
                )

        constraint(v_k)
        import numpy as np
        sizes_np = bench.macro_sizes.numpy()
        movable_np = bench.get_movable_mask().numpy()
        num_hard = bench.num_hard_macros

        def _legalize_and_score(candidate):
            with torch.no_grad():
                probe = make_all_pos(candidate).detach().cpu()
                pos_np = probe[:num_macros].numpy().copy()
                np.random.seed(self.seed)
                pos_np = _legalize_hard(
                    pos_np, sizes_np, num_hard, movable_np,
                    canvas_w, canvas_h, gap=0.0, max_passes=200,
                )
                if _has_hard_overlap(pos_np, sizes_np, num_hard):
                    pos_np = _greedy_slot(
                        pos_np, sizes_np, num_hard, movable_np, canvas_w, canvas_h,
                    )
            if plc is None:
                return pos_np, None
            probe_leg = torch.from_numpy(pos_np).float()
            return pos_np, float(compute_proxy_cost(probe_leg, bench, plc)['proxy_cost'])

        picked_pr = None
        if plc is not None:
            cands = [('v_k', v_k), ('best_ovf', best_pos)]
            if best_proxy_pos is not None:
                cands.append(('best_proxy', best_proxy_pos))
            scored = []
            for name, c in cands:
                pos_np, pr = _legalize_and_score(c)
                scored.append((name, pos_np, pr))
            init_pos_np = bench.macro_positions.numpy().copy()
            init_pr = float(compute_proxy_cost(bench.macro_positions, bench, plc)['proxy_cost'])
            scored.append(('initial', init_pos_np, init_pr))
            name_best, pos_np, picked_pr = min(scored, key=lambda x: x[2])
            if self.verbose:
                for name, _, sc in scored:
                    tag = " <-- picked" if name == name_best else ""
                    print(f"[exp_dcgp] cand {name}: proxy={sc:.4f}{tag}")
        else:
            final_mov = best_pos if best_ovf < prev_ovf else v_k
            pos_np, _ = _legalize_and_score(final_mov)

        if self.verbose:
            tag = f" picked_proxy={picked_pr:.4f}" if picked_pr is not None else ""
            print(f"[exp_dcgp] done {time.time()-t0:.1f}s best_ovf={best_ovf:.4f}{tag}")

        return torch.from_numpy(pos_np).float()


# ---------------------------------------------------------------------------
# Multi-config wrapper (mirrors dreamplace_multi)
# ---------------------------------------------------------------------------

def _fix_touching_edges(pos: torch.Tensor, benchmark: Benchmark, eps: float = 0.005,
                        max_passes: int = 400) -> torch.Tensor:
    import numpy as np
    n_hard = benchmark.num_hard_macros
    canvas_w = benchmark.canvas_width
    canvas_h = benchmark.canvas_height
    sizes = benchmark.macro_sizes.numpy()
    pos_np = pos.detach().cpu().numpy().copy()
    movable = (benchmark.get_movable_mask().numpy()
               if hasattr(benchmark, 'get_movable_mask')
               else np.ones(benchmark.num_macros, dtype=bool))
    best_fixed = None
    best_viols = float('inf')
    for s in [42, 0, 1, 2, 3, 7, 11]:
        np.random.seed(s)
        fixed = _legalize_hard(
            pos_np.copy(), sizes, n_hard, movable, canvas_w, canvas_h,
            gap=eps, max_passes=max_passes,
        )
        ok, viol = validate_placement(torch.tensor(fixed, dtype=torch.float32), benchmark)
        if ok:
            return torch.tensor(fixed, dtype=pos.dtype)
        if len(viol) < best_viols:
            best_viols = len(viol)
            best_fixed = fixed
        if _has_hard_overlap(fixed, sizes, n_hard):
            slot_fixed = _greedy_slot(
                fixed.copy(), sizes, n_hard, movable, canvas_w, canvas_h, gap=eps,
            )
            ok, viol = validate_placement(torch.tensor(slot_fixed, dtype=torch.float32), benchmark)
            if ok:
                return torch.tensor(slot_fixed, dtype=pos.dtype)
            if len(viol) < best_viols:
                best_viols = len(viol)
                best_fixed = slot_fixed
    return torch.tensor(best_fixed if best_fixed is not None else pos_np, dtype=pos.dtype)


def _try_load_plc_for_bench(benchmark: Benchmark):
    try:
        from macro_place.loader import load_benchmark_from_dir
    except ImportError:
        return None
    name = getattr(benchmark, 'name', None)
    if not name:
        return None
    candidates = [
        f"external/MacroPlacement/Testcases/ICCAD04/{name}",
        f"external/MacroPlacement/Flows/NanGate45/{name.replace('_ng45','')}/netlist/output_CT_Grouping",
        f"external/MacroPlacement/Flows/ASAP7/{name.replace('_asap7','')}/netlist/output_CT_Grouping",
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'netlist.pb.txt')):
            try:
                from macro_place.loader import load_benchmark_from_dir
                _, plc = load_benchmark_from_dir(path)
                return plc
            except Exception:
                continue
    return None


# Also import the vanilla DP port to use as a baseline candidate.
DreamPlacePort = _port_mod.DreamPlacePort


class DCGP:
    """Multi-config DCGP-aware placer for the contest leaderboard.

    Default strategy mirrors dreamplace_multi (5 seeds × 2 fillers); each
    config runs DCGP. We additionally include vanilla DreamPlacePort runs
    (alpha=0 / no net-moving) as candidates — the wrapper picks the
    minimum-proxy among:
        - bench's initial placement
        - vanilla DP candidates (per seed × fillers)
        - DCGP candidates (per seed × fillers)
    so we never regress vs the production baseline.
    """

    DEFAULT_SEEDS: List[int] = [1000, 2000, 3000, 4000, 5000]
    DEFAULT_FILLERS: List[bool] = [False, True]

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        fillers: Optional[List[bool]] = None,
        schedule: str = 'late_ramp',
        congestion_target: float = 0.1,
        congestion_cap: float = 1.0,
        net_moving_target: float = 0.05,
        net_moving_thresh: float = 0.7,
        net_moving_samples: int = 5,
        inflation_momentum: float = 0.9,
        inflation_max: float = 1.5,
        include_vanilla_dp: bool = True,
        verbose: bool = False,
    ):
        self.seeds = list(seeds) if seeds is not None else list(self.DEFAULT_SEEDS)
        self.fillers = list(fillers) if fillers is not None else list(self.DEFAULT_FILLERS)
        self.schedule = schedule
        self.congestion_target = congestion_target
        self.congestion_cap = congestion_cap
        self.net_moving_target = net_moving_target
        self.net_moving_thresh = net_moving_thresh
        self.net_moving_samples = net_moving_samples
        self.inflation_momentum = inflation_momentum
        self.inflation_max = inflation_max
        self.include_vanilla_dp = include_vanilla_dp
        self.verbose = verbose

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = _try_load_plc_for_bench(benchmark)
        if self.verbose:
            print(f"[exp_dcgp_multi] plc: {plc is not None}; "
                  f"{len(self.seeds)}x{len(self.fillers)} configs; "
                  f"sched={self.schedule} cong_t={self.congestion_target} "
                  f"netmove_t={self.net_moving_target}")

        # Score-after-legalize helper. Returns (cleaned_pos, score_after_clean).
        # This is the fix for the ibm06 regression: previously we picked best_pos
        # by RAW proxy then legalized once at the end, but legalization can
        # degrade the score — so a candidate that wins on raw score may lose
        # after legalize. Now we legalize THEN score every candidate.
        def _legalize_and_score(pos):
            cleaned = _fix_touching_edges(pos if isinstance(pos, torch.Tensor)
                                          else torch.tensor(pos), benchmark)
            if plc is not None:
                sc = float(compute_proxy_cost(cleaned, benchmark, plc)['proxy_cost'])
            else:
                sc = self._fast_hpwl(cleaned, benchmark)
            return cleaned, sc

        # Initial — raw and cleaned (touching-edge gotcha may make them differ).
        init_raw_pos = benchmark.macro_positions.clone()
        if plc is not None:
            init_raw_score = float(compute_proxy_cost(init_raw_pos, benchmark, plc)['proxy_cost'])
        else:
            init_raw_score = float('inf')
        init_clean_pos, init_clean_score = _legalize_and_score(init_raw_pos)
        # Pick whichever scores lower as the initial candidate.
        if init_clean_score <= init_raw_score:
            best_pos, best_score, best_label = init_clean_pos, init_clean_score, 'initial+fix'
        else:
            best_pos, best_score, best_label = init_clean_pos, init_raw_score, 'initial(raw)'
            # Note: we still RETURN the cleaned version (raw may be invalid),
            # but record the raw score as the bar to beat.
            best_score = init_clean_score
        if self.verbose:
            print(f"[exp_dcgp_multi] initial: raw={init_raw_score:.4f} clean={init_clean_score:.4f}")

        def _consider(pos, label):
            nonlocal best_pos, best_score, best_label
            cleaned, sc = _legalize_and_score(pos)
            if self.verbose:
                print(f"[exp_dcgp_multi] {label}: clean_proxy={sc:.4f}")
            if sc < best_score:
                best_score = sc
                best_pos = cleaned
                best_label = label

        # Vanilla DP (no net-moving, no congestion loss) as baseline candidates
        if self.include_vanilla_dp:
            for seed in self.seeds:
                for use_fillers in self.fillers:
                    vp = DreamPlacePort(verbose=False, seed=seed, use_fillers=use_fillers)
                    pos = vp.place(benchmark, plc=plc, proxy_check_every=50)
                    _consider(pos, f'dp_s{seed}_f{int(use_fillers)}')

        # DCGP candidates
        for seed in self.seeds:
            for use_fillers in self.fillers:
                p = DreamPlaceDCGP(
                    verbose=False, seed=seed, use_fillers=use_fillers,
                    schedule=self.schedule,
                    congestion_target=self.congestion_target,
                    congestion_cap=self.congestion_cap,
                    net_moving_target=self.net_moving_target,
                    net_moving_thresh=self.net_moving_thresh,
                    net_moving_samples=self.net_moving_samples,
                    inflation_momentum=self.inflation_momentum,
                    inflation_max=self.inflation_max,
                )
                pos = p.place(benchmark, plc=plc, proxy_check_every=50)
                _consider(pos, f'dcgp_s{seed}_f{int(use_fillers)}')

        if self.verbose:
            ok, viol = validate_placement(best_pos, benchmark)
            print(f"[exp_dcgp_multi] best: {best_label} score={best_score:.4f} "
                  f"validate={'OK' if ok else f'FAIL ({len(viol)} viol)'}")
        return best_pos

    @staticmethod
    def _fast_hpwl(pos: torch.Tensor, benchmark: Benchmark) -> float:
        if benchmark.num_nets == 0 or len(benchmark.net_nodes) == 0:
            return float((pos - benchmark.macro_positions).abs().mean())
        full = pos.detach().cpu().numpy()
        if benchmark.port_positions.shape[0] > 0:
            import numpy as np
            full = np.concatenate([full, benchmark.port_positions.numpy()], axis=0)
        total = 0.0
        for net in benchmark.net_nodes:
            nids = net.numpy()
            if len(nids) == 0:
                continue
            valid = nids[nids < full.shape[0]]
            if len(valid) == 0:
                continue
            xs = full[valid, 0]
            ys = full[valid, 1]
            total += (xs.max() - xs.min()) + (ys.max() - ys.min())
        return total


def place(benchmark: Benchmark) -> torch.Tensor:
    return DCGP().place(benchmark)


if __name__ == "__main__":
    sys.path.insert(0, _PROJECT_ROOT)
    from macro_place.loader import load_benchmark_from_dir
    bench, plc = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm10")
    p = DCGP(verbose=True, schedule='late_ramp', congestion_target=0.1,
             net_moving_target=0.05, seeds=[1000], fillers=[False])
    pos = p.place(bench)
    print("output:", pos.shape, "score:",
          compute_proxy_cost(pos, bench, plc)['proxy_cost'])
