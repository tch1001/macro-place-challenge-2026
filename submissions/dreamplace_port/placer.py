"""
DREAMPlace port — pure PyTorch, no docker, reads TILOS Benchmark directly.

Based on DREAMPlace 4.3 / ePlace:
  Liao et al., "DREAMPlace 4.0: Timing-Driven Placement ...", TCAD 2023
  Lu et al., "ePlace: Electrostatics-Based Placement ...", TODAES 2015

We implement the *electrostatics-based global placement* core (non-timing):

  min_x   Σ_e w_e · W̃(e; x, γ)   +   λ · D(x)

  W̃  — weighted-average wirelength (smooth HPWL approx, eq. 2 in DP4.0)
        W̃_x = Σxᵢ·e^{xᵢ/γ}/Σe^{xᵢ/γ}  -  Σxᵢ·e^{-xᵢ/γ}/Σe^{-xᵢ/γ}

  D   — electrostatic density potential (ePlace).
        ρ_mn = bin density map (movable contribution + fixed obstacles)
        ψ_mn = FFT-based Poisson solver output (zero-Neumann BC)
        D     = ½ Σ ρ_mn · ψ_mn             (total energy)

  λ   — density weight, adaptive schedule driven by overall overflow.

Optimizer: Adam (lr proportional to canvas scale). The paper uses Nesterov
with BB step size, but Adam is numerically more forgiving and achieves
similar quality for our short runs.

No C++/CUDA extensions. CPU by default (GPU via `device=...`).
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import torch
import torch.nn.functional as F

from macro_place.benchmark import Benchmark


# ---------------------------------------------------------------------------
# Pin / net flattening
# ---------------------------------------------------------------------------

def _build_pin_arrays(bench: Benchmark):
    """Flatten nets into parallel (pin_node_id, pin_net_id) tensors."""
    pin_node, pin_net = [], []
    for i, net in enumerate(bench.net_nodes):
        for nid in net.tolist():
            pin_node.append(nid)
            pin_net.append(i)
    return (
        torch.tensor(pin_node, dtype=torch.long),
        torch.tensor(pin_net, dtype=torch.long),
    )


# ---------------------------------------------------------------------------
# Weighted-average wirelength
# ---------------------------------------------------------------------------

def wa_wirelength(
    all_pos: torch.Tensor,
    pin_node: torch.Tensor,
    pin_net: torch.Tensor,
    num_nets: int,
    gamma: float,
    net_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pin_pos = all_pos[pin_node]
    total = pin_pos.new_tensor(0.0)
    for ax in (0, 1):
        x = pin_pos[:, ax]
        x_max = x.new_zeros(num_nets).scatter_reduce(
            0, pin_net, x, reduce="amax", include_self=False
        )
        x_min = x.new_zeros(num_nets).scatter_reduce(
            0, pin_net, x, reduce="amin", include_self=False
        )
        x_pos = x - x_max[pin_net]
        x_neg = x - x_min[pin_net]
        e_pos = torch.exp(x_pos / gamma)
        e_neg = torch.exp(-x_neg / gamma)
        num_pos = torch.zeros(num_nets, dtype=x.dtype, device=x.device).scatter_add_(0, pin_net, x * e_pos)
        den_pos = torch.zeros(num_nets, dtype=x.dtype, device=x.device).scatter_add_(0, pin_net, e_pos)
        num_neg = torch.zeros(num_nets, dtype=x.dtype, device=x.device).scatter_add_(0, pin_net, x * e_neg)
        den_neg = torch.zeros(num_nets, dtype=x.dtype, device=x.device).scatter_add_(0, pin_net, e_neg)
        wl = num_pos / (den_pos + 1e-30) - num_neg / (den_neg + 1e-30)
        if net_weights is not None:
            wl = wl * net_weights
        total = total + wl.sum()
    return total


# ---------------------------------------------------------------------------
# Bin density map + FFT-based electric potential (ePlace)
# ---------------------------------------------------------------------------

def bin_density_map(
    node_pos: torch.Tensor,
    node_size: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    num_bins_x: int,
    num_bins_y: int,
    stretch: bool = True,
) -> torch.Tensor:
    """Return unnormalized bin density map (area per bin). Differentiable.

    When `stretch=True`, applies DREAMPlace's cell-stretching trick: clamp each
    cell's footprint to at least `bin_size * sqrt(2)` and scale its
    contribution by `ratio = orig_area / stretched_area`. Preserves total
    mass but forces each cell to span ≥2 bins, giving a smooth gradient.
    """
    bw = canvas_w / num_bins_x
    bh = canvas_h / num_bins_y
    if stretch:
        sqrt2 = math.sqrt(2)
        sx = torch.clamp(node_size[:, 0], min=bw * sqrt2)
        sy = torch.clamp(node_size[:, 1], min=bh * sqrt2)
        ratio = (node_size[:, 0] * node_size[:, 1]) / (sx * sy)
    else:
        sx = node_size[:, 0]
        sy = node_size[:, 1]
        ratio = None
    x0 = node_pos[:, 0] - sx / 2
    x1 = node_pos[:, 0] + sx / 2
    y0 = node_pos[:, 1] - sy / 2
    y1 = node_pos[:, 1] + sy / 2

    bin_lx = torch.arange(num_bins_x, dtype=node_pos.dtype, device=node_pos.device) * bw
    bin_rx = bin_lx + bw
    bin_ly = torch.arange(num_bins_y, dtype=node_pos.dtype, device=node_pos.device) * bh
    bin_ry = bin_ly + bh

    ox = torch.clamp(
        torch.minimum(x1[:, None], bin_rx[None, :])
        - torch.maximum(x0[:, None], bin_lx[None, :]),
        min=0.0,
    )
    oy = torch.clamp(
        torch.minimum(y1[:, None], bin_ry[None, :])
        - torch.maximum(y0[:, None], bin_ly[None, :]),
        min=0.0,
    )
    if ratio is not None:
        ox = ox * ratio[:, None]
    return oy.T @ ox  # [bins_y, bins_x]  — area units


def electric_potential_energy(
    density_map: torch.Tensor,
    bin_w: float,
    bin_h: float,
) -> torch.Tensor:
    """Solve Poisson ∇²ψ = -ρ on the bin grid under **zero-Neumann** BC,
    return D = ½ Σ ρ·ψ.

    Implementation: DCT-II-based Poisson solve via even-symmetric extension
    of the density map to [2H, 2W] + periodic FFT. This is equivalent to
    ψ_mn = Σ ρ̂_mn · cos(π m x / H) · cos(π n y / W) / (k_m² + k_n²)
    with k_m = π m / (H · bin_h), matching DREAMPlace's DCT2 solver.

    Versus periodic FFT: eliminates wraparound attraction (cells near one
    edge pulling toward the opposite edge), which was the main source of
    residual density / drift in v4.
    """
    H, W = density_map.shape
    # Even-symmetric extension: [H,W] -> [2H, 2W]
    top = torch.cat([density_map, density_map.flip(-1)], dim=-1)
    bot = torch.cat([density_map.flip(-2), density_map.flip(-2).flip(-1)], dim=-1)
    dm_ext = torch.cat([top, bot], dim=-2)

    rho_hat = torch.fft.rfft2(dm_ext)
    # Frequencies on the extended [2H, 2W] grid: k_m = 2π m / (2H · bin_h)
    # = π m / (H · bin_h). In rfft2, row index runs 0..2H-1 (with negative
    # freqs in second half), col index runs 0..W.
    two_H = 2 * H
    m = torch.arange(two_H, dtype=density_map.dtype, device=density_map.device)
    m = torch.where(m > H, m - two_H, m)
    a = math.pi * m / (H * bin_h)
    n = torch.arange(W + 1, dtype=density_map.dtype, device=density_map.device)
    b = math.pi * n / (W * bin_w)
    k2 = a[:, None] ** 2 + b[None, :] ** 2
    k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)  # DC -> safe; zeroed below
    psi_hat = rho_hat / k2
    # Zero out DC (mean of ψ is arbitrary under pure Neumann)
    mask_dc = torch.zeros_like(psi_hat, dtype=torch.bool)
    mask_dc[0, 0] = True
    psi_hat = torch.where(mask_dc, torch.zeros_like(psi_hat), psi_hat)
    psi_ext = torch.fft.irfft2(psi_hat, s=(two_H, 2 * W))
    psi = psi_ext[:H, :W]
    return 0.5 * (density_map * psi).sum()


# ---------------------------------------------------------------------------
# Overflow metric (for schedule / early stop)
# ---------------------------------------------------------------------------

def compute_overflow(density_map: torch.Tensor, bin_area: float, target_density: float,
                     movable_area: float) -> float:
    """Total excess-over-target area (as fraction of movable-area).

    Matches ePlace's "overflow" definition:
      overflow = Σ max(0, ρ_bin - target·bin_area) / movable_area
    where ρ_bin is in area units (not density fraction).
    """
    target_area_per_bin = target_density * bin_area
    over = F.relu(density_map - target_area_per_bin).sum().item()
    return over / max(movable_area, 1e-9)


# ---------------------------------------------------------------------------
# Main placer
# ---------------------------------------------------------------------------

class DreamPlacePort:
    def __init__(
        self,
        iterations: int = 1500,
        num_bins: int = 64,
        target_density: float = 0.9,
        gamma_factor: float = 4.0,
        lr_frac: float = 0.003,      # lr as fraction of canvas diagonal
        density_weight_init: float = 8e-5,  # DP default; used with HPWL-delta update
        density_weight_mu_up: float = 1.10,
        density_weight_mu_down: float = 0.99,
        stop_overflow: float = 0.08,
        use_fillers: bool = False,
        use_preconditioner: bool = False,
        precond_alpha_init: float = 1.0,
        calibrate_dw: bool = False,
        center_init: bool = False,      # DP-style Gaussian centered init
        center_init_std_frac: float = 0.001,  # DP's default: 0.1% of canvas
        warm_start: bool = False,       # Keep bench.macro_positions as init (skip random/center init)
        hpwl_dw_update: bool = True,    # DP RePlAce HPWL-delta-based dw update
        hpwl_ref: float = 350000.0,
        hpwl_upper_pcof: float = 1.05,
        hpwl_lower_pcof: float = 0.95,
        seed: int = 1000,
        device: str = "cpu",
        verbose: bool = True,
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
        self.use_preconditioner = use_preconditioner
        self.precond_alpha_init = precond_alpha_init
        self.calibrate_dw = calibrate_dw
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

    def place(self, bench: Benchmark, plc=None, proxy_check_every: int = 50) -> torch.Tensor:
        """Optional `plc`: if given, every `proxy_check_every` iterations compute
        the full proxy cost and keep the lowest-proxy position. Port's proxy is
        non-monotonic in iterations (late-iter HPWL refinement re-clusters cells),
        so best-by-proxy is often far better than the final placement."""
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
            # Keep bench.macro_positions (already loaded into init_pos). No-op.
            pass
        elif self.center_init:
            # DP-style Gaussian centered init (BasicPlace.py line 275):
            #   loc=(xl+xh)/2, scale=(xh-xl)*0.001
            std_x = canvas_w * self.center_init_std_frac
            std_y = canvas_h * self.center_init_std_frac
            rnd = torch.empty(n_mov, 2, device=dev, dtype=dtype)
            rnd[:, 0] = torch.randn(n_mov, device=dev, dtype=dtype) * std_x + cx
            rnd[:, 1] = torch.randn(n_mov, device=dev, dtype=dtype) * std_y + cy
            init_pos[movable_mac] = rnd
        else:
            # Uniform random init across inner canvas
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

        nbx = nby = self.num_bins
        bw = canvas_w / nbx
        bh = canvas_h / nby
        bin_area = bw * bh
        # Base gamma = γ_factor · (bw + bh); updated each iter based on overflow
        # per DP (PlaceObj.update_gamma):
        #   coef = 10^((overflow - 0.1) · 20/9 - 1)
        # So gamma shrinks from ~10·base at overflow=1.0 to ~0.1·base at
        # overflow=0.1. Early iters: soft HPWL (broad smoothing), late: tight.
        base_gamma = self.gamma_factor * (bw + bh)
        gamma = base_gamma

        # Fixed / port obstacle map (no grad)
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

        # Filler cells — virtual cells that soak up whitespace and push real
        # cells apart. Count + size from ePlace's formula:
        #   filler_sx = trimmed mean of movable widths (5-95th percentile)
        #   filler_sy = canvas_h / grid_rows   (≈ row height)
        #   total_filler_area = target_density * placeable_area - movable_area
        # Fillers are included in density but NOT in wirelength.
        mov_w_np = mov_sizes[:, 0].detach().cpu().numpy()
        if len(mov_w_np) > 0 and self.use_fillers:
            import numpy as _np
            lo = _np.percentile(mov_w_np, 5)
            hi = _np.percentile(mov_w_np, 95)
            trimmed = mov_w_np[(mov_w_np >= lo) & (mov_w_np <= hi)]
            filler_sx = float(trimmed.mean()) if len(trimmed) else float(mov_w_np.mean())
            filler_sy = canvas_h / max(1, int(bench.grid_rows))
            canvas_area = canvas_w * canvas_h
            # Fixed + port footprint as obstacle area (roughly unplaceable)
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
            print(f"[dreamplace_port] fillers: n={n_filler} size=({filler_sx:.2f}, {filler_sy:.2f})"
                  f"  movable_area={movable_area:.0f}  canvas={canvas_w:.0f}x{canvas_h:.0f}")

        # Per-node preconditioner inputs (ePlace / DP eq 13-16 diagonal Hessian
        # approximation). precond[i] = Σ net_weight over pins of node i
        #                             + α · density_weight · area[i]
        # Applied as grad[i] /= max(precond[i], 1.0). Larger cells and cells
        # with more pins take smaller steps.
        with torch.no_grad():
            pin_count_full = torch.zeros(num_macros + num_ports, dtype=dtype, device=dev)
            if net_weights is not None:
                per_pin_w = net_weights[pin_net]
            else:
                per_pin_w = torch.ones(pin_net.shape[0], dtype=dtype, device=dev)
            pin_count_full.scatter_add_(0, pin_node, per_pin_w)
            real_pin_count = pin_count_full[mov_idx]
            if n_filler > 0:
                all_pin_count = torch.cat(
                    [real_pin_count, torch.zeros(n_filler, dtype=dtype, device=dev)]
                )
            else:
                all_pin_count = real_pin_count
            all_node_area = all_mov_sizes[:, 0] * all_mov_sizes[:, 1]

        half_w = all_mov_sizes[:, 0] / 2
        half_h = all_mov_sizes[:, 1] / 2
        xlo, xhi = half_w, canvas_w - half_w
        ylo, yhi = half_h, canvas_h - half_h

        def constraint(pos: torch.Tensor):
            with torch.no_grad():
                pos[:, 0] = torch.minimum(torch.maximum(pos[:, 0], xlo), xhi)
                pos[:, 1] = torch.minimum(torch.maximum(pos[:, 1], ylo), yhi)

        def make_all_pos(mov_pos: torch.Tensor) -> torch.Tensor:
            # mov_pos may include filler rows at the end — only take the real movable rows.
            real_mov_pos = mov_pos[:n_mov]
            all_pos = torch.empty(num_macros + num_ports, 2, dtype=dtype, device=dev)
            all_pos[mov_idx] = real_mov_pos
            if fix_idx.numel():
                all_pos[fix_idx] = init_pos[fix_idx]
            if num_ports:
                all_pos[num_macros:] = port_pos
            return all_pos

        # Closure: forward + backward at x (leaf tensor). Returns (obj, wl, d, dmap, grad).
        # x has shape [n_mov + n_filler, 2]. Only real movables contribute to WL;
        # all (real + filler) contribute to density.
        target_area = self.target_density * bin_area

        def obj_and_grad(x_data: torch.Tensor, dw: float, alpha: float):
            x = x_data.detach().clone().requires_grad_(True)
            all_pos = make_all_pos(x)
            wl = wa_wirelength(all_pos, pin_node, pin_net, num_nets, gamma, net_weights)
            dmap = bin_density_map(x, all_mov_sizes, canvas_w, canvas_h, nbx, nby) + fixed_map
            d_elec = electric_potential_energy(dmap, bw, bh)
            d_quad = (F.relu(dmap - target_area) ** 2).sum() / (bin_area * bin_area)
            d = d_elec + d_quad
            obj_val = (wl + dw * d).detach()
            (wl + dw * d).backward()
            grad = x.grad.detach().clone()
            if self.use_preconditioner:
                # Diagonal Hessian preconditioner (DP PreconditionOp):
                #   precond[i] = pin_count[i] + α · dw · area[i]
                # Applied to combined (wl + dw·d) gradient.
                precond = (all_pin_count + alpha * dw * all_node_area).clamp(min=1.0)
                grad = grad / precond[:, None]
            return obj_val, wl.detach(), d.detach(), dmap.detach(), grad

        # Nesterov-accelerated gradient with Barzilai-Borwein step size
        # (ePlace Algorithm 2 / DP's step_bb). u_k: major sequence,
        # v_k: reference (where gradient is evaluated).
        lr0 = self.lr_frac * math.hypot(canvas_w, canvas_h)
        real_init = init_pos[mov_idx].clone()
        if n_filler > 0:
            filler_init = torch.empty(n_filler, 2, dtype=dtype, device=dev)
            if self.center_init:
                std_x = canvas_w * self.center_init_std_frac
                std_y = canvas_h * self.center_init_std_frac
                filler_init[:, 0] = torch.randn(n_filler, device=dev, dtype=dtype) * std_x + cx
                filler_init[:, 1] = torch.randn(n_filler, device=dev, dtype=dtype) * std_y + cy
            else:
                filler_init[:, 0] = torch.rand(n_filler, device=dev, dtype=dtype) * canvas_w * 0.9 + canvas_w * 0.05
                filler_init[:, 1] = torch.rand(n_filler, device=dev, dtype=dtype) * canvas_h * 0.9 + canvas_h * 0.05
            v_k = torch.cat([real_init, filler_init], dim=0)
        else:
            v_k = real_init
        constraint(v_k)
        u_k = v_k.clone()
        a_k = 1.0

        # Calibrate initial density_weight via DP's rule:
        #   dw = params_dw · ||∇_x wl||_1  /  ||∇_x D||_1
        # Evaluated at initial (uniform-random) layout. params_dw corresponds
        # to DP's params.density_weight (default 8e-5) — balances the L1 grad
        # norms. Without this, adding the preconditioner explodes because the
        # wl term's per-node gradient is large for pin-heavy cells while the
        # density term is near-flat at t=0.
        precond_alpha = self.precond_alpha_init
        if self.calibrate_dw:
            x_init = v_k.detach().clone().requires_grad_(True)
            wl_init = wa_wirelength(
                make_all_pos(x_init), pin_node, pin_net, num_nets, gamma, net_weights,
            )
            wl_init.backward()
            wl_grad_norm = x_init.grad.abs().sum().item()
            x_init.grad = None
            dmap_init = (
                bin_density_map(x_init, all_mov_sizes, canvas_w, canvas_h, nbx, nby)
                + fixed_map
            )
            d_init = (
                electric_potential_energy(dmap_init, bw, bh)
                + (F.relu(dmap_init - target_area) ** 2).sum() / (bin_area * bin_area)
            )
            d_init.backward()
            density_grad_norm = max(x_init.grad.abs().sum().item(), 1e-30)
            density_weight = float(self.dw0 * wl_grad_norm / density_grad_norm)
            if self.verbose:
                print(f"[dreamplace_port] init: ||wl_grad||={wl_grad_norm:.3e} "
                      f"||d_grad||={density_grad_norm:.3e} dw_init={density_weight:.3e}")
        else:
            density_weight = self.dw0

        obj_k, wl_k, d_k, dmap_k, g_k = obj_and_grad(v_k, density_weight, precond_alpha)

        # Bootstrap v_{k-1}, g_{k-1} with one gradient step
        v_km1 = v_k - lr0 * g_k
        constraint(v_km1)
        _, _, _, _, g_km1 = obj_and_grad(v_km1, density_weight, precond_alpha)
        alpha_k = (v_k - v_km1).norm(p=2) / (g_k - g_km1).norm(p=2).clamp(min=1e-30)

        prev_ovf = float("inf")
        prev_hpwl = None
        best_pos = v_k.clone()
        best_ovf = float("inf")
        best_proxy = float("inf")
        best_proxy_pos = None
        # Lazy import to avoid cost when plc not provided
        if plc is not None:
            from macro_place.objective import compute_proxy_cost

        t0 = time.time()
        for it in range(self.iterations):
            # BB step size from (s_k, y_k)
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
                step = torch.minimum(lip, alpha_k)
            step = step.clamp(min=1e-8)

            # Nesterov coefficient
            a_kp1 = (1.0 + (4.0 * a_k * a_k + 1.0) ** 0.5) / 2.0
            coef = (a_k - 1.0) / a_kp1

            # Step
            u_kp1 = v_k - step * g_k
            v_kp1 = u_kp1 + coef * (u_kp1 - u_k)
            constraint(v_kp1)

            # Advance state
            v_km1 = v_k
            g_km1 = g_k
            u_k = u_kp1
            v_k = v_kp1
            a_k = a_kp1
            alpha_k = step

            # Evaluate at new v_k
            obj_k, wl_k, d_k, dmap_k, g_k = obj_and_grad(v_k, density_weight, precond_alpha)

            # NaN guard: tight overflow targets + high density_weight can explode
            if torch.isnan(g_k).any() or torch.isnan(v_k).any():
                if self.verbose:
                    print(f"[dreamplace_port] iter {it}: NaN detected, restoring best and stopping")
                v_k = best_pos.clone()
                break

            with torch.no_grad():
                ovf = compute_overflow(dmap_k, bin_area, self.target_density, movable_area)

            # Update gamma based on overflow (DP PlaceObj.update_gamma).
            coef = 10.0 ** ((ovf - 0.1) * (20.0 / 9.0) - 1.0)
            gamma = base_gamma * coef

            # Escalate preconditioner strength in the tail (DP: α×=2 every 20
            # iters when overflow < 0.3, capped at 1024). Pushes convergence
            # on the last 20% of the run.
            if self.use_preconditioner and ovf < 0.3 and precond_alpha < 1024 and (it + 1) % 20 == 0:
                precond_alpha *= 2
            if ovf < best_ovf:
                best_ovf = ovf
                best_pos = v_k.clone()

            # Proxy tracking (if plc provided) — the port's proxy is non-monotonic
            # across iterations; keep the lowest-proxy checkpoint.
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
                    print(f"[dreamplace_port] iter {it}: overflow {ovf:.4f} <= stop; break")
                break

            if self.hpwl_dw_update:
                # DP RePlAce-style HPWL-delta-based update (update_density_weight_op_hpwl).
                #   if delta_hpwl < 0:  mu = UPPER_PCOF · max(0.9999^iter, 0.98)
                #   else:               mu = UPPER_PCOF · UPPER_PCOF^(-delta/ref),
                #                            clamp(pow, LOWER_PCOF, UPPER_PCOF)
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

            if self.verbose and (it % 50 == 0 or it == self.iterations - 1):
                print(
                    f"[dreamplace_port] iter {it:4d}: wl={wl_k.item():.3e} "
                    f"d={d_k.item():.3e} λ={density_weight:.3e} "
                    f"ovf={ovf:.4f} step={float(step):.3e}"
                )

        constraint(v_k)
        # Final selection: evaluate all candidates (best-by-overflow, last v_k,
        # best-by-proxy) with the SAME hard-legalization pass used for final
        # return (same seed), then pick the lowest-proxy legalized output.
        import numpy as np
        sizes_np = bench.macro_sizes.numpy()
        movable_np = bench.get_movable_mask().numpy()
        num_hard = bench.num_hard_macros
        def _legalize_and_score(candidate):
            with torch.no_grad():
                probe = make_all_pos(candidate).detach().cpu()
                pos_np = probe[:num_macros].numpy().copy()
                # Deterministic legalize (seed np.random for consistent tie-break
                # across runs; otherwise candidate comparison is noisy).
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

        if plc is not None:
            cands = [('v_k', v_k), ('best_ovf', best_pos)]
            if best_proxy_pos is not None:
                cands.append(('best_proxy', best_proxy_pos))
            scored = []
            for name, c in cands:
                pos_np, pr = _legalize_and_score(c)
                scored.append((name, pos_np, pr))
            # Also consider the untouched initial placement — on some benches
            # (e.g. ibm01/06/08) the provided initial beats our optimized result.
            init_pos_np = bench.macro_positions.numpy().copy()
            init_pr = float(compute_proxy_cost(bench.macro_positions, bench, plc)['proxy_cost'])
            scored.append(('initial', init_pos_np, init_pr))
            name_best, pos_np, best_proxy = min(scored, key=lambda x: x[2])
            if self.verbose:
                for name, _, sc in scored:
                    tag = " <-- picked" if name == name_best else ""
                    print(f"[dreamplace_port] candidate {name}: proxy={sc:.4f}{tag}")
        else:
            final_mov = best_pos if best_ovf < prev_ovf else v_k
            pos_np, _ = _legalize_and_score(final_mov)

        if self.verbose:
            tag = f" best_proxy={best_proxy:.4f}" if plc is not None else ""
            print(f"[dreamplace_port] done in {time.time()-t0:.1f}s  best_ovf={best_ovf:.4f}{tag}")

        return torch.from_numpy(pos_np).float()


# ---------------------------------------------------------------------------
# Legalization helpers (ported from submissions/dreamplace_vanilla/placer.py)
# ---------------------------------------------------------------------------

def _has_hard_overlap(pos_np, sizes_np, num_hard, gap: float = 0.0) -> bool:
    import numpy as np
    p = pos_np[:num_hard]
    s = sizes_np[:num_hard]
    sep_x = (s[:, 0:1] + s[:, 0:1].T) / 2 + gap
    sep_y = (s[:, 1:2] + s[:, 1:2].T) / 2 + gap
    dx = np.abs(p[:, 0:1] - p[:, 0:1].T)
    dy = np.abs(p[:, 1:2] - p[:, 1:2].T)
    ov = (sep_x - dx > 0) & (sep_y - dy > 0)
    np.fill_diagonal(ov, False)
    return ov.any()


def _legalize_hard(pos_np, sizes_np, num_hard, movable_np, cw, ch,
                   gap=0.0, max_passes=400):
    import numpy as np
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2
    pos = pos_np.copy()
    pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)
    h = slice(0, num_hard)
    sizes_h = sizes_np[h]
    sep_x = (sizes_h[:, 0:1] + sizes_h[:, 0:1].T) / 2 + gap
    sep_y = (sizes_h[:, 1:2] + sizes_h[:, 1:2].T) / 2 + gap
    half_w_h = half_w[h]
    half_h_h = half_h[h]
    mov_h = movable_np[:num_hard]
    for _ in range(max_passes):
        p = pos[:num_hard]
        dx = p[:, 0:1] - p[:, 0:1].T
        dy = p[:, 1:2] - p[:, 1:2].T
        ox = np.maximum(0.0, sep_x - np.abs(dx))
        oy = np.maximum(0.0, sep_y - np.abs(dy))
        ov = (ox > 0) & (oy > 0)
        np.fill_diagonal(ov, False)
        pairs = np.argwhere(np.triu(ov, k=1))
        if len(pairs) == 0:
            break
        np.random.shuffle(pairs)
        for i, j in pairs:
            dxi = p[i, 0] - p[j, 0]
            dyi = p[i, 1] - p[j, 1]
            oxi = sep_x[i, j] - abs(dxi)
            oyi = sep_y[i, j] - abs(dyi)
            if oxi <= 0 or oyi <= 0:
                continue
            if oxi <= oyi:
                push = oxi / 2.0 + 1e-4
                sign = 1.0 if dxi >= 0 else -1.0
                if mov_h[i]:
                    p[i, 0] = np.clip(p[i, 0] + sign * push, half_w_h[i], cw - half_w_h[i])
                if mov_h[j]:
                    p[j, 0] = np.clip(p[j, 0] - sign * push, half_w_h[j], cw - half_w_h[j])
            else:
                push = oyi / 2.0 + 1e-4
                sign = 1.0 if dyi >= 0 else -1.0
                if mov_h[i]:
                    p[i, 1] = np.clip(p[i, 1] + sign * push, half_h_h[i], ch - half_h_h[i])
                if mov_h[j]:
                    p[j, 1] = np.clip(p[j, 1] - sign * push, half_h_h[j], ch - half_h_h[j])
    return pos


def _greedy_slot(pos, sizes, num_hard, movable, cw, ch, gap=0.0):
    import numpy as np
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap
    origin = pos.copy()
    result = pos.copy()
    hard_order = np.argsort(-(sizes[:num_hard, 0] * sizes[:num_hard, 1]))
    placed_mask = np.zeros(pos.shape[0], dtype=bool)
    placed_mask[num_hard:] = True
    for idx in hard_order:
        if not movable[idx]:
            placed_mask[idx] = True
            continue
        other = placed_mask.copy(); other[idx] = False
        dx = np.abs(result[idx, 0] - result[:num_hard, 0])
        dy = np.abs(result[idx, 1] - result[:num_hard, 1])
        mask_hard = other[:num_hard]
        if not ((dx < sep_x[idx, :num_hard]) & (dy < sep_y[idx, :num_hard]) & mask_hard).any():
            placed_mask[idx] = True
            continue
        step = max(sizes[idx, 0], sizes[idx, 1]) * 0.35
        best = result[idx].copy(); best_d = float('inf'); found = False
        for r in range(1, 500):
            for dxm in range(-r, r + 1):
                for dym in range(-r, r + 1):
                    if abs(dxm) != r and abs(dym) != r:
                        continue
                    cx = float(np.clip(origin[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx]))
                    cy = float(np.clip(origin[idx, 1] + dym * step, half_h[idx], ch - half_h[idx]))
                    dxc = np.abs(cx - result[:num_hard, 0])
                    dyc = np.abs(cy - result[:num_hard, 1])
                    conf = (dxc < sep_x[idx, :num_hard]) & (dyc < sep_y[idx, :num_hard]) & mask_hard
                    if not conf.any():
                        d = (cx - origin[idx, 0]) ** 2 + (cy - origin[idx, 1]) ** 2
                        if d < best_d:
                            best_d, best = d, np.array([cx, cy])
                        found = True
            if found:
                break
        result[idx] = best
        placed_mask[idx] = True
    return result


def place(benchmark: Benchmark) -> torch.Tensor:
    return DreamPlacePort().place(benchmark)


if __name__ == "__main__":
    sys.path.insert(0, "/home/degen2/macro-place-challenge-2026")
    from macro_place.loader import load_benchmark_from_dir
    bench, _ = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm01")
    p = DreamPlacePort(iterations=1500, verbose=True)
    pos = p.place(bench)
    print("output:", pos.shape, "mean:", pos.mean(0).tolist(), "std:", pos.std(0).tolist())
