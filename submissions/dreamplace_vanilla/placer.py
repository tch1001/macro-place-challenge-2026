"""
Vanilla DREAMPlace 4.3 wrapped as a macro_place submission.

Flow:
  1. Convert Benchmark -> Bookshelf in a temp dir.
  2. Run DREAMPlace (inside the prebuilt docker image) on the .aux.
  3. Parse the *.gp.pl output back into center positions.
  4. Return positions tensor the evaluator expects.

DREAMPlace doesn't know about soft macros vs hard macros — it treats
every placeable node as a block. Movable nodes are moved; ports are
fixed terminals at their exact positions.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from macro_place.benchmark import Benchmark

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from bookshelf_writer import write_bookshelf, read_pl_to_positions, SCALE


def _check_overlaps_hard(pos, sizes, num_hard, gap):
    """Return a [num_hard, num_hard] bool matrix of hard-macro overlaps."""
    p = pos[:num_hard]
    s = sizes[:num_hard]
    sep_x = (s[:, 0:1] + s[:, 0:1].T) / 2 + gap
    sep_y = (s[:, 1:2] + s[:, 1:2].T) / 2 + gap
    dx = np.abs(p[:, 0:1] - p[:, 0:1].T)
    dy = np.abs(p[:, 1:2] - p[:, 1:2].T)
    ov = (sep_x - dx > 0) & (sep_y - dy > 0)
    np.fill_diagonal(ov, False)
    return ov


def _greedy_slot(pos, sizes, num_hard, movable, cw, ch, gap=0.0):
    """Place hard macros one-by-one in decreasing-size order into nearest
    free slot relative to their initial (DREAMPlace GP) position. O(N² * R²)
    where R is search radius in step units — fine for N≤a few thousand."""
    half_w = sizes[:, 0] / 2
    half_h = sizes[:, 1] / 2
    sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2 + gap
    sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2 + gap
    origin = pos.copy()
    result = pos.copy()
    # Process hard macros by area descending; soft macros are left at their GP spot.
    hard_order = np.argsort(-(sizes[:num_hard, 0] * sizes[:num_hard, 1]))
    placed_mask = np.zeros(pos.shape[0], dtype=bool)
    placed_mask[num_hard:] = True  # soft macros are not obstacles for hard legalize
    for idx in hard_order:
        if not movable[idx]:
            placed_mask[idx] = True
            continue
        # quick check: is current location free against already-placed hard?
        other = placed_mask.copy(); other[idx] = False
        dx = np.abs(result[idx, 0] - result[:num_hard, 0])
        dy = np.abs(result[idx, 1] - result[:num_hard, 1])
        mask_hard = other[:num_hard]
        if not ((dx < sep_x[idx, :num_hard]) & (dy < sep_y[idx, :num_hard]) & mask_hard).any():
            placed_mask[idx] = True
            continue
        # Spiral search
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


def _legalize_hard(pos_np, sizes_np, num_hard, movable_np, cw, ch,
                   gap=0.0, max_passes=400):
    """Push-apart legalization: only enforce non-overlap among hard macros
    (indices [0, num_hard)). Soft macros may overlap with anything — the
    proxy evaluator doesn't count overlaps there. Operates in-place style.
    """
    half_w = sizes_np[:, 0] / 2
    half_h = sizes_np[:, 1] / 2
    pos = pos_np.copy()
    pos[:, 0] = np.clip(pos[:, 0], half_w, cw - half_w)
    pos[:, 1] = np.clip(pos[:, 1], half_h, ch - half_h)

    # Only iterate over hard-macro sub-slice
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


REPO_ROOT = str(Path(__file__).resolve().parents[2])
DREAMPLACE_SRC = os.path.join(REPO_ROOT, "external", "DREAMPlace_4_3")
DREAMPLACE_INSTALL = os.path.join(DREAMPLACE_SRC, "install")
DOCKER_IMAGE = "dreamplace:4.3"


class DreamPlaceVanilla:
    """Thin wrapper around upstream DREAMPlace."""

    def __init__(
        self,
        num_bins: int = 512,
        iterations: int = 1000,
        target_density: float = 0.9,
        density_weight: float = 8e-5,
        seed: int = 1000,
        use_gpu: bool = False,
        work_dir: Optional[str] = None,
        scale_factor: float = 1.0,
    ):
        self.num_bins = num_bins
        self.iterations = iterations
        self.target_density = target_density
        self.density_weight = density_weight
        self.seed = seed
        self.use_gpu = use_gpu
        self._work_dir = work_dir
        self.scale_factor = scale_factor

    def _config(self, aux_path: str, out_dir: str) -> dict:
        return {
            "aux_input": aux_path,
            "gpu": 1 if self.use_gpu else 0,
            "num_bins_x": self.num_bins,
            "num_bins_y": self.num_bins,
            "global_place_stages": [
                {
                    "num_bins_x": self.num_bins,
                    "num_bins_y": self.num_bins,
                    "iteration": self.iterations,
                    "learning_rate": 0.01,
                    "wirelength": "weighted_average",
                    "optimizer": "nesterov",
                    "Llambda_density_weight_iteration": 1,
                    "Lsub_iteration": 1,
                }
            ],
            "target_density": self.target_density,
            "density_weight": self.density_weight,
            "gamma": 4.0,
            "random_seed": self.seed,
            # NG45 bookshelf is emitted in nm (µm × 1000); DREAMPlace numeric
            # stability requires coordinates in low thousands, not millions.
            # Pass scale_factor=0.001 for NG45 (maps nm→µm internally); leave
            # the default 1.0 for IBMs.
            "scale_factor": self.scale_factor,
            "ignore_net_degree": 100,
            "enable_fillers": 0,  # our benchmark has no stdcells, disable fillers
            "gp_noise_ratio": 0.025,
            "global_place_flag": 1,
            "legalize_flag": 0,  # crashes on fractional-height soft macros; we do our own
            "detailed_place_flag": 0,
            "stop_overflow": 0.07,
            "dtype": "float32",
            "plot_flag": 0,
            "random_center_init_flag": 1,
            "sort_nets_by_degree": 0,
            "num_threads": 8,
            "deterministic_flag": 1,
            "result_dir": out_dir,
        }

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        work = self._work_dir or tempfile.mkdtemp(prefix=f"dreamplace_{benchmark.name}_")
        os.makedirs(work, exist_ok=True)
        design = benchmark.name.replace("/", "_")
        bs_dir = os.path.join(work, "bookshelf")
        out_dir = os.path.join(work, "out")
        os.makedirs(out_dir, exist_ok=True)

        paths = write_bookshelf(benchmark, bs_dir, design)

        # Config paths are relative to the work dir because we mount /work inside docker.
        rel_aux = os.path.relpath(paths.aux, work)
        rel_out = os.path.relpath(out_dir, work)
        cfg = self._config(rel_aux, rel_out)
        cfg_path = os.path.join(work, f"{design}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

        self._run_dreamplace(work, f"{design}.json")

        # Find resulting .pl: DREAMPlace writes {design}.gp.pl (global placement)
        # or {design}.lg.pl after legalize. We prefer legalize if present.
        candidates = [
            os.path.join(out_dir, design, f"{design}.lg.pl"),
            os.path.join(out_dir, design, f"{design}.gp.pl"),
            os.path.join(out_dir, f"{design}.lg.pl"),
            os.path.join(out_dir, f"{design}.gp.pl"),
        ]
        pl_out = next((p for p in candidates if os.path.exists(p)), None)
        if pl_out is None:
            raise RuntimeError(
                f"No DREAMPlace .pl output found. Searched: {candidates}\n"
                f"out_dir tree: {_tree(out_dir)}"
            )

        positions, touched = read_pl_to_positions(pl_out, benchmark)
        print(f"[dreamplace_vanilla] updated {int(touched.sum())} / {benchmark.num_macros} macros from {pl_out}")

        # DREAMPlace's built-in legalizer can't handle our soft-macro shape mix;
        # disabling it would still leave micro-overlaps. Run a quick push-apart.
        pos_np = positions.numpy().copy()
        sizes_np = benchmark.macro_sizes.numpy()
        movable_np = benchmark.get_movable_mask().numpy()
        pos_np = _legalize_hard(
            pos_np, sizes_np, benchmark.num_hard_macros, movable_np,
            benchmark.canvas_width, benchmark.canvas_height,
            gap=0.0, max_passes=200,
        )
        # If residual hard-macro overlaps, fall back to greedy slot assignment.
        if _check_overlaps_hard(pos_np, sizes_np, benchmark.num_hard_macros, gap=0.0).any():
            pos_np = _greedy_slot(
                pos_np, sizes_np, benchmark.num_hard_macros, movable_np,
                benchmark.canvas_width, benchmark.canvas_height, gap=0.0,
            )
        return torch.from_numpy(pos_np).float()

    def _run_dreamplace(self, work_dir: str, cfg_name: str):
        """Run DREAMPlace via docker using the prebuilt image."""
        docker_install = "/dp/install"
        # The baked image ships numpy 1.19.2; matplotlib requires >=1.20.
        # Upgrade in-place on each run (cached in container's conda).
        bootstrap = (
            "pip install -q 'numpy>=1.20,<2.0' matplotlib torch_optimizer "
            "ncg_optimizer 2>/dev/null"
        )
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{DREAMPLACE_SRC}:/dp",
            "-v", f"{work_dir}:/work",
            "-w", "/work",
            "-e", f"PYTHONPATH={docker_install}",
            DOCKER_IMAGE,
            "bash", "-c",
            f"{bootstrap} && python {docker_install}/dreamplace/Placer.py {cfg_name}",
        ]
        print(f"[dreamplace_vanilla] running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print("=== stdout ===")
            print(result.stdout[-2000:])
            print("=== stderr ===")
            print(result.stderr[-2000:])
            raise RuntimeError(f"DREAMPlace failed (exit {result.returncode})")
        print(result.stdout[-2000:])


def _tree(root: str) -> str:
    lines = []
    for dirpath, dirs, files in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        lines.append(f"  {rel}/")
        for fname in files:
            lines.append(f"    {fname}")
    return "\n".join(lines)


if __name__ == "__main__":
    from macro_place.loader import load_benchmark_from_dir
    bench, _ = load_benchmark_from_dir(
        os.path.join(REPO_ROOT, "external/MacroPlacement/Testcases/ICCAD04/ibm01")
    )
    placer = DreamPlaceVanilla(iterations=200, use_gpu=False, work_dir="/tmp/dp_ibm01")
    pos = placer.place(bench)
    print("placement shape:", pos.shape)
    print("first 3 positions:", pos[:3].tolist())
