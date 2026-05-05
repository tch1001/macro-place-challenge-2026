"""Smoke test for HybridRouter on ibm10 (DCGP win) + ibm12 (vanilla win).

Reports:
  - DCGP candidate proxy
  - Vanilla DP candidate proxy (if cache or docker available)
  - Final pick (HybridRouter.place output) proxy + validate

Usage:
    python submissions/exp_per_bench_router/run_smoke.py
    python submissions/exp_per_bench_router/run_smoke.py ibm07     # one bench
    python submissions/exp_per_bench_router/run_smoke.py --quick   # just ibm10 quick

Expectations:
  ibm10 — DCGP wins (~1.07). Vanilla DP underperforms; hybrid picks DCGP.
  ibm12 — Vanilla DP wins (~1.33). DCGP gets ~1.55; hybrid picks vanilla.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_PLACER_PATH = os.path.join(_HERE, 'placer.py')
_spec = importlib.util.spec_from_file_location('hybrid_router_placer_smoke',
                                                _PLACER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
HybridRouter = _mod.HybridRouter

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


DEFAULT_BENCHES = ['ibm10', 'ibm12']


def smoke(bench_name: str):
    print(f"\n{'=' * 60}")
    print(f"=== smoke: {bench_name} ===")
    print(f"{'=' * 60}")

    bench, plc = load_benchmark_from_dir(
        os.path.join(_PROJECT_ROOT,
                     f'external/MacroPlacement/Testcases/ICCAD04/{bench_name}')
    )

    init_pr = float(compute_proxy_cost(bench.macro_positions, bench, plc)['proxy_cost'])
    print(f"[initial] proxy={init_pr:.4f}")

    t0 = time.time()
    p = HybridRouter(verbose=True)
    pos = p.place(bench)
    dt = time.time() - t0

    ok, viol = validate_placement(pos, bench)
    pr = float(compute_proxy_cost(pos, bench, plc)['proxy_cost'])

    print(f"\n[hybrid pick] {bench_name}: proxy={pr:.4f} valid="
          f"{'OK' if ok else f'FAIL({len(viol)})'} time={dt:.1f}s")
    if pr < init_pr:
        print(f"            improvement vs initial: "
              f"{(pr - init_pr) / init_pr * 100:+.2f}%")
    return bench_name, pr, ok, dt


def main():
    args = sys.argv[1:]
    if args and args[0] == '--quick':
        benches = ['ibm10']
    elif args:
        benches = args
    else:
        benches = DEFAULT_BENCHES

    results = []
    for b in benches:
        results.append(smoke(b))

    print(f"\n{'=' * 60}")
    print("=== overall ===")
    print(f"{'=' * 60}")
    for name, pr, ok, dt in results:
        print(f"  {name}: proxy={pr:.4f}  valid={'OK' if ok else 'FAIL'}  "
              f"time={dt:.1f}s")


if __name__ == '__main__':
    main()
