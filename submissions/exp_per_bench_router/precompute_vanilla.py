"""Precompute vanilla DREAMPlace 4.3 placements for all 17 IBM benchmarks.

Saves results to ./vanilla_initials/<bench>.npz.  HybridRouter loads
these at place() time, so this script only needs to run ONCE per machine
(and on the contest infra it can be run as a one-shot offline step).

Usage:
    python submissions/exp_per_bench_router/precompute_vanilla.py
    python submissions/exp_per_bench_router/precompute_vanilla.py ibm10 ibm12
    python submissions/exp_per_bench_router/precompute_vanilla.py --force

If a cache file already exists for a given bench it is skipped unless
--force is passed.

Each .npz contains:
    positions  — float32 array (num_macros, 2)  center coords in microns
    bench      — bytes  benchmark name (sanity check)
    proxy      — float  proxy cost achieved (for logging)
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_VANILLA_PATH = os.path.join(_PROJECT_ROOT, 'submissions',
                             'dreamplace_vanilla', 'placer.py')
_MULTI_PATH = os.path.join(_PROJECT_ROOT, 'submissions',
                           'dreamplace_multi', 'placer.py')

CACHE_DIR = os.path.join(_HERE, 'vanilla_initials')
ALL_IBMS = [f'ibm{i:02d}' for i in range(1, 19) if i != 5]  # ibm05 not in repo


def _load_module(path, modname):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def precompute_one(bench_name: str, force: bool = False,
                   vanilla_kwargs: dict = None) -> dict:
    """Precompute vanilla DP for one bench. Returns a result dict for logging."""
    from macro_place.loader import load_benchmark_from_dir
    from macro_place.objective import compute_proxy_cost
    from macro_place.utils import validate_placement

    multi_mod = _load_module(_MULTI_PATH, 'multi_for_precompute')
    _fix_touching_edges = multi_mod._fix_touching_edges

    out_path = os.path.join(CACHE_DIR, f"{bench_name}.npz")
    result = {'bench': bench_name, 'status': None, 'proxy': None,
              'time': None, 'path': out_path}

    if os.path.exists(out_path) and not force:
        # Load and report current cached result
        try:
            data = np.load(out_path)
            result['status'] = 'cached'
            result['proxy'] = float(data['proxy']) if 'proxy' in data else None
        except Exception as e:
            result['status'] = f'cache_corrupt: {e}'
        return result

    bench_path = os.path.join(
        _PROJECT_ROOT, 'external/MacroPlacement/Testcases/ICCAD04', bench_name
    )
    if not os.path.isdir(bench_path):
        result['status'] = f'bench_dir_missing: {bench_path}'
        return result

    try:
        bench, plc = load_benchmark_from_dir(bench_path)
    except Exception as e:
        result['status'] = f'load_failed: {type(e).__name__}: {e}'
        return result

    vmod = _load_module(_VANILLA_PATH, 'vp_for_precompute')
    kwargs = dict(seed=1000, use_gpu=False, iterations=1000,
                  target_density=0.9, density_weight=8e-5, num_bins=512)
    if vanilla_kwargs:
        kwargs.update(vanilla_kwargs)

    t0 = time.time()
    try:
        placer = vmod.DreamPlaceVanilla(**kwargs)
        pos = placer.place(bench)
    except Exception as e:
        result['status'] = f'vanilla_failed: {type(e).__name__}: {e}'
        result['time'] = time.time() - t0
        return result

    # Legalize + validate before caching so we know we have something usable.
    try:
        cleaned = _fix_touching_edges(pos, bench)
    except Exception as e:
        result['status'] = f'legalize_failed: {type(e).__name__}: {e}'
        result['time'] = time.time() - t0
        return result

    ok, viol = validate_placement(cleaned, bench)
    if plc is not None:
        proxy = float(compute_proxy_cost(cleaned, bench, plc)['proxy_cost'])
    else:
        proxy = float('nan')

    if not ok:
        # Still cache it, but flag in status — HybridRouter validates again.
        result['status'] = f'invalid({len(viol)}_viol); cached anyway'
    else:
        result['status'] = 'OK'

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(out_path,
             positions=cleaned.detach().cpu().numpy().astype(np.float32),
             bench=np.array(bench_name, dtype='S'),
             proxy=np.array(proxy, dtype=np.float32))

    result['proxy'] = proxy
    result['time'] = time.time() - t0
    return result


def main(argv: List[str]):
    force = False
    args = []
    for a in argv:
        if a == '--force':
            force = True
        else:
            args.append(a)

    benches = args if args else ALL_IBMS

    print(f"=== precompute vanilla DP ===")
    print(f"benches: {benches}")
    print(f"force={force}")
    print(f"cache_dir: {CACHE_DIR}")
    print()

    results = []
    for name in benches:
        print(f"--- {name} ---")
        r = precompute_one(name, force=force)
        results.append(r)
        proxy_s = (f"proxy={r['proxy']:.4f}"
                   if r['proxy'] is not None else 'proxy=N/A')
        time_s = f"time={r['time']:.1f}s" if r['time'] else 'time=cache_hit'
        print(f"    {r['status']}    {proxy_s}    {time_s}")

    print()
    print("=== summary ===")
    okay = [r for r in results
            if r['status'] in ('OK', 'cached')
            or (r['status'] and 'cached anyway' in r['status'])]
    failed = [r for r in results if r not in okay]
    print(f"OK/cached: {len(okay)}/{len(results)}")
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  {r['bench']}: {r['status']}")


if __name__ == '__main__':
    main(sys.argv[1:])
