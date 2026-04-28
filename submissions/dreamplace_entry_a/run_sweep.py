"""Reproducer for dreamplace_entry_a — same interface as dreamplace_port/run_sweep.py
but uses the entry_a placer (DP + experimental SA refinement on top).

Usage:
    python submissions/dreamplace_entry_a/run_sweep.py
    python submissions/dreamplace_entry_a/run_sweep.py --benches ibm10
    python submissions/dreamplace_entry_a/run_sweep.py --sa-moves 0  # disables SA, equivalent to dreamplace_port
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'submissions/dreamplace_port'))
os.chdir(REPO)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

_spec = importlib.util.spec_from_file_location(
    'entry_a_placer',
    os.path.join(REPO, 'submissions/dreamplace_entry_a/placer.py')
)
entry_a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entry_a)
DreamPlaceEntryA = entry_a.DreamPlaceEntryA

DEFAULT_BENCHES = [f'ibm{i:02d}' for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
DEFAULT_SEEDS = [1000, 2000, 3000, 4000, 5000]
ICCAD_DIR = 'external/MacroPlacement/Testcases/ICCAD04'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', nargs='*', default=DEFAULT_BENCHES)
    ap.add_argument('--seeds', nargs='*', type=int, default=DEFAULT_SEEDS)
    ap.add_argument('--fillers', nargs='*', choices=['off', 'on'], default=['off', 'on'])
    ap.add_argument('--sa-moves', type=int, default=200)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    fillers_options = [f == 'on' for f in args.fillers]

    rows = []
    t_total = time.time()
    for n in args.benches:
        bench_dir = os.path.join(ICCAD_DIR, n)
        if not os.path.isdir(bench_dir):
            print(f"{n}  SKIP", flush=True)
            continue
        b, plc = load_benchmark_from_dir(bench_dir)
        init_pr = float(compute_proxy_cost(b.macro_positions, b, plc)['proxy_cost'])

        t0 = time.time()
        cands = [('initial', None, None, init_pr)]
        for seed in args.seeds:
            for uf in fillers_options:
                p = DreamPlaceEntryA(verbose=False, seed=seed, use_fillers=uf, sa_moves=args.sa_moves)
                pos = p.place(b, plc=plc, proxy_check_every=50)
                pr = float(compute_proxy_cost(pos, b, plc)['proxy_cost'])
                cands.append(('placed', seed, uf, pr))

        src, best_seed, best_uf, best_pr = min(cands, key=lambda x: x[3])
        dt = time.time() - t0
        tag = f"s{best_seed},{'F' if best_uf is False else 'T'}" if src == 'placed' else 'init'
        print(f"{n:<6}  best={best_pr:.4f} ({tag})  init={init_pr:.4f}  ({dt:.0f}s)", flush=True)
        rows.append({'bench': n, 'best': best_pr, 'src': src, 'seed': best_seed,
                     'use_fillers': best_uf, 'initial': init_pr})

    if not rows:
        return 1
    avg = sum(r['best'] for r in rows) / len(rows)
    print()
    print(f"=== SUMMARY ({len(rows)} benches, {time.time()-t_total:.0f}s) ===")
    print(f"entry_a avg proxy : {avg:.4f}")
    print(f"vanilla DP 4.3 ref : 1.3452")
    print(f"gap : {(avg-1.3452)/1.3452*100:+.2f}%")

    if args.out:
        with open(args.out, 'w') as f:
            json.dump({'avg': avg, 'rows': rows}, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
