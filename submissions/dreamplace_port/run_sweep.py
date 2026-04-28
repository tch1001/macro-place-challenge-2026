"""One-command reproducer for the DREAMPlace port result on the 17 IBM ICCAD04 benchmarks.

Runs `n_seeds` x {fillers=False, fillers=True} configs per bench, plus the provided
initial placement as a final candidate. Picks the min-proxy candidate per bench.

Usage:
    python submissions/dreamplace_port/run_sweep.py
    python submissions/dreamplace_port/run_sweep.py --seeds 1000 2000          # quick: 2 seeds
    python submissions/dreamplace_port/run_sweep.py --benches ibm10 ibm12      # subset
    python submissions/dreamplace_port/run_sweep.py --out results.json
"""
from __future__ import annotations

import argparse
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
import placer as port_mod

DEFAULT_BENCHES = [f'ibm{i:02d}' for i in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
DEFAULT_SEEDS = [1000, 2000, 3000, 4000, 5000]
ICCAD_DIR = 'external/MacroPlacement/Testcases/ICCAD04'


def run(benches, seeds, fillers_options, out_path):
    rows = []
    t_total = time.time()
    for n in benches:
        bench_dir = os.path.join(ICCAD_DIR, n)
        if not os.path.isdir(bench_dir):
            print(f"{n}  SKIP (not found at {bench_dir})", flush=True)
            continue
        b, plc = load_benchmark_from_dir(bench_dir)
        init_pr = float(compute_proxy_cost(b.macro_positions, b, plc)['proxy_cost'])

        t0 = time.time()
        cands = [('initial', None, None, init_pr)]
        for seed in seeds:
            for uf in fillers_options:
                p = port_mod.DreamPlacePort(verbose=False, seed=seed, use_fillers=uf)
                pos = p.place(b, plc=plc, proxy_check_every=50)
                pr = float(compute_proxy_cost(pos, b, plc)['proxy_cost'])
                cands.append(('placed', seed, uf, pr))

        src, best_seed, best_uf, best_pr = min(cands, key=lambda x: x[3])
        dt = time.time() - t0
        tag = f"s{best_seed},{'F' if best_uf is False else 'T'}" if src == 'placed' else 'init'
        print(f"{n:<6}  best={best_pr:.4f} ({tag})  init={init_pr:.4f}  ({dt:.0f}s)", flush=True)
        rows.append({
            'bench': n, 'best': best_pr, 'src': src, 'seed': best_seed, 'use_fillers': best_uf,
            'initial': init_pr, 'all_cands': [{'src': c[0], 'seed': c[1], 'use_fillers': c[2], 'pr': c[3]} for c in cands],
        })

    if not rows:
        print("no benches ran", file=sys.stderr)
        return 1

    avg = sum(r['best'] for r in rows) / len(rows)
    print()
    print(f"=== SUMMARY ({len(rows)} benches, {time.time() - t_total:.0f}s total) ===")
    print(f"port avg proxy        : {avg:.4f}")
    print(f"vanilla DP 4.3 ref    : 1.3452 (per submissions/dreamplace_vanilla/logs/eval_all.log)")
    print(f"gap                   : {(avg - 1.3452) / 1.3452 * 100:+.2f}%")

    if out_path:
        with open(out_path, 'w') as f:
            json.dump({'avg': avg, 'rows': rows}, f, indent=2)
        print(f"saved : {out_path}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--benches', nargs='*', default=DEFAULT_BENCHES, help='bench names (default: 17 IBMs)')
    ap.add_argument('--seeds', nargs='*', type=int, default=DEFAULT_SEEDS, help='random seeds (default: 5)')
    ap.add_argument('--fillers', nargs='*', choices=['off', 'on'], default=['off', 'on'])
    ap.add_argument('--out', default='', help='optional path to save JSON results')
    args = ap.parse_args()
    fillers_options = [f == 'on' for f in args.fillers]
    return run(args.benches, args.seeds, fillers_options, args.out)


if __name__ == '__main__':
    sys.exit(main())
