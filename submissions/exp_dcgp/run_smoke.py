"""Smoke test for exp_dcgp on ibm10.

Reports:
  - initial proxy
  - vanilla DP single-config (seed=1000, fillers=False) proxy + time
  - DCGP single-config (seed=1000, fillers=False) proxy + time
  - validate_placement result for the DCGP output

This is a single-config test (no multi-seed sweep) to keep wallclock down
during iteration. The full multi-config wrapper is in `placer.py:DCGP`.

Usage:
    python submissions/exp_dcgp/run_smoke.py
    python submissions/exp_dcgp/run_smoke.py ibm07
    python submissions/exp_dcgp/run_smoke.py ibm10 0.05  # override netmove_target
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time

# Make project importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Load placer.py directly
_PLACER_PATH = os.path.join(_HERE, 'placer.py')
_spec = importlib.util.spec_from_file_location('exp_dcgp_placer_smoke', _PLACER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DreamPlaceDCGP = _mod.DreamPlaceDCGP
DreamPlacePort = _mod.DreamPlacePort

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from macro_place.utils import validate_placement


def main():
    bench_name = sys.argv[1] if len(sys.argv) > 1 else 'ibm10'
    netmove_target = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    cong_target = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

    bench, plc = load_benchmark_from_dir(
        f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}"
    )

    print(f"=== smoke test: {bench_name} ===")
    print(f"num_macros={bench.num_macros}, num_hard={bench.num_hard_macros}, "
          f"num_nets={bench.num_nets}, canvas={bench.canvas_width}x{bench.canvas_height}")
    print(f"hyperparams: cong_target={cong_target} netmove_target={netmove_target}")

    # Initial
    init_pos = bench.macro_positions.clone()
    init_pr = float(compute_proxy_cost(init_pos, bench, plc)['proxy_cost'])
    print(f"\n[initial] proxy={init_pr:.4f}")

    # Vanilla DP
    t0 = time.time()
    vp = DreamPlacePort(verbose=False, seed=1000, use_fillers=False)
    dp_pos = vp.place(bench, plc=plc, proxy_check_every=50)
    dp_t = time.time() - t0
    dp_pr = float(compute_proxy_cost(dp_pos, bench, plc)['proxy_cost'])
    dp_ok, dp_viol = validate_placement(dp_pos, bench)
    print(f"[DP-only]  proxy={dp_pr:.4f}  time={dp_t:.1f}s  "
          f"valid={'OK' if dp_ok else f'FAIL({len(dp_viol)})'}")

    # DCGP
    t0 = time.time()
    p = DreamPlaceDCGP(
        verbose=True, seed=1000, use_fillers=False,
        schedule='late_ramp',
        congestion_target=cong_target,
        net_moving_target=netmove_target,
    )
    dcgp_pos = p.place(bench, plc=plc, proxy_check_every=50)
    dcgp_t = time.time() - t0
    dcgp_pr = float(compute_proxy_cost(dcgp_pos, bench, plc)['proxy_cost'])
    dcgp_ok, dcgp_viol = validate_placement(dcgp_pos, bench)
    print(f"\n[DCGP]     proxy={dcgp_pr:.4f}  time={dcgp_t:.1f}s  "
          f"valid={'OK' if dcgp_ok else f'FAIL({len(dcgp_viol)})'}")

    print("\n--- summary ---")
    print(f"initial: {init_pr:.4f}")
    print(f"DP-only: {dp_pr:.4f} ({(dp_pr-init_pr)/init_pr*100:+.2f}% vs init)")
    print(f"DCGP:    {dcgp_pr:.4f} ({(dcgp_pr-dp_pr)/dp_pr*100:+.2f}% vs DP, "
          f"{(dcgp_pr-init_pr)/init_pr*100:+.2f}% vs init)")


if __name__ == "__main__":
    main()
