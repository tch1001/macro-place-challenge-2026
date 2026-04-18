# Session log — OpenROAD baseline run for Partcl/HRT Macro Place Challenge 2026

Date: 2026-04-18
Host: Ubuntu 24.04, AMD EPYC-Milan (Zen 3, **no AVX-512**), 30 GB RAM, no GPU
Goal: clone the challenge repo and run OpenROAD for the baseline (default) macro placement on `ariane133_ng45`.

## Outcome

Setup completed successfully through OpenROAD stage 3 (global placement), but the `openroad/orfs:latest` Docker image SIGILLs during stage 4 (CTS / `repair_timing`) because the prebuilt binary uses instructions this Zen 3 CPU doesn't have (AVX-512 family). The run must continue on a host with AVX-512 (or OpenROAD must be rebuilt from source for this CPU).

---

## What's installed / prepared on this host

- `uv 0.11.7` at `/root/.local/bin/uv`
- `docker 29.1.3` (system docker)
- `openroad/orfs:latest` image pulled (~6.5 GB)
- `xauth` (needed by `util/docker_shell`)
- `/root/macro-place-challenge-2026` — cloned, `uv sync` run, `external/MacroPlacement` submodule initialized
- `/root/OpenROAD-flow-scripts` — cloned `--recursive --depth=1` (~3.2 GB with submodules)

## Code changes made (keep them)

### 1. `scripts/evaluate_with_orfs.py` — stub missing `orfs_integration`

The `orfs_integration/` package is gitignored in the public repo but the script imports it at top-level. Wrapped the import so the script can run for benchmarks that have an existing ORFS config (our case: `ariane133_ng45`).

```python
try:
    from orfs_integration.design_generator import create_orfs_design, ORFSDesign
except ModuleNotFoundError:
    create_orfs_design = None
    ORFSDesign = None
```

### 2. `/root/OpenROAD-flow-scripts/flow/util/docker_shell` — mount host flow over baked flow

`docker_shell` mounts host's flow dir at `/work` but `cd`s into the image's baked `/OpenROAD-flow-scripts/flow`. The evaluator edits host `config.mk`, `macros.tcl`, `macro_place_util.tcl` — none of which take effect unless the host flow shadows the baked one. Added this bind mount:

```diff
     -v "$WORKSPACE:/work:Z"
+    -v "$WORKSPACE:/OpenROAD-flow-scripts/flow:Z"
     --network host
```

This edit lives outside the challenge repo (in ORFS). On a new host, re-apply after cloning ORFS.

---

## The SIGILL

```
Error: cts.tcl, 82 child killed: illegal instruction
make[1]: *** [Makefile:512: do-4_1_cts] Error 1
```

Diagnosis:
- `openroad -version` works inside the image (main-thread code is fine)
- SIGILL hits in a worker thread during `repair_timing`/`detailed_placement`/`check_placement`
- Host CPU flags (no AVX-512): `avx, avx2, bmi1, bmi2, fma, sse4_1, sse4_2, sse4a`
- Prebuilt ORFS binary is compiled with AVX-512 intrinsics

Verified successful pipeline stages before SIGILL:
- ✅ `1_synth` (Yosys synthesis)
- ✅ `2_floorplan` (floorplan + our baseline macro TCL was applied — 133 macros across 25 SRAM groups, `CORE_AREA = (10.07, 9.94, 2062.07, 2109.94)`)
- ✅ `3_1`..`3_3` global placement
- ✅ `3_4` resize, `3_5` hold repair (just before CTS)
- ❌ `4_1` CTS — SIGILL

Proxy cost for the baseline placement (computed before ORFS runs): **0.710927**.

## Hetzner CPU recommendation (for the new host)

Need AVX-512. Pick one:

| Server | CPU | Cores/RAM | Notes |
|---|---|---|---|
| **AX42** | Ryzen 7 7700 (Zen 4) | 8c / 64 GB | Cheapest with AVX-512 |
| **AX52** | Ryzen 7 7700 (Zen 4) | 8c / 64 GB | Slightly better I/O |
| **AX102** | Ryzen 9 7950X3D (Zen 4) | 16c / 128 GB | Closest to competition's EPYC 9655P eval box (16c / 100 GB) |
| AX162-R | EPYC 9454P (Zen 4) | 48c / 256 GB | Overkill but matches eval in spirit |

**Avoid**: Hetzner Cloud CCX/CPX (EPYC Milan, no AVX-512), any AX with EPYC 7xx2/7xx3, and Intel consumer (EX44/EX101) — Intel disabled AVX-512 on consumer chips from 12th gen.

Recommendation: **AX102** if budget allows, otherwise **AX52**.

---

## Full setup recipe on the new host

```bash
# 1. System deps
apt-get update
apt-get install -y docker.io build-essential ca-certificates curl xauth git
systemctl start docker

# 2. uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 3. Clone the challenge repo (with the commit from this session applied)
cd /root
git clone <your-fork-or-this-repo-url> macro-place-challenge-2026
cd macro-place-challenge-2026
git submodule update --init external/MacroPlacement
uv sync

# 4. Clone ORFS and re-apply the docker_shell patch
cd /root
git clone --depth=1 --recursive https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
# Patch flow/util/docker_shell — add the flow bind-mount:
sed -i 's|-v "\$WORKSPACE:/work:Z"|-v "$WORKSPACE:/work:Z"\n    -v "$WORKSPACE:/OpenROAD-flow-scripts/flow:Z"|' \
    /root/OpenROAD-flow-scripts/flow/util/docker_shell

# 5. Pull ORFS image
docker pull openroad/orfs:latest

# 6. Run baseline eval
cd /root/macro-place-challenge-2026
uv run python scripts/evaluate_with_orfs.py \
    --benchmark ariane133_ng45 \
    --orfs-root /root/OpenROAD-flow-scripts
```

Expected runtime: 20–40 min/benchmark.

## Commands to run all 5 NG45 benchmarks

```bash
for bm in ariane133_ng45 ariane136_ng45 bp_quad_ng45 nvdla_ng45 mempool_tile_ng45; do
    uv run python scripts/evaluate_with_orfs.py \
        --benchmark $bm \
        --orfs-root /root/OpenROAD-flow-scripts
done
```

Results are written to `output/orfs_evaluation/evaluation_summary.json`.

## Competition context (for reference)

- **Proxy cost** formula: `1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion` (TILOS MacroPlacement eval)
- **Tier 1**: rank by avg proxy across 17 IBM benchmarks (SA baseline = 2.1251, RePlAce baseline = 1.4578, current leader ~1.317)
- **Tier 2**: top 7 re-evaluated on full OpenROAD flow on NG45 designs for WNS/TNS/Area
- **Grand Prize** ($20K): beat both SA & RePlAce on all three OpenROAD metrics, by widest margin
- **Deadline**: May 21, 2026, 23:59 PT
- **Runtime cap**: 1 hr/benchmark
- **Eval hardware**: AMD EPYC 9655P (16c/100GB) + RTX 6000 Ada 48GB

## Open questions / TODO on new host

- Re-run `ariane133_ng45` — should reach `6_final` and produce WNS/TNS/Area/wirelength
- Run remaining NG45 benchmarks (`ariane136`, `bp_quad`, `nvdla`, `mempool_tile`)
- Baseline numbers from this run become the "default placement" floor to beat with a custom algorithm

## Verifying the new host before launching

```bash
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^avx512' | sort -u
# Should show at least: avx512bw, avx512cd, avx512dq, avx512f, avx512vl
```

If that's empty, stop and pick a different host — the run will fail the same way.
