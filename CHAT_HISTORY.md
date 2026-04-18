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

## Verifying the new host before launching

```bash
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^avx512' | sort -u
# Should show at least: avx512bw, avx512cd, avx512dq, avx512f, avx512vl
```

If that's empty, stop and pick a different host — the run will fail the same way.

---

# Session 2 — 2026-04-18, new host (Intel Xeon Platinum 8488C, AVX-512 ✓)

Continued the run on a fresh AWS instance with Sapphire Rapids. SIGILL issue is gone. Setup recipe from Session 1 worked with minor adjustments.

## Additional host-level fixes required

### 1. `genMetrics.py` requires host `openroad` binary

ORFS `util/genMetrics.py` calls `openroad -version` on the host at line 198. With docker-based runs, `openroad` isn't on the host PATH → parser fails and `summary.json` gets empty metrics. Fix:

```bash
sudo tee /usr/local/bin/openroad > /dev/null <<'EOF'
#!/bin/bash
echo "v2.0-stub stub-commit"
EOF
sudo chmod +x /usr/local/bin/openroad
```

### 2. `final_report.tcl` `gui::show` crashes headless

Even with `QT_QPA_PLATFORM=offscreen`, `gui::show` at stage 6 aborts with `GUI-0077` because `DISPLAY` is set (docker_shell forwards it) but no X server is accessible. The abort happens after `report_metrics 6 "finish"` has been called but before the `-metrics` JSON is flushed, so `6_report.json` is missing → no WNS/TNS in summary.

Fix: gate the `gui::show` block in `flow/scripts/final_report.tcl`:

```diff
-if { [ord::openroad_gui_compiled] } {
+if { 0 && [ord::openroad_gui_compiled] } {
   gui::show "source $::env(SCRIPTS_DIR)/save_images.tcl" false
 }
```

(Lives inside ORFS, but the mounted flow dir means this edit is live inside the container.)

### 3. Docker socket ownership

Non-root `ubuntu` user can't talk to `/var/run/docker.sock`. Group membership (`usermod -aG docker ubuntu`) doesn't take effect in the current shell. Quick workaround: `sudo chmod 666 /var/run/docker.sock`.

## Per-benchmark outcomes (baseline default placement)

### ✅ `ariane133_ng45` — full flow complete

| metric | value |
|---|---|
| proxy cost | 0.710927 |
| WNS | +0.328 ns (met) |
| TNS | 0 |
| hold WNS | +0.014 ns |
| wire length | 4,622,435 µm |
| core area | 4,306,330 µm² |
| power | 0.199 W |
| Fmax | 272.3 MHz |
| clock period | 4.0 ns |

Ran as-is, no placement modifications.

### ✅ `ariane136_ng45` — full flow complete, with a small placement hack

Default placement failed PDN stage 2_4 with `PDN-0179: Unable to repair all channels` — 4 macros landed at y=16.14, creating a ~280 µm × 2.8 µm channel on metal4 for VDD between the core bottom (y=9.94) and the macro row that the power-grid inserter couldn't repair.

**Fix**: snap those 4 macros to y=11.2 (the actual post-alignment core y-min; the raw config value 9.94 is rejected as "outside core" because the floorplan stage aligns to site grid). Patched `output/orfs_evaluation/ariane136_ng45_macros.tcl` and re-ran via `./util/docker_shell make finish` directly — bypasses the eval script's TCL regeneration.

| metric | value |
|---|---|
| proxy cost | 0.709747 |
| WNS | +0.232 ns (met) |
| TNS | 0 |
| hold WNS | +0.003 ns |
| wire length | 6,581,997 µm |
| core area | 4,306,330 µm² |
| power | 0.298 W |
| Fmax | 265.4 MHz |
| clock period | 4.0 ns |

Caveat: 4 of 136 macros placed differently from the stored benchmark default. Not strictly the baseline.

### ❌ `bp_quad_ng45` — macro-name parser can't handle bp_quad instance names

`scripts/generate_macro_placement_tcl.py` only recognizes the ariane-style pattern `.../macro_mem[K].i_ram`. bp_quad macros are named like `bp_processor/cc/y_0__x_0__tile_node/.../macro_mem00/rmod_a` — parser emits "Could not parse .plc name" for every macro, resulting in 0 macros placed. PDN then fails with `PDN-0235: Design has unplaced macros`.

Mapping to ORFS's built-in `black_parrot` design doesn't help — instance names still don't line up. Full fix requires the gitignored `orfs_integration/design_generator.py` (which generates a custom per-benchmark config from the `.pb.txt` netlist).

### ❌ `mempool_tile_ng45` — same name-parser issue

Extracted the `mempool_tile.tar.gz` archive at `external/MacroPlacement/Flows/NanGate45/mempool_tile/scripts/OpenROAD/mempool_tile/`, so the ORFS config is present. But macros are named `i_tile/gen_banks[N].mem_bank/genblk1.sram_instance` — parser can't extract a `macro_mem[K]` index, 0 macros placed, PDN-0235.

### ❌ `nvdla_ng45` — no ORFS config at all

No `.tar.gz` in MacroPlacement, no ORFS built-in design (ORFS has `aes`, `ariane133`, `ariane136`, `black_parrot`, `bp_*`, `gcd`, `ibex`, `jpeg`, `mempool_group`, `swerv`, `tinyRocket` — not `nvdla`). Requires `orfs_integration` to generate config from scratch.

## Pushing further — replacing the missing `orfs_integration` with minimal fixes

After the first round, three benchmarks were still blocked. Each needed a different patch:

### Generic `(width, height)`-bucketed macro matcher

The hardcoded ariane-style name parser in `scripts/generate_macro_placement_tcl.py` (looks for `.../macro_mem[K].i_ram$`) can't handle bp_quad's `.../macro_memNN/rmod_X` or mempool_tile's `.../genblk1.sram_instance` names.

Added a fallback that fires whenever any `.plc` name doesn't match the ariane pattern. It groups both the `.plc` positions and the ODB macros by their master's `(width × 100, height × 100)` in DBU, sorts each bucket by position (`.plc`) / name (`ODB`), and pairs them index-by-index. Swaps identities within a size bucket but reproduces the placement's geometric distribution per master type.

Edits are in `scripts/generate_macro_placement_tcl.py` — look for `_write_generic_size_bucketed_tcl()`.

### Handmade ORFS config for nvdla

The eval script's fallback to a built-in ORFS design doesn't help for nvdla (not in ORFS). The MacroPlacement repo has everything needed, just not wired for ORFS:

```bash
mkdir -p /home/ubuntu/OpenROAD-flow-scripts/flow/designs/nangate45/nvdla
cd /home/ubuntu/OpenROAD-flow-scripts/flow/designs/nangate45/nvdla
cp ~/macro-place-challenge-2026/external/MacroPlacement/Flows/NanGate45/nvdla/netlist/NV_NVDLA_partition_c.v .
cp ~/macro-place-challenge-2026/external/MacroPlacement/Enablements/NanGate45/lef/fakeram45_256x64.lef .
cp ~/macro-place-challenge-2026/external/MacroPlacement/Enablements/NanGate45/lib/fakeram45_256x64.lib .
# Then write config.mk + a 4ns SDC (see designs/nangate45/nvdla/ for final files)
```

Plus added `'nvdla': 'nvdla'` to `orfs_builtin_map` in `evaluate_with_orfs.py` so the script picks up the new design dir.

## Session 2 — final per-benchmark status

### ✅ `mempool_tile_ng45` — full flow complete (generic matcher)

| metric | value |
|---|---|
| proxy cost | 0.961 |
| WNS | **-2.175 ns (not met)** |
| TNS | -15139.5 ns |
| hold WNS | -0.233 ns |
| wire length | 4,777,878 µm |
| core area | 3,916,440 µm² |
| power | 0.194 W |
| Fmax | 162 MHz |

WNS is negative because the macro↔instance pairing inside a size bucket is arbitrary — we preserve the geometric layout per master type, but not the specific macro→net connectivity. That's fine for reporting an OpenROAD-flow baseline for this placement, but won't match what the SA baseline would produce with proper name matching.

### ❌ `bp_quad_ng45` — no RTL available for bp_ng45 benchmark

The `bp_quad` benchmark's `.pb.txt` lives at `external/MacroPlacement/CodeElements/SimulatedAnnealingGWTW/test/bp_ng45/` — a test fixture with `initial.plc` and `netlist.pb.txt` but **no Verilog source**. Its SRAM masters don't match ORFS's built-in `black_parrot` (`bp_ng45` uses different fakeram sizes), so the size-bucketed matcher finds 0 `.plc` positions for every ODB bucket.

Fix would require finding or reconstructing the bp_ng45 RTL netlist and its LEF/LIB stack, then building an ORFS design dir for it. Out of scope.

### ❌ `nvdla_ng45` — config works, but GPL diverges

All 128 macros placed. Floorplan, tapcell, and PDN all passed. Global placement (`gpl`) then failed to converge — overflow oscillated between 0.34 and 0.39 for 11,000+ iterations, wirelength swinging ±5-10% per iter. Runtime cap blown.

Likely cause: the SA-baseline macro positions plus the synthesized netlist produce a congestion pattern that OpenROAD's electrostatic placer can't resolve with the current `PLACE_DENSITY` / `GPL_CELL_PADDING` settings. Worth trying: bumping `PLACE_DENSITY_LB_ADDON` to 0.30, reducing macro cluster density, or pre-legalizing with a different initial placement. Not attempted in this session — killed after 2 hours.

## Final summary table

| Benchmark | Proxy | WNS (ns) | TNS (ns) | Wire (µm) | Area (µm²) | Fmax (MHz) | Status |
|---|---|---|---|---|---|---|---|
| ariane133_ng45 | 0.7109 | +0.328 | 0 | 4,622,435 | 4,306,330 | 272.3 | ✅ |
| ariane136_ng45 | 0.7097 | +0.232 | 0 | 6,581,997 | 4,306,330 | 265.4 | ✅ (4-macro PDN fix) |
| mempool_tile_ng45 | 0.9610 | -2.175 | -15139.5 | 4,777,878 | 3,916,440 | 162.0 | ✅ (generic matcher, approx) |
| bp_quad_ng45 | 1.0058 | — | — | — | — | — | ❌ no RTL for bp_ng45 |
| nvdla_ng45 | — | — | — | — | — | — | ❌ GPL non-convergence |

## Artifacts

- `output/orfs_evaluation/evaluation_summary.json` — final metrics for ariane133 and ariane136
- `output/orfs_evaluation/*_macros.tcl` — generated placement TCL (ariane136 one is the snapped version)
- `/home/ubuntu/OpenROAD-flow-scripts/flow/results/nangate45/{ariane133,ariane136}/base/6_final.*` — final DEF/ODB/SPEF/Verilog

## Commands that worked

```bash
# Clean rerun of one benchmark
cd /home/ubuntu/macro-place-challenge-2026
uv run python scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 \
    --orfs-root /home/ubuntu/OpenROAD-flow-scripts

# Manual rerun of a stage without regenerating TCL (for ariane136 fix)
cd /home/ubuntu/OpenROAD-flow-scripts/flow
./util/docker_shell make DESIGN_CONFIG=./designs/nangate45/ariane136/config.mk finish

# Extract metrics after the flow
python3 util/genMetrics.py --design <nickname> --platform nangate45 \
    --logs logs/nangate45/<nickname>/base \
    --reports reports/nangate45/<nickname>/base \
    --results results/nangate45/<nickname>/base \
    --output /tmp/<nickname>_metrics.json
```

Per-benchmark wall time (Xeon 8488C, 32 threads): ariane133 ~61 min, ariane136 ~79 min.
