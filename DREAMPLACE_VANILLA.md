# DREAMPlace vanilla — OpenROAD-flow evaluation (partial)

Status: placements generated for all 4 NG45 benchmarks. OpenROAD flow was started on `ariane133_ng45` but killed partway through detailed routing (wall-clock budget exhausted). No final WNS/TNS metrics collected yet.

## What was built / debugged

### 1. DREAMPlace installation (new this session)

DREAMPlace wasn't present on the host. Set up end-to-end:

```bash
# Clone + submodules
git clone --depth=1 --recursive https://github.com/limbo018/DREAMPlace.git \
    external/DREAMPlace_4_3

# Build the base docker image (has deps + pytorch+cuda)
docker build -t dreamplace:4.3 external/DREAMPlace_4_3/
# ~10 min

# Compile DREAMPlace inside container with /dp volume mount so output lands on host
docker run --rm -v $(pwd)/external/DREAMPlace_4_3:/dp -w /dp dreamplace:4.3 bash -c \
    "mkdir -p build && cd build && cmake .. -DCMAKE_INSTALL_PREFIX=/dp/install && make"
# ~15 min
```

#### Issue: stripped .so files break `make install`

Docker's `make -j$(nproc)` stripped the built `.so` files. `cmake install` then failed on the first .so with "No valid ELF RPATH or RUNPATH entry exists" when trying to rewrite RPATH.

Workaround: instead of `make install`, manually copy the built artifacts into `install/`:

```bash
docker run --rm -v $(pwd)/external/DREAMPlace_4_3:/dp dreamplace:4.3 bash -c "
    mkdir -p /dp/install
    cp -r /dp/dreamplace /dp/install/
    find /dp/build/dreamplace/ops -name '*.so' -o -name '*.py' | while read f; do
        rel=\${f#/dp/build/}
        dest=/dp/install/\$rel
        mkdir -p \$(dirname \$dest)
        cp \$f \$dest
    done"
```

### 2. Patch needed in `submissions/dreamplace_vanilla/placer.py`

Without this patch, DREAMPlace produces all-NaN placements on every NG45 benchmark. With it, placements converge in ~1 min per benchmark.

```diff
             "gamma": 4.0,
             "random_seed": self.seed,
-            "scale_factor": 1.0,
+            # NG45 bookshelf is emitted in nm (µm × 1000); DREAMPlace numeric
+            # stability requires coordinates in low thousands, not millions.
+            # scale_factor=0.001 maps nm→µm internally.
+            "scale_factor": 0.001,
```

**Why**: `bookshelf_writer.SCALE = 1000` emits coordinates in nm (e.g. canvas 1,433,400 units for ariane133). DREAMPlace's `gamma` (wirelength smoothing) scales with the canvas extent, pushing it to ~1.4 × 10⁷. Gradients overflow to ±inf on iteration 1, yielding NaN forever after.

Diagnostic trail:
- With `scale_factor=1.0` (default): `iteration 0: gamma=1.4e7, wHPWL=5e8` → `iteration 1: Obj=NAN, wHPWL=-INF` and every subsequent iteration is NaN.
- With `scale_factor=0.001`: `iteration 0: gamma=1.4e4, wHPWL=5e5` → converges to `iter 199: Overflow=0.14` as expected.

### 3. One important loader detail

The `.pt` files at `benchmarks/processed/public/*_ng45.pt` have **`net_nodes = []`** — they don't carry connectivity. DREAMPlace then emits `NumNets : 0` in the bookshelf and crashes (`unexpected end of file`).

Fix: load the benchmark via `macro_place.loader.load_benchmark(netlist.pb.txt, initial.plc)` which re-parses connectivity from the protobuf. The evaluate CLI uses this path when invoked with `--ng45`; my driver script (`/tmp/run_dreamplace.py`, temporary) does the same.

## Placements generated

All four saved to `output/dreamplace_placements/` as `.pt` tensors (no NaN):

| Benchmark | Macros | Nets | Canvas (µm) |
|---|---|---|---|
| ariane133_ng45 | 915 | 12,422 | 1433.4 × 1433.4 |
| ariane136_ng45 | 908 | 12,102 | 1446.4 × 1446.4 |
| mempool_tile_ng45 | 581 | 14,534 | 885.4 × 885.4 |
| nvdla_ng45 | 799 | 23,518 | 2127.9 × 2127.9 |

DREAMPlace config used: `iterations=1000`, `num_bins=16`, `target_density=0.9`, `scale_factor=0.001`, CPU-only.

## OpenROAD flow — ariane133 attempt

Launched via `scripts/evaluate_with_orfs.py --benchmark ariane133_ng45 --placement output/dreamplace_placements/ariane133_ng45.pt`. Got through:

- 1_synth ✅
- 2_1..2_4 floorplan + PDN ✅ (all 915 macros placed via the ariane-style name matcher; no PDN channel issue like the baseline had on ariane136)
- 3_1..3_5 placement (GP + resize + DP) ✅
- 4_1 CTS ✅
- 5_1 global route ✅
- 5_2 **detailed route** — in progress at 30% with 2,224 DRC violations when killed

No metrics extracted. The run was consuming ~2 hrs wall-clock at DRT when budget ran out.

## Comparison table (filled in as data becomes available)

Baseline numbers come from the earlier "default placement" runs — see `CHAT_HISTORY.md`.

| Benchmark | Placer | Proxy | WNS (ns) | TNS (ns) | Wire (µm) | Area (µm²) | Fmax (MHz) | Status |
|---|---|---|---|---|---|---|---|---|
| ariane133_ng45 | baseline (SA) | 0.7109 | +0.328 | 0 | 4,622,435 | 4,306,330 | 272.3 | ✅ ORFS done |
| ariane133_ng45 | dreamplace_vanilla | *(pending)* | *(pending)* | *(pending)* | *(pending)* | *(pending)* | *(pending)* | ⏸ killed during DRT |
| ariane136_ng45 | baseline (SA) | 0.7097 | +0.232 | 0 | 6,581,997 | 4,306,330 | 265.4 | ✅ (with 4-macro PDN fix) |
| ariane136_ng45 | dreamplace_vanilla | *(pending)* | — | — | — | — | — | ⏸ not launched |
| mempool_tile_ng45 | baseline (SA) | 0.9610 | -2.175 | -15,140 | 4,777,878 | 3,916,440 | 162.0 | ✅ (size-bucket matcher) |
| mempool_tile_ng45 | dreamplace_vanilla | *(pending)* | — | — | — | — | — | ⏸ not launched |
| nvdla_ng45 | baseline (SA) | — | — | — | — | — | — | ❌ GPL diverged |
| nvdla_ng45 | dreamplace_vanilla | *(pending)* | — | — | — | — | — | ⏸ not launched |

## To resume

1. Placements are already saved in `output/dreamplace_placements/`.
2. Clear stale ORFS state (`rm -rf /home/ubuntu/OpenROAD-flow-scripts/flow/{logs,results,reports,objects}/nangate45/ariane133`) before rerunning, since the previous run left partial artifacts.
3. Per-benchmark ORFS command:
   ```bash
   uv run python scripts/evaluate_with_orfs.py \
       --benchmark ariane133_ng45 \
       --placement output/dreamplace_placements/ariane133_ng45.pt \
       --orfs-root /home/ubuntu/OpenROAD-flow-scripts
   ```
4. Expect 60-90 min/benchmark wall clock. The detailed-routing stage with 2,000+ violations is the long pole — DREAMPlace's non-legalized positions likely produced routing congestion the baseline SA placement didn't have. Compare congestion maps from `5_1_grt.log` vs the baseline run to see if the dreamplace placement is worse for routability.
5. nvdla_ng45 will need its hand-built config (`OpenROAD-flow-scripts/flow/designs/nangate45/nvdla/`) and the `orfs_builtin_map` entry already added to `scripts/evaluate_with_orfs.py`. GPL may or may not converge with dreamplace's output — see `NVDLA.md` for why the baseline didn't.

## Files changed this session

- `submissions/dreamplace_vanilla/placer.py` — `scale_factor: 1.0 → 0.001`. **Required** for DREAMPlace to produce non-NaN placements on any NG45 benchmark.
- `external/DREAMPlace_4_3/` — new (cloned + built)
- `external/DREAMPlace_4_3/install/` — manually assembled from `build/` (cmake install fails due to stripped .so; not tracked in git)
- `output/dreamplace_placements/*.pt` — generated placements
