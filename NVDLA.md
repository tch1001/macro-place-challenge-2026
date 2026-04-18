# NVDLA — unresolved OpenROAD-flow reproduction

Status: **blocked at stage 3_3 (global placement)**. All earlier stages pass. Needs attention if this placement algorithm is to be a Grand-Prize contender, because `nvdla_ng45` is one of the four named Tier-2 benchmarks.

## What currently works

A complete ORFS design directory was built from scratch at `/home/ubuntu/OpenROAD-flow-scripts/flow/designs/nangate45/nvdla/` (the public challenge repo ships no config for NVDLA):

```
designs/nangate45/nvdla/
├── NV_NVDLA_partition_c.v      # from external/MacroPlacement/Flows/NanGate45/nvdla/netlist/
├── constraint.sdc               # hand-written: 4ns core_clock on nvdla_core_clk
├── config.mk                    # DIE_AREA 2353.92 × 2352.56, CORE (10.07,10.08)-(2343.84,2342.48)
├── fakeram45_256x64.lef         # from external/MacroPlacement/Enablements/NanGate45/lef/
└── fakeram45_256x64.lib         # from external/MacroPlacement/Enablements/NanGate45/lib/
```

`orfs_builtin_map` in `scripts/evaluate_with_orfs.py` now has `'nvdla': 'nvdla'` so the eval script wires this design dir into the pipeline correctly.

**Confirmed passing stages**: 1_synth, 2_1_floorplan, 2_2_floorplan_macro (all 128 macros placed via the generic size-bucketed TCL matcher), 2_3_tapcell, 2_4_pdn, 3_1_place_gp_skip_io, 3_2_place_iop.

## What fails

**Stage 3_3 `gpl` (global placement) oscillates indefinitely.** Over 11,700 iterations (~2 hours) the overflow metric bounced between 0.34 and 0.39 and wirelength swung ±5-10% per iteration. Normal convergence ends at ≤0.10 overflow within ~2,000-3,000 iterations. Representative log lines:

```
5750 |   0.3495 |  1.317337e+08 |   -0.46% |  6.02e-08
...
11240 |   0.3756 |  8.653172e+07 |   +3.20% |  1.81e-08
```

The placer is not converging, just thrashing. After 2h I killed it.

## Root cause hypothesis: scrambled macro→net connectivity

NVDLA has exactly **128 macros, all of the same master** (`fakeram45_256x64`). The generic placement matcher I wrote (`_write_generic_size_bucketed_tcl` in `scripts/generate_macro_placement_tcl.py`) groups by `(master_width, master_height)`, so all 128 macros land in **one bucket**. Within the bucket, it sorts `.plc` entries by `(x, y)` and ODB instances by alphabetical name, then pairs by index.

Problem: **alphabetical order of ODB instance names has no relationship to the semantic order the `.plc` file expects.** `.plc` macro #0 is whichever macro happens to sit at the lowest `(x, y)` in the SA baseline; ODB alphabetical-first is whichever synthesized instance name sorts first (e.g. something like `NV_NVDLA_partition_c.u_something.ram_0`). The standard-cell↔macro connectivity is then effectively scrambled — every net that was short in the SA placement might now connect macros on opposite corners of the die.

This is why `mempool_tile` (20 macros, 2 masters, same matcher) still completed but with WNS=-2.17 — scrambling hurt but wasn't catastrophic. For ariane133/136, the name-based matcher (not the generic one) preserves identity per instance, so no scrambling, and both close timing with WNS > 0.

For NVDLA with 128 identical macros, the scrambling is maximal: effectively a random pairing of 128 instances, producing nearly-random wire topology, which GPL cannot untangle.

## Things to try — ranked by expected payoff

### 1. Write a real name-matching matcher for NVDLA (high payoff, medium effort)

The `.plc` names and the post-Yosys ODB names should be related by a deterministic Verilog→synthesis transformation. For NVDLA the `.plc` names will look like `NV_NVDLA_partition_c/...some_path.../ram_instance` (need to peek at `external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping/netlist.pb.txt` to confirm the exact pattern).

The transformation rules observed in this session:
- `/` → `.`
- `[N].` → `_N__` (generate-for indexing)
- `[N]` at end of segment → `_N__`
- `genblk1.<name>` → `genblk1_<name>` (anonymous generate-begin blocks)

Apply these to each `.plc` name to produce an expected ODB name, then have the TCL match by name first and fall back to size-bucket only if the name isn't found. This would preserve identity for the 128 NVDLA macros and give GPL a real starting topology.

Risk: if Yosys mangles names differently from these heuristics, matching silently reverts to the broken generic matcher. Verify by dumping actual ODB names from the first floorplan stage:

```tcl
# dump to a file then diff against .plc names
foreach inst [[ord::get_db_block] getInsts] {
    if { [$inst isBlock] } { puts [$inst getName] }
}
```

### 2. Side-step the problem: use the MacroPlacement DEF directly (low payoff, low effort)

`external/MacroPlacement/Flows/NanGate45/nvdla/def/NV_NVDLA_partition_c_fp.def` is the project's own floorplan DEF with macros already placed. Setting `export FLOORPLAN_DEF = ...` in `config.mk` tells ORFS to load that DEF instead of doing its own macro placement. Gets you *an* OpenROAD baseline for NVDLA — but it's the MacroPlacement baseline, **not** the challenge placement. Only useful as a sanity check that the rest of the flow works end-to-end.

### 3. Loosen density / padding for the scrambled placement (low-medium payoff, low effort)

If keeping the current scrambled pairing but giving GPL more room to untangle helps:

```makefile
# in designs/nangate45/nvdla/config.mk
export PLACE_DENSITY_LB_ADDON = 0.40   # currently 0.20
export CELL_PAD_IN_SITES_GLOBAL_PLACEMENT = 4   # default is 2 on nangate45
export MACRO_BLOCKAGE_HALO = 10        # pushes stdcells farther from macros
```

This is a bandage — if the underlying topology is random, more whitespace just means slower convergence to a worse wirelength, not true convergence. Worth one attempt before putting effort into option (1).

### 4. Cap GPL iterations and accept non-convergence (low value)

```makefile
export GPL_MAX_ITER = 2500
```

OpenROAD will stop at 2500 iters and proceed with whatever placement it has. The result will route but timing will be terrible. Only useful if we need *some* numbers for a table and we accept them as garbage.

### 5. Alternative starting placement (diagnostic)

Replace the benchmark placement with a trivial grid (sqrt(128) ≈ 11×12 grid at spacing 200µm) to establish whether the problem is our specific placement or the design itself. If a simple grid placement converges, it confirms the scrambling hypothesis from (1). If it also diverges, the problem is in the config (clock, floorplan size, PDN interference with placement).

Quick test:

```python
# generate a grid-based macros.tcl: 16 cols x 8 rows, start at (200, 200), spacing 130 x 200
# paste into designs/nangate45/nvdla/macros.tcl, run ./util/docker_shell make DESIGN_CONFIG=... finish
```

## Files to touch when resuming

- `scripts/generate_macro_placement_tcl.py` — add NVDLA name-mapping path (option 1)
- `OpenROAD-flow-scripts/flow/designs/nangate45/nvdla/config.mk` — density/padding knobs (option 3)
- `OpenROAD-flow-scripts/flow/designs/nangate45/nvdla/macros.tcl` — direct grid or DEF-based placement (options 2, 5)

## Runtime budget

The challenge caps evaluation at 1 hour per benchmark. This run blew past that at stage 3_3 alone. Any fix has to get GPL to converge in well under 60 minutes total. The ariane133 full flow took 61 min, ariane136 took 79 min, mempool_tile ~75 min — NVDLA is roughly the size of ariane136, so a healthy NVDLA flow should finish in the same ballpark.

## Why this matters

NVDLA is explicitly named in the Tier 2 / Grand Prize evaluation set in the README. A placement algorithm that doesn't produce convergent NVDLA placements can still win the First-Place-Proxy prize ($20K) based on IBM benchmarks alone — but it cannot win the Grand Prize ($20K). Fixing this is on the critical path to the $20K Grand Prize, not the $20K Proxy prize.
