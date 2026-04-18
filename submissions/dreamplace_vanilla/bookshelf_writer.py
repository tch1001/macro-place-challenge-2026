"""
Benchmark -> Bookshelf format converter for DREAMPlace 4.3.

Bookshelf is the native format for ISPD/IBM placement benchmarks and is what
DREAMPlace reads via `aux_input`. We emit 5 files plus the .aux manifest:

  design.aux    header
  design.nodes  list of placeable / fixed blocks with (w, h)
  design.nets   hyperedges with pin offsets
  design.pl     initial placement (x, y, orient, /Fixed flag)
  design.scl    standard-cell rows tiling the core area

All Benchmark coordinates are in microns. Bookshelf requires integer units,
so we multiply by SCALE (default 1000 → nm-ish precision) and round.

The Benchmark loses per-pin connectivity (net_nodes is a set of macro
indices, not pins), so every net-member contributes a single center-offset
pin. This matches the HPWL used by the evaluator.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

from macro_place.benchmark import Benchmark


SCALE = 1000  # microns -> integer units


def _r(x: float) -> int:
    return int(round(float(x) * SCALE))


@dataclass
class BookshelfPaths:
    aux: str
    nodes: str
    nets: str
    pl: str
    scl: str
    wts: str


def write_bookshelf(bench: Benchmark, out_dir: str, design: str | None = None) -> BookshelfPaths:
    """Write the 5 bookshelf files for `bench` into `out_dir`. Returns paths."""
    if design is None:
        design = bench.name
    os.makedirs(out_dir, exist_ok=True)

    paths = BookshelfPaths(
        aux=os.path.join(out_dir, f"{design}.aux"),
        nodes=os.path.join(out_dir, f"{design}.nodes"),
        nets=os.path.join(out_dir, f"{design}.nets"),
        pl=os.path.join(out_dir, f"{design}.pl"),
        scl=os.path.join(out_dir, f"{design}.scl"),
        wts=os.path.join(out_dir, f"{design}.wts"),
    )

    num_ports = int(bench.port_positions.shape[0])
    num_macros = bench.num_macros

    _write_nodes(bench, paths.nodes, num_ports)
    _write_pl(bench, paths.pl, num_ports)
    _write_nets(bench, paths.nets, num_ports)
    _write_scl(bench, paths.scl)
    _write_wts(bench, paths.wts)

    with open(paths.aux, "w") as f:
        f.write(
            f"RowBasedPlacement : {design}.nodes {design}.nets {design}.pl "
            f"{design}.scl {design}.wts\n"
        )

    return paths


def _macro_name(bench: Benchmark, i: int) -> str:
    # Bookshelf names may not contain whitespace or slashes — normalize.
    raw = bench.macro_names[i]
    return raw.replace("/", "_").replace(" ", "_")


def _port_name(i: int) -> str:
    return f"port{i}"


def _write_nodes(bench: Benchmark, path: str, num_ports: int):
    num_macros = bench.num_macros
    num_nodes = num_macros + num_ports
    # Fixed nodes (terminals): ports + any macro_fixed=True
    fixed_macros = bench.macro_fixed.tolist()
    num_terminals = num_ports + sum(fixed_macros)

    with open(path, "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes  :  {num_nodes}\n")
        f.write(f"NumTerminals  :  {num_terminals}\n\n")
        # Movable macros first (bookshelf loaders require terminals listed last)
        for i in range(num_macros):
            if fixed_macros[i]:
                continue
            w, h = bench.macro_sizes[i].tolist()
            f.write(f"\t{_macro_name(bench, i)}\t{_r(w)}\t{_r(h)}\n")
        # Now terminals: fixed macros + ports.
        # Ports are tagged `terminal_NI` (not-in-core-area) so DREAMPlace's
        # place_io does not check them for overlap against movable macros —
        # they're pin-sized boundary points, not real obstacles.
        for i in range(num_macros):
            if not fixed_macros[i]:
                continue
            w, h = bench.macro_sizes[i].tolist()
            f.write(f"\t{_macro_name(bench, i)}\t{_r(w)}\t{_r(h)}\tterminal\n")
        for p in range(num_ports):
            f.write(f"\t{_port_name(p)}\t1\t1\tterminal_NI\n")


def _write_pl(bench: Benchmark, path: str, num_ports: int):
    num_macros = bench.num_macros
    fixed_macros = bench.macro_fixed.tolist()
    with open(path, "w") as f:
        f.write("UCLA pl 1.0\n\n")
        # Movable first
        for i in range(num_macros):
            if fixed_macros[i]:
                continue
            x, y = bench.macro_positions[i].tolist()
            w, h = bench.macro_sizes[i].tolist()
            # bookshelf uses lower-left; our positions are centers
            llx = _r(x - w / 2)
            lly = _r(y - h / 2)
            f.write(f"{_macro_name(bench, i)}\t{llx}\t{lly}\t:\tN\n")
        # Terminals
        for i in range(num_macros):
            if not fixed_macros[i]:
                continue
            x, y = bench.macro_positions[i].tolist()
            w, h = bench.macro_sizes[i].tolist()
            llx = _r(x - w / 2)
            lly = _r(y - h / 2)
            f.write(f"{_macro_name(bench, i)}\t{llx}\t{lly}\t:\tN\t/FIXED\n")
        # Ports: emit /FIXED_NI (IO pin) so DREAMPlace classifies them as
        # IOPin, not as FIXED macros — otherwise the terminal_NI node entry
        # + FIXED pl entry double-counts them and trips sortNodeByPlaceStatus.
        for p in range(num_ports):
            x, y = bench.port_positions[p].tolist()
            f.write(f"{_port_name(p)}\t{_r(x)}\t{_r(y)}\t:\tN\t/FIXED_NI\n")


def _write_nets(bench: Benchmark, path: str, num_ports: int):
    num_macros = bench.num_macros
    nets = bench.net_nodes
    num_nets = len(nets)
    num_pins = sum(int(n.numel()) for n in nets)

    with open(path, "w") as f:
        f.write("UCLA nets 1.0\n\n")
        f.write(f"NumNets : {num_nets}\n")
        f.write(f"NumPins : {num_pins}\n\n")
        for i, net in enumerate(nets):
            idxs = net.tolist()
            f.write(f"NetDegree : {len(idxs)}  n{i}\n")
            for nid in idxs:
                if nid < num_macros:
                    name = _macro_name(bench, nid)
                else:
                    name = _port_name(nid - num_macros)
                # Offset 0,0 — we don't have per-pin offsets in net_nodes.
                f.write(f"\t{name}\tB : 0.000000\t0.000000\n")


def _write_scl(bench: Benchmark, path: str):
    """Emit rows that tile the canvas. Row height = canvas_h / grid_rows."""
    canvas_w = bench.canvas_width
    canvas_h = bench.canvas_height
    rows = max(1, int(bench.grid_rows))
    row_h_float = canvas_h / rows
    row_h = max(1, _r(row_h_float))
    width_int = _r(canvas_w)
    # Site width = 1 unit (like the simple example) so x is fine-grained.
    site_w = 1
    num_sites_x = width_int // site_w

    with open(path, "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {rows}\n\n")
        for r in range(rows):
            y_coord = r * row_h
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate   :   {y_coord}\n")
            f.write(f"  Height       :   {row_h}\n")
            f.write(f"  Sitewidth    :   {site_w}\n")
            f.write(f"  Sitespacing  :   {site_w}\n")
            f.write("  Siteorient   :   1\n")
            f.write("  Sitesymmetry :   1\n")
            f.write(f"  SubrowOrigin :   0\tNumSites  :  {num_sites_x}\n")
            f.write("End\n")


def _write_wts(bench: Benchmark, path: str):
    with open(path, "w") as f:
        f.write("UCLA wts 1.0\n\n")
        for i, w in enumerate(bench.net_weights.tolist()):
            if abs(w - 1.0) > 1e-9:
                f.write(f"n{i}  {w:.6f}\n")


def read_pl_to_positions(pl_path: str, bench: Benchmark) -> Tuple[list, list]:
    """Parse a Bookshelf .pl file and return (updated_positions, touched_mask).

    Positions are centers in microns (undoes SCALE).
    """
    import torch

    name_to_idx = {}
    for i in range(bench.num_macros):
        name_to_idx[_macro_name(bench, i)] = i
    positions = bench.macro_positions.clone()
    touched = torch.zeros(bench.num_macros, dtype=torch.bool)

    with open(pl_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("UCLA"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0]
            if name not in name_to_idx:
                continue
            try:
                llx = float(parts[1]) / SCALE
                lly = float(parts[2]) / SCALE
            except (ValueError, IndexError):
                continue
            i = name_to_idx[name]
            w = float(bench.macro_sizes[i, 0])
            h = float(bench.macro_sizes[i, 1])
            positions[i, 0] = llx + w / 2
            positions[i, 1] = lly + h / 2
            touched[i] = True

    return positions, touched


if __name__ == "__main__":
    # Smoke test
    import sys
    sys.path.insert(0, "/home/degen2/macro-place-challenge-2026")
    from macro_place.benchmark import Benchmark as B

    bench = B.load("/home/degen2/macro-place-challenge-2026/benchmarks/processed/public/ibm01.pt")
    print(bench)
    out = "/tmp/ibm01_bookshelf"
    paths = write_bookshelf(bench, out, "ibm01")
    for attr in ("aux", "nodes", "nets", "pl", "scl", "wts"):
        p = getattr(paths, attr)
        sz = os.path.getsize(p)
        print(f"{attr:6s} {p}  {sz} bytes")
