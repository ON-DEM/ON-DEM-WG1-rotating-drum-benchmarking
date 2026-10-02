#!/usr/bin/env python3
"""Convert particle VTU snapshots to species-split text files.

The expected VTU layout is the one used by RotatingDrumSphereParticle_*.vtu:

    Points/Position, PointData/Velocity, PointData/AngularVelocity,
    PointData/Radius, PointData/speciesType, PointData/Id

The snapshot time is read from a leading XML comment such as ``<!-- time 0.05-->``.
By default species 0 becomes Particle_A, species 1 becomes Particle_B, and one
is added to every source ID.  This turns the attached file's global 0-based IDs
into the earlier global convention 1..6000 / 6001...

Examples:
    python convert_vtu_particles.py
    python convert_vtu_particles.py --stride 20
    python convert_vtu_particles.py path/to/vtu_folder path/to/output
    python convert_vtu_particles.py "path/to/RotatingDrumSphereParticle_*.vtu" output
    python convert_vtu_particles.py one_snapshot.vtu output

Mass is not stored in the supplied VTU.  Pass both densities to calculate it
from Radius using mass = density * 4/3*pi*radius**3.  Force is not present and
cannot be produced by this converter.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------
# Default paths: running the script without arguments uses these folders.
# -----------------------------------------------------------------------
BASE_DIR = (
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)
INPUT_SUBFOLDER = os.path.join("MercuryDPM", "Raw_Data")
OUTPUT_SUBFOLDER = os.path.join("MercuryDPM", "Spheres")

# Process the first snapshot and then every Nth snapshot after it.
# With a source timestep of 0.002 s, 10 gives an output interval of 0.02 s.
FILE_STRIDE = 50

DEFAULT_INPUT = os.path.join(BASE_DIR, INPUT_SUBFOLDER)
DEFAULT_OUTPUT = os.path.join(BASE_DIR, OUTPUT_SUBFOLDER)

TIME_RE = re.compile(rb"<!--\s*time\s+([-+0-9.eE]+)\s*-->", re.IGNORECASE)
TRAILING_NUMBER_RE = re.compile(r"([-+0-9.eE]+)(?=\.vtu$)", re.IGNORECASE)


@dataclass
class Snapshot:
    path: Path
    time: float
    position: np.ndarray
    velocity: np.ndarray
    angular_velocity: np.ndarray
    radius: np.ndarray
    species: np.ndarray
    particle_id: np.ndarray


def read_time(path: Path, filename_time_scale: float | None = None) -> float:
    with path.open("rb") as fh:
        prefix = fh.read(4096)
    match = TIME_RE.search(prefix)
    if match:
        return float(match.group(1))
    if filename_time_scale is not None:
        match = TRAILING_NUMBER_RE.search(path.name)
        if match:
            return float(match.group(1)) * filename_time_scale
    raise ValueError(
        f"{path}: no '<!-- time VALUE-->' comment found; use "
        "--filename-time-scale if the trailing filename number encodes time"
    )


def discover_files(source: str, pattern: str) -> list[Path]:
    path = Path(source)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.glob(pattern))
    else:
        files = [Path(p) for p in glob.glob(source)]
    files = sorted({p.resolve() for p in files if p.suffix.lower() == ".vtu"})
    if not files:
        raise FileNotFoundError(f"No VTU files found for {source!r}")
    return files


def _array(element: ET.Element, components: int, points: int) -> np.ndarray:
    if element.get("format", "ascii").lower() != "ascii":
        raise ValueError(f"DataArray {element.get('Name')!r} is not ASCII")
    values = np.fromstring(element.text or "", sep=" ")
    expected = points * components
    if values.size != expected:
        raise ValueError(
            f"DataArray {element.get('Name')!r}: expected {expected} values, "
            f"found {values.size}"
        )
    return values.reshape(points, components) if components > 1 else values


def read_snapshot(path: Path, time: float) -> Snapshot:
    root = ET.parse(path).getroot()
    piece = root.find("./UnstructuredGrid/Piece")
    if piece is None:
        raise ValueError(f"{path}: UnstructuredGrid/Piece not found")
    points = int(piece.get("NumberOfPoints", "-1"))
    if points < 0:
        raise ValueError(f"{path}: invalid NumberOfPoints")

    named = {
        (item.get("Name") or "").lower(): item
        for item in piece.findall(".//DataArray")
        if item.get("Name")
    }

    def need(name: str, components: int) -> np.ndarray:
        item = named.get(name.lower())
        if item is None:
            raise ValueError(f"{path}: required DataArray {name!r} not found")
        actual = int(item.get("NumberOfComponents", "1"))
        if actual != components:
            raise ValueError(
                f"{path}: DataArray {name!r} has {actual} components; "
                f"expected {components}"
            )
        return _array(item, components, points)

    position = need("Position", 3)
    velocity = need("Velocity", 3)
    angular_velocity = need("AngularVelocity", 3)
    radius = need("Radius", 1)
    species_raw = need("speciesType", 1)
    ids_raw = need("Id", 1)

    if not np.allclose(species_raw, np.rint(species_raw)):
        raise ValueError(f"{path}: speciesType contains non-integer values")
    if not np.allclose(ids_raw, np.rint(ids_raw)):
        raise ValueError(f"{path}: Id contains non-integer values")

    species = np.rint(species_raw).astype(np.int64)
    particle_id = np.rint(ids_raw).astype(np.int64)
    if np.unique(particle_id).size != points:
        raise ValueError(f"{path}: particle IDs are not unique within the snapshot")

    return Snapshot(
        path=path,
        time=time,
        position=position,
        velocity=velocity,
        angular_velocity=angular_velocity,
        radius=radius,
        species=species,
        particle_id=particle_id,
    )


def parse_species_map(items: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in items:
        try:
            raw_value, folder = item.split(":", 1)
            value = int(raw_value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid species mapping {item!r}; expected INTEGER:FOLDER"
            ) from exc
        if not folder or Path(folder).name != folder:
            raise argparse.ArgumentTypeError(
                f"Invalid output folder {folder!r} in species mapping"
            )
        if value in result:
            raise argparse.ArgumentTypeError(f"Species value {value} is mapped twice")
        result[value] = folder
    return result


def write_rows(
    fh,
    ids: np.ndarray,
    time: float,
    values: np.ndarray,
    time_decimals: int,
    value_decimals: int,
) -> None:
    count = ids.size
    if count == 0:
        return
    values_2d = values.reshape(count, -1)
    table = np.column_stack((ids, np.full(count, time), values_2d))
    formats = ["%d", f"%.{time_decimals}f"] + [
        f"%.{value_decimals}f"
    ] * values_2d.shape[1]
    np.savetxt(fh, table, fmt=formats, delimiter=" ")


def convert(args: argparse.Namespace) -> None:
    files = discover_files(args.source, args.pattern)
    timed_files = sorted(
        ((read_time(path, args.filename_time_scale), path) for path in files),
        key=lambda item: (item[0], item[1].name),
    )
    total_files = len(timed_files)
    timed_files = timed_files[:: args.stride]
    species_map = parse_species_map(args.species_map)

    if (args.density_a is None) != (args.density_b is None):
        raise ValueError("Provide both --density-a and --density-b, or neither")
    densities = None
    if args.density_a is not None:
        densities = {0: args.density_a, 1: args.density_b}
        unknown = set(species_map) - set(densities)
        if unknown:
            raise ValueError(
                "Mass calculation has no density option for species values "
                + ", ".join(map(str, sorted(unknown)))
            )

    print(
        f"Found {total_files} VTU snapshot(s); processing "
        f"{len(timed_files)} with stride {args.stride}"
    )
    if args.inspect_only:
        for time, path in timed_files:
            snapshot = read_snapshot(path, time)
            counts = {
                int(value): int(np.count_nonzero(snapshot.species == value))
                for value in np.unique(snapshot.species)
            }
            print(
                f"  {path.name}: time={time:g}, points={snapshot.particle_id.size}, "
                f"id={snapshot.particle_id.min()}..{snapshot.particle_id.max()}, "
                f"species={counts}"
            )
        return

    output = Path(args.output_dir)
    specs = {
        "positions.txt": ("# id time x y z", "position", 6),
        "velocities.txt": ("# id time vx vy vz", "velocity", 6),
        "angular_velocities.txt": (
            "# id time wx wy wz",
            "angular_velocity",
            6,
        ),
        "radii.txt": ("# id time radius", "radius", 6),
    }
    if densities is not None:
        specs["masses.txt"] = ("# id time mass", "mass", 12)

    with ExitStack() as stack:
        handles = {}
        for species_value, folder in species_map.items():
            species_dir = output / folder
            species_dir.mkdir(parents=True, exist_ok=True)
            for filename, (header, _field, _decimals) in specs.items():
                fh = stack.enter_context(
                    (species_dir / filename).open("w", newline="\n", encoding="utf-8")
                )
                fh.write(header + "\n")
                handles[(species_value, filename)] = fh

        for time, path in timed_files:
            snapshot = read_snapshot(path, time)
            present = set(map(int, np.unique(snapshot.species)))
            unmapped = present - set(species_map)
            if unmapped:
                raise ValueError(
                    f"{path}: unmapped speciesType values {sorted(unmapped)}; "
                    "extend --species-map"
                )

            summary = []
            for species_value, folder in species_map.items():
                selected = np.flatnonzero(snapshot.species == species_value)
                order = np.argsort(snapshot.particle_id[selected], kind="stable")
                selected = selected[order]
                ids = snapshot.particle_id[selected] + args.id_add
                summary.append(f"{folder}={selected.size}")

                for filename, (_header, field, decimals) in specs.items():
                    if field == "mass":
                        values = (
                            densities[species_value]
                            * (4.0 / 3.0)
                            * math.pi
                            * snapshot.radius[selected] ** 3
                        )
                    else:
                        values = getattr(snapshot, field)[selected]
                    write_rows(
                        handles[(species_value, filename)],
                        ids,
                        time,
                        values,
                        args.time_decimals,
                        decimals,
                    )

            print(f"  {path.name}: time={time:g}; " + ", ".join(summary))

    print(f"Wrote output under {output.resolve()}")
    print("Note: force is absent from the VTU input, so forces.txt was not created.")
    if densities is None:
        print(
            "Note: mass is absent from the VTU input. Use --density-a and "
            "--density-b to calculate masses.txt from Radius."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "source",
        nargs="?",
        default=DEFAULT_INPUT,
        help=f"A VTU file, directory, or quoted glob (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Output root (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--pattern",
        default="*.vtu",
        help="Pattern used when source is a directory (default: *.vtu)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=FILE_STRIDE,
        help=f"Process the first file and every Nth file after it (default: {FILE_STRIDE})",
    )
    parser.add_argument(
        "--species-map",
        nargs="+",
        default=["0:Particle_A", "1:Particle_B"],
        metavar="VALUE:FOLDER",
        help="speciesType-to-folder mapping (default: 0:Particle_A 1:Particle_B)",
    )
    parser.add_argument(
        "--id-add",
        type=int,
        default=1,
        help="Value added to every source ID (default: 1; use 0 to keep IDs)",
    )
    parser.add_argument(
        "--time-decimals",
        type=int,
        default=2,
        help="Digits after the decimal point in output times (default: 2)",
    )
    parser.add_argument(
        "--filename-time-scale",
        type=float,
        default=None,
        help="Fallback multiplier for a trailing filename number when no time comment exists",
    )
    parser.add_argument(
        "--density-a",
        type=float,
        default=None,
        help="Particle_A density; with --density-b, calculates masses.txt",
    )
    parser.add_argument(
        "--density-b",
        type=float,
        default=None,
        help="Particle_B density; with --density-a, calculates masses.txt",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate and summarize input without writing output",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.time_decimals < 0:
        parser.error("--time-decimals must be non-negative")
    if args.density_a is not None and args.density_a <= 0:
        parser.error("--density-a must be positive")
    if args.density_b is not None and args.density_b <= 0:
        parser.error("--density-b must be positive")
    try:
        convert(args)
    except (OSError, ValueError, ET.ParseError, argparse.ArgumentTypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
