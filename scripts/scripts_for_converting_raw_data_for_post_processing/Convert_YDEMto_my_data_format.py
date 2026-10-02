"""
Convert per-timestep DEM particle CSV exports (one file per timestep, one row
per particle, columns like BID/MatID/mass/x/y/z/velocities/...) into the
Particle_A / Particle_B folder format used by Particle_Counting_LMI.py:

    <out_dir>/Particle_A/positions.txt          # id time x y z
    <out_dir>/Particle_A/velocities.txt         # id time vx vy vz
    <out_dir>/Particle_A/angular_velocities.txt # id time wx wy wz
    <out_dir>/Particle_A/forces.txt             # id time Fx Fy Fz          (if present)
    <out_dir>/Particle_A/masses.txt             # id time mass
    <out_dir>/Particle_A/moments_of_inertia.txt # id time Ixx Ixy Ixz Iyy Iyz Izz (if present)
    <out_dir>/Particle_A/id_mapping.csv         # original_id new_id (for traceability)
    <out_dir>/Particle_B/...                    # same, for species B

Particle IDs are re-numbered to a contiguous sequential range so that
Particle_A gets 1..nA and Particle_B gets (nA+1)..(nA+nB) (e.g. A = 1-6000,
B = 6001-54000), rather than reusing the source BID values, which may be
zero-based, interleaved between species, or offset differently between
sources. A given physical particle keeps the same new ID across all
timesteps. The original_id -> new_id mapping is written to
id_mapping.csv per species in case you need to trace a particle back to the
source data.

Handles multiple source naming conventions seen so far:
    rotatingDrum-case_A_volumetric_time_0_0s_2026_08_24_v1.csv   (comma-sep, MatID 0/1/2)
    rotatingDrum-Case_B_linear_time__0_2.csv                     (whitespace-sep, MatID 1/2)

Species detection is automatic rather than hardcoded to specific MatID values,
since different exports have used different MatID numbering:
    - The two MatID groups with the most particles are treated as the two real
      species. Any other MatID group (e.g. a handful of oversized particles
      representing wall/boundary geometry) is dropped, with a printed warning.
    - Of the two species groups, the one with the LARGER mean particle mass is
      written to Particle_A, the SMALLER to Particle_B (confirmed convention:
      "larger particles are Particle A, smaller are Particle B, and there are
      more of B than A").

Usage (defaults to the paths below, matching Particle_Counting_LMI.py's base_dir):
    python convert_volumetric_csv.py
    python convert_volumetric_csv.py <input_dir> <output_dir> [--pattern GLOB]
"""

import argparse
import glob
import os
import re

import pandas as pd

# -----------------------------------------------------------------------
# Config: edit here if paths change
# -----------------------------------------------------------------------
BASE_DIR = (
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)
INPUT_SUBFOLDER = os.path.join("YDEM", "Cubes_Raw")  # raw per-timestep CSVs
OUTPUT_SUBFOLDER = os.path.join("YDEM", "Cubes")  # Particle_A / Particle_B go here

# filename -> time. Handles all separator styles seen so far:
#   "..._time_0_0s_..."    -> 0.0   (underscore decimal point, trailing s)
#   "..._time_0.0s_..."    -> 0.0   (literal decimal point, trailing s)
#   "..._time__0_2..."     -> 0.2   (double underscore, no trailing s)
#   "..._time_ 0.2.csv"    -> 0.2   (space before the number, no trailing s)
#   "..._time_1.25s_..."   -> 1.25
TIME_PATTERN = re.compile(r"time[_ ]+(\d+)[._](\d+)s?")

# column-name aliases: first match found in a file's columns is used
COLUMN_ALIASES = {
    "x": ["x"], "y": ["y"], "z": ["z"],
    "vx": ["vx"], "vy": ["vy"], "vz": ["vz"],
    "avx": ["avx", "wx"], "avy": ["avy", "wy"], "avz": ["avz", "wz"],
    "fx": ["fx"], "fy": ["fy"], "fz": ["fz"],
    "mass": ["mass"],
    "Ixx": ["Ixx"], "Ixy": ["Ixy"], "Ixz": ["Ixz"],
    "Iyy": ["Iyy"], "Iyz": ["Iyz"], "Izz": ["Izz"],
    "BID": ["BID"], "MatID": ["MatID"],
}

# output value formatting (kept close to the EDEM/Kratos reference format)
TIME_FMT = "{:.2f}"
POS_FMT = "{:.6f}"
VEL_FMT = "{:.6f}"
FORCE_FMT = "{:.6f}"
MASS_FMT = "{:.12f}"
INERTIA_FMT = "{:.12e}"

# candidate output files: name -> (canonical cols, header comment, format)
OUTPUT_SPECS = {
    "positions.txt": (["x", "y", "z"], "# id time x y z", POS_FMT),
    "velocities.txt": (["vx", "vy", "vz"], "# id time vx vy vz", VEL_FMT),
    "angular_velocities.txt": (["avx", "avy", "avz"], "# id time wx wy wz", VEL_FMT),
    "forces.txt": (["fx", "fy", "fz"], "# id time Fx Fy Fz", FORCE_FMT),
    "masses.txt": (["mass"], "# id time mass", MASS_FMT),
    "moments_of_inertia.txt": (
        ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"],
        "# id time Ixx Ixy Ixz Iyy Iyz Izz",
        INERTIA_FMT,
    ),
}


def parse_time_from_filename(path):
    name = os.path.basename(path)
    m = TIME_PATTERN.search(name)
    if not m:
        raise ValueError(f"Could not parse a timestep from filename: {name}")
    whole, frac = m.groups()
    return float(f"{whole}.{frac}")


def resolve_columns(df):
    """Map canonical column name -> actual column name present in df (or None)."""
    resolved = {}
    for canon, aliases in COLUMN_ALIASES.items():
        found = next((a for a in aliases if a in df.columns), None)
        resolved[canon] = found
    return resolved


def read_one_file(path):
    # Try comma-separated first, fall back to whitespace-separated
    try:
        df = pd.read_csv(path)
        if df.shape[1] < 3:  # comma-parse degenerated -> probably whitespace file
            raise ValueError
    except (ValueError, pd.errors.ParserError):
        df = pd.read_csv(path, sep=r"\s+")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df.columns = [c.strip() for c in df.columns]
    return df


def load_all_timesteps(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern {pattern!r} in {input_dir}"
        )

    frames = []
    for f in files:
        t = parse_time_from_filename(f)
        df = read_one_file(f)
        df["time"] = t
        frames.append(df)
        print(f"  read {os.path.basename(f):55s} time={t:6.2f}  rows={len(df)}")

    full = pd.concat(frames, ignore_index=True, sort=False)
    return full


def classify_species(full, matid_col):
    """
    Auto-detect which MatID values are the two real particle species vs.
    wall/boundary geometry, and which is Particle_A (larger mass) vs
    Particle_B (smaller mass).
    """
    counts = full[matid_col].value_counts().sort_values(ascending=False)
    if len(counts) < 2:
        raise ValueError(
            f"Expected at least 2 distinct {matid_col} groups, found {len(counts)}"
        )

    species_ids = counts.index[:2].tolist()
    dropped_ids = counts.index[2:].tolist()
    if dropped_ids:
        dropped_counts = {i: int(counts[i]) for i in dropped_ids}
        print(f"  dropping {matid_col} groups {dropped_counts} "
              f"(not among the two largest-count groups -> treated as wall/boundary)")

    mean_mass = full[full[matid_col].isin(species_ids)].groupby(matid_col)["mass"].mean()
    ordered = mean_mass.sort_values(ascending=False)  # largest mass first
    a_id, b_id = ordered.index[0], ordered.index[1]
    print(f"  Particle_A = {matid_col} {a_id} (mean mass {ordered[a_id]:.6e}, "
          f"n={int(counts[a_id])})")
    print(f"  Particle_B = {matid_col} {b_id} (mean mass {ordered[b_id]:.6e}, "
          f"n={int(counts[b_id])})")

    return {a_id: "Particle_A", b_id: "Particle_B"}


def write_species_file(df, id_col, time_col, value_cols, header_comment, out_path, fmts):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n") as fh:
        fh.write(header_comment + "\n")
        for row in df.itertuples(index=False):
            pid = int(getattr(row, id_col))
            t = fmts["time"].format(getattr(row, time_col))
            vals = " ".join(
                fmts["value"].format(getattr(row, c)) for c in value_cols
            )
            fh.write(f"{pid} {t} {vals}\n")


def assign_sequential_ids(sub, id_col, start):
    """
    Replace the original per-species particle IDs with a contiguous sequential
    range starting at `start`, ordered by the original ID (stable across
    timesteps for a given particle). Returns (relabeled df, id_map, next_start).
    """
    original_ids = sorted(sub[id_col].unique())
    id_map = {old: new for new, old in enumerate(original_ids, start=start)}
    sub = sub.copy()
    sub[id_col] = sub[id_col].map(id_map)
    next_start = start + len(original_ids)
    return sub, id_map, next_start


def write_id_mapping(id_map, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n") as fh:
        fh.write("# original_id new_id\n")
        for old, new in sorted(id_map.items(), key=lambda kv: kv[1]):
            fh.write(f"{old} {new}\n")


def convert(input_dir, output_dir, pattern):
    full = load_all_timesteps(input_dir, pattern)

    cols = resolve_columns(full)
    id_col = cols["BID"]
    matid_col = cols["MatID"]
    if id_col is None or matid_col is None:
        raise ValueError("Could not find BID / MatID columns in the input files")

    mat_to_species = classify_species(full, matid_col)

    next_id = 1
    for mat_id, species in mat_to_species.items():
        sub = full[full[matid_col] == mat_id].sort_values(["time", id_col])
        n_particles = sub[id_col].nunique()
        n_times = sub["time"].nunique()
        print(f"\n{species}: {n_particles} particles x {n_times} timesteps = {len(sub)} rows")

        sub, id_map, next_id = assign_sequential_ids(sub, id_col, next_id)
        sub = sub.sort_values(["time", id_col])
        first_id, last_id = min(id_map.values()), max(id_map.values())
        print(f"  re-IDed: {species} -> {first_id}..{last_id}")

        id_map_path = os.path.join(output_dir, species, "id_mapping.csv")
        write_id_mapping(id_map, id_map_path)
        print(f"  wrote {id_map_path}")

        for fname, (canon_value_cols, header, fmt) in OUTPUT_SPECS.items():
            actual_cols = [cols.get(c) for c in canon_value_cols]
            if any(c is None for c in actual_cols):
                print(f"  skipping {fname} (columns not present in source: "
                      f"{[c for c, a in zip(canon_value_cols, actual_cols) if a is None]})")
                continue

            out_path = os.path.join(output_dir, species, fname)
            renamed = sub.rename(columns={a: c for a, c in zip(actual_cols, canon_value_cols)})
            write_species_file(
                renamed,
                id_col=id_col,
                time_col="time",
                value_cols=canon_value_cols,
                header_comment=header,
                out_path=out_path,
                fmts={"time": TIME_FMT, "value": fmt},
            )
            print(f"  wrote {out_path}")


def main():
    default_input = os.path.join(BASE_DIR, INPUT_SUBFOLDER)
    default_output = os.path.join(BASE_DIR, OUTPUT_SUBFOLDER)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "input_dir",
        nargs="?",
        default=default_input,
        help=f"Folder containing the per-timestep CSV files (default: {default_input})",
    )
    ap.add_argument(
        "output_dir",
        nargs="?",
        default=default_output,
        help=f"Folder to write Particle_A/ and Particle_B/ into (default: {default_output})",
    )
    ap.add_argument(
        "--pattern",
        default="*_time*.csv",
        help="Glob pattern for the input CSV files (default: %(default)s)",
    )
    args = ap.parse_args()

    print(f"Scanning {args.input_dir} for {args.pattern!r} ...")
    convert(args.input_dir, args.output_dir, args.pattern)
    print("\nDone.")


if __name__ == "__main__":
    main()