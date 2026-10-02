"""
Convert rotatingDrum-case_*_volumetric_time_*.csv files (one file per timestep,
one row per particle: BID, MatID, mass, x, y, z, vx, vy, vz, avx, avy, avz,
fx, fy, fz, tx, ty, tz) into the Particle_A / Particle_B folder format used
by Particle_Counting_LMI.py:

    <out_dir>/Particle_A/positions.txt          # id time x y z
    <out_dir>/Particle_A/velocities.txt         # id time vx vy vz
    <out_dir>/Particle_A/angular_velocities.txt # id time wx wy wz
    <out_dir>/Particle_A/forces.txt             # id time Fx Fy Fz
    <out_dir>/Particle_A/masses.txt             # id time mass
    <out_dir>/Particle_B/...                    # same, for species B

MatID mapping (confirmed for the rotatingDrum case_A dataset):
    MatID 0 -> Particle A (larger particles, fewer of them)
    MatID 1 -> Particle B (smaller particles, more of them)
    MatID 2 -> wall / boundary geometry -> dropped

BID is used directly as the particle id (it is stable across timesteps for
a given particle, which is what the downstream binning/Lacey-index script
relies on).

Usage (defaults to the paths below, matching Particle_Counting_LMI.py's base_dir):
    python convert_volumetric_csv.py
    python convert_volumetric_csv.py <input_dir> <output_dir> [--pattern GLOB]
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

# -----------------------------------------------------------------------
# Config: edit here if a new source uses different MatID / column naming
# or paths change
# -----------------------------------------------------------------------
BASE_DIR = (
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)
INPUT_SUBFOLDER = os.path.join("YADE", "Cubes_Raw")  # raw per-timestep CSVs
OUTPUT_SUBFOLDER = os.path.join("YADE", "Cubes")  # Particle_A / Particle_B go here

MATID_TO_SPECIES = {0: "Particle_A", 1: "Particle_B"}  # MatID 2 = wall, dropped

# filename -> time. Handles both separators seen in the wild:
#   "..._time_0_0s_..."  -> 0.0   (underscore as decimal point)
#   "..._time_0.0s_..."  -> 0.0   (literal decimal point)
#   "..._time_1.25s_..." -> 1.25
TIME_PATTERN = re.compile(r"time_(\d+)[._](\d+)s")

# output value formatting (kept close to the EDEM/Kratos reference format)
TIME_FMT = "{:.2f}"
POS_FMT = "{:.6f}"
VEL_FMT = "{:.6f}"
FORCE_FMT = "{:.6f}"
MASS_FMT = "{:.12f}"


def parse_time_from_filename(path):
    name = os.path.basename(path)
    m = TIME_PATTERN.search(name)
    if not m:
        raise ValueError(f"Could not parse a timestep from filename: {name}")
    whole, frac = m.groups()
    return float(f"{whole}.{frac}")


def load_all_timesteps(input_dir, pattern):
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern {pattern!r} in {input_dir}"
        )

    frames = []
    for f in files:
        t = parse_time_from_filename(f)
        df = pd.read_csv(f)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
        df["time"] = t
        frames.append(df)
        print(f"  read {os.path.basename(f):55s} time={t:6.2f}  rows={len(df)}")

    full = pd.concat(frames, ignore_index=True)

    n_wall = (~full["MatID"].isin(MATID_TO_SPECIES)).sum()
    if n_wall:
        wall_ids = sorted(full.loc[~full["MatID"].isin(MATID_TO_SPECIES), "MatID"].unique())
        print(f"  dropping {n_wall} rows with MatID {wall_ids} (wall/boundary, not a species)")

    return full


def write_species_file(df, id_col, time_col, value_cols, header_comment, out_path, fmts):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n") as fh:
        fh.write(header_comment + "\n")
        for row in df.itertuples(index=False):
            pid = int(getattr(row, id_col)) - 51
            t = fmts["time"].format(getattr(row, time_col))
            vals = " ".join(
                fmts["value"].format(getattr(row, c)) for c in value_cols
            )
            fh.write(f"{pid} {t} {vals}\n")


def convert(input_dir, output_dir, pattern):
    full = load_all_timesteps(input_dir, pattern)

    outputs = {
        "positions.txt": (["x", "y", "z"], "# id time x y z", POS_FMT),
        "velocities.txt": (["vx", "vy", "vz"], "# id time vx vy vz", VEL_FMT),
        "angular_velocities.txt": (["avx", "avy", "avz"], "# id time wx wy wz", VEL_FMT),
        "forces.txt": (["fx", "fy", "fz"], "# id time Fx Fy Fz", FORCE_FMT),
        "masses.txt": (["mass"], "# id time mass", MASS_FMT),
    }

    for mat_id, species in MATID_TO_SPECIES.items():
        sub = full[full["MatID"] == mat_id].sort_values(["time", "BID"])
        if sub.empty:
            print(f"  WARNING: no rows found for MatID {mat_id} ({species})")
            continue

        n_particles = sub["BID"].nunique()
        n_times = sub["time"].nunique()
        print(f"\n{species}: {n_particles} particles x {n_times} timesteps = {len(sub)} rows")

        for fname, (cols, header, fmt) in outputs.items():
            out_path = os.path.join(output_dir, species, fname)
            write_species_file(
                sub,
                id_col="BID",
                time_col="time",
                value_cols=cols,
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
        default="rotatingDrum-case_*_volumetric_time_*.csv",
        help="Glob pattern for the input CSV files (default: %(default)s)",
    )
    args = ap.parse_args()

    print(f"Scanning {args.input_dir} for {args.pattern!r} ...")
    convert(args.input_dir, args.output_dir, args.pattern)
    print("\nDone.")


if __name__ == "__main__":
    main()