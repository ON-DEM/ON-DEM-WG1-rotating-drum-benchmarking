"""
Convert per-quantity CSV exports that are ALREADY split into Particle_A/ and
Particle_B/ subfolders (species comes from the folder, not a filename
prefix), e.g.:

    <in_dir>/Particle_A/avel.csv    # id,time,wx,wy,wz
    <in_dir>/Particle_A/force.csv   # id,time,Fx,Fy,Fz
    <in_dir>/Particle_B/avel.csv
    ...

into the .txt format used by Particle_Counting_LMI.py:

    <out_dir>/Particle_A/positions.txt          # id time x y z
    <out_dir>/Particle_A/velocities.txt         # id time vx vy vz
    <out_dir>/Particle_A/angular_velocities.txt # id time wx wy wz
    <out_dir>/Particle_A/forces.txt             # id time Fx Fy Fz
    <out_dir>/Particle_A/masses.txt             # id time mass
    <out_dir>/Particle_B/...                    # same, for species B

Quantity is detected from each filename by substring match against
QUANTITY_MAP below (works whether the file is named "avel.csv",
"l_avel.csv", "angular_vel.csv", etc. -- as long as it contains one of the
known quantity keys).

Particle_B IDs are offset by OFFSET_FOR_B so IDs stay globally unique and
consistent with the earlier converters' 1..6000 / 6001.. scheme (adjust
OFFSET_FOR_B if Particle_A's particle count differs from 6000). Set to 0 to
keep each species' original IDs unchanged.

Usage (defaults to the paths below, matching Particle_Counting_LMI.py's base_dir):
    python convert_species_folders_csv.py
    python convert_species_folders_csv.py <input_dir> <output_dir>
"""

import argparse
import glob
import os

import pandas as pd

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
BASE_DIR = (
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)
INPUT_SUBFOLDER = os.path.join("Liggghts", "Spheres_Raw")  # has Particle_A/ Particle_B/ inside
OUTPUT_SUBFOLDER = os.path.join("Liggghts", "Spheres")

SPECIES_FOLDERS = ["Particle_A", "Particle_B"]
OFFSET_FOR_B = 6000  # Particle_A particle count -> Particle_B ids become original_id + this

TIME_FMT = "{:.2f}"
POS_FMT = "{:.6f}"
VEL_FMT = "{:.6f}"
FORCE_FMT = "{:.6f}"
MASS_FMT = "{:.12f}"

# quantity key (lowercase, matched as a substring of the filename) ->
# (output filename, header comment, component columns as named in the source, format)
QUANTITY_MAP = {
    "avel": ("angular_velocities.txt", "# id time wx wy wz", ["wx", "wy", "wz"], VEL_FMT),
    "vel": ("velocities.txt", "# id time vx vy vz", ["vx", "vy", "vz"], VEL_FMT),
    "pos": ("positions.txt", "# id time x y z", ["x", "y", "z"], POS_FMT),
    "force": ("forces.txt", "# id time Fx Fy Fz", ["Fx", "Fy", "Fz"], FORCE_FMT),
    "mass": ("masses.txt", "# id time mass", ["mass"], MASS_FMT),
}
# order matters: "avel" must be checked before "vel" since "vel" is a substring of "avel"
QUANTITY_ORDER = ["avel", "vel", "pos", "force", "mass"]


def detect_quantity(filename):
    stem = os.path.splitext(filename)[0].lower()
    for key in QUANTITY_ORDER:
        if key in stem:
            return key
    return None


def write_species_file(df, value_cols, header_comment, out_path, value_fmt):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n") as fh:
        fh.write(header_comment + "\n")
        for row in df.itertuples(index=False):
            pid = int(getattr(row, "id"))
            t = TIME_FMT.format(getattr(row, "time"))
            vals = " ".join(value_fmt.format(getattr(row, c)) for c in value_cols)
            fh.write(f"{pid} {t} {vals}\n")


def convert(input_dir, output_dir):
    for species in SPECIES_FOLDERS:
        species_dir = os.path.join(input_dir, species)
        if not os.path.isdir(species_dir):
            print(f"  skipping {species}: folder not found at {species_dir}")
            continue

        files = sorted(glob.glob(os.path.join(species_dir, "*.csv")))
        if not files:
            print(f"  {species}: no .csv files found in {species_dir}")
            continue

        print(f"\n{species} ({species_dir}):")
        for f in files:
            name = os.path.basename(f)
            quantity = detect_quantity(name)
            if quantity is None:
                print(f"  skipping {name} (couldn't detect a known quantity)")
                continue

            out_fname, header, value_cols, fmt = QUANTITY_MAP[quantity]
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]

            # known export quirk: a "mass" header sometimes gets mangled into
            # separate single-letter columns "m,a,s,s" (data rows still only
            # have id,time,<value>); the real mass value lands in column "m"
            if quantity == "mass" and "mass" not in df.columns and "m" in df.columns:
                df = df.rename(columns={"m": "mass"})

            missing = [c for c in ["id", "time"] + value_cols if c not in df.columns]
            if missing:
                print(f"  skipping {name}: missing expected columns {missing} "
                      f"(found {df.columns.tolist()})")
                continue

            if species == "Particle_B" and OFFSET_FOR_B:
                df = df.copy()
                df["id"] = df["id"] + OFFSET_FOR_B

            df = df.sort_values(["time", "id"])

            out_path = os.path.join(output_dir, species, out_fname)
            write_species_file(df, value_cols, header, out_path, fmt)
            n_particles = df["id"].nunique()
            n_times = df["time"].nunique()
            print(f"  {name} -> {out_path}  ({n_particles} particles x {n_times} timesteps)")


def main():
    default_input = os.path.join(BASE_DIR, INPUT_SUBFOLDER)
    default_output = os.path.join(BASE_DIR, OUTPUT_SUBFOLDER)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "input_dir", nargs="?", default=default_input,
        help=f"Folder containing Particle_A/ and Particle_B/ subfolders (default: {default_input})",
    )
    ap.add_argument(
        "output_dir", nargs="?", default=default_output,
        help=f"Folder to write converted Particle_A/ and Particle_B/ into (default: {default_output})",
    )
    args = ap.parse_args()

    print(f"Scanning {args.input_dir} ...")
    convert(args.input_dir, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()