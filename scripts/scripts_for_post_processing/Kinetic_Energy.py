#!/usr/bin/env python3
"""Plot total particle kinetic energy over time for each DEM code.

Input layout:
    Particle_Data / Code / Shape / Particle_A / masses.txt
    Particle_Data / Code / Shape / Particle_A / velocities.txt
    Particle_Data / Code / Shape / Particle_A / angular_velocities.txt
    ...and the same three files under Particle_B.

Each text file has a commented header and whitespace-separated rows:
    masses.txt:             # id time mass
    velocities.txt:         # id time vx vy vz
    angular_velocities.txt: # id time wx wy wz

Energy includes translation and rotation. Mass can vary by particle ID and
time; the three files are matched by ID at each saved time.
"""

import argparse
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# User configuration
# ======================================================================
legendfont  = 17
labelfont   = 17
titlefont   = 15

base_dir = Path(
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)

shapes = ["Spheres", "Cubes"]
sphere_codes = ["EDEM", "Kratos-Multiphysics", "MercuryDPM", "Liggghts"]
cube_codes = ["EDEM", "YDEM", "Yade"]

# These position settings are used in the free-surface analysis. Kinetic
# energy uses speed magnitudes, so recentering/mirroring is unnecessary.
coordinate_rules = {
    "Yade": ("x", "z", True),
    "YDEM": ("x", "z", True),
    "MercuryDPM": ("x", "y", True),
}
source_drum_centres = {"YDEM": (0.11, 0.11)}
source_thickness_centres = {"YDEM": 0.03}

mass_units_by_code = {"Kratos-Multiphysics": "g"}
mass_unit_to_kg = {"kg": 1.0, "g": 1.0e-3, "mg": 1.0e-6}
particle_density_kg_m3 = 2500.0
chunk_size = 250_000
skip_missing_datasets = True

MASS_COLUMNS = ["id", "time", "mass"]
VELOCITY_COLUMNS = ["id", "time", "vx", "vy", "vz"]
ANGULAR_COLUMNS = ["id", "time", "wx", "wy", "wz"]


def text_chunks(path, columns, usecols=None):
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        usecols=usecols,
        chunksize=chunk_size,
        encoding="utf-8-sig",
    )


def time_frames(path, columns):
    """Yield one saved time at a time, even if a chunk splits that time."""
    current_time = None
    parts = []
    for chunk in text_chunks(path, columns):
        times = pd.to_numeric(chunk["time"], errors="raise").to_numpy(float)
        if not np.isfinite(times).all():
            raise ValueError(f"Non-finite time in {path}")
        if np.any(np.diff(times) < 0):
            raise ValueError(f"Times must be grouped in increasing order in {path}")
        boundaries = np.r_[0, np.flatnonzero(np.diff(times) != 0) + 1, len(times)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            time = float(times[start])
            part = chunk.iloc[start:end].copy()
            if current_time is None:
                current_time = time
            elif time != current_time:
                if time < current_time:
                    raise ValueError(f"Times must be in increasing order in {path}")
                yield current_time, pd.concat(parts, ignore_index=True)
                current_time, parts = time, []
            parts.append(part)
    if parts:
        yield current_time, pd.concat(parts, ignore_index=True)


def index_particles(frame, path, time):
    ids = pd.to_numeric(frame["id"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids != np.floor(ids)).any():
        raise ValueError(f"Invalid particle ID in {path} at t={time:g}")
    frame = frame.copy()
    frame["id"] = ids.astype(np.int64)
    if frame["id"].duplicated().any():
        raise ValueError(f"Duplicate particle ID in {path} at t={time:g}")
    return frame.set_index("id")


def moment_of_inertia(mass, shape):
    """One diagonal component of the inertia tensor, in kg m^2."""
    volume = mass / particle_density_kg_m3
    if shape == "Spheres":
        radius = (3.0 * volume / (4.0 * np.pi)) ** (1.0 / 3.0)
        return 2.0 * mass * radius**2 / 5.0
    if shape == "Cubes":
        side = volume ** (1.0 / 3.0)
        return mass * side**2 / 6.0
    raise ValueError(f"Unsupported shape {shape}")


def species_energy(folder, code, shape):
    mass_path = folder / "masses.txt"
    velocity_path = folder / "velocities.txt"
    angular_path = folder / "angular_velocities.txt"
    unit = mass_units_by_code.get(code, "kg").lower()
    if unit not in mass_unit_to_kg:
        raise ValueError(f"Unknown mass unit {unit!r} for {code}")
    factor = mass_unit_to_kg[unit]

    totals = defaultdict(float)
    row_count = 0
    mass_reader = time_frames(mass_path, MASS_COLUMNS)
    velocity_reader = time_frames(velocity_path, VELOCITY_COLUMNS)
    angular_reader = time_frames(angular_path, ANGULAR_COLUMNS)
    for frames in zip_longest(mass_reader, velocity_reader, angular_reader):
        if any(item is None for item in frames):
            raise ValueError(
                f"Mass, velocity, and angular files have different saved times "
                f"under {folder}"
            )
        (mass_time, mass), (velocity_time, velocity), (angular_time, angular) = frames
        if not (
            np.isclose(mass_time, velocity_time, rtol=0.0, atol=1.0e-12)
            and np.isclose(angular_time, velocity_time, rtol=0.0, atol=1.0e-12)
        ):
            raise ValueError(f"Saved times do not align under {folder}")

        mass = index_particles(mass, mass_path, mass_time)
        velocity = index_particles(velocity, velocity_path, velocity_time)
        angular = index_particles(angular, angular_path, angular_time)
        if not (
            len(mass) == len(velocity) == len(angular)
            and velocity.index.isin(mass.index).all()
            and velocity.index.isin(angular.index).all()
        ):
            raise ValueError(
                f"Particle IDs do not match across mass and velocity files "
                f"under {folder} at t={velocity_time:g}"
            )
        masses = pd.to_numeric(
            mass.loc[velocity.index, "mass"], errors="raise"
        ).to_numpy(float) * factor
        if not np.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError(f"Invalid mass in {mass_path} at t={mass_time:g}")
        inertia = moment_of_inertia(masses, shape)
        v = velocity[["vx", "vy", "vz"]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(float)
        w = angular.loc[velocity.index, ["wx", "wy", "wz"]].apply(
            pd.to_numeric, errors="raise"
        ).to_numpy(float)
        if not (np.isfinite(v).all() and np.isfinite(w).all()):
            raise ValueError(f"Non-finite velocity in {folder} at t={velocity_time:g}")
        energy = (
            0.5 * masses * np.sum(v * v, axis=1)
            + 0.5 * inertia * np.sum(w * w, axis=1)
        )
        totals[velocity_time] = float(np.sum(energy))
        row_count += len(velocity)

    if row_count == 0:
        raise ValueError(f"No particle velocities in {velocity_path}")
    return totals


def case_energy(folder, code, shape):
    by_species = [
        species_energy(folder / species, code, shape)
        for species in ("Particle_A", "Particle_B")
    ]
    if set(by_species[0]) != set(by_species[1]):
        raise ValueError(
            f"{folder}: Particle_A and Particle_B have different saved times"
        )
    return [
        (time, by_species[0][time] + by_species[1][time])
        for time in sorted(by_species[0])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=base_dir)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.base_dir.expanduser()
    if not root.is_dir():
        parser.error(f"Particle_Data folder not found: {root}")
    if particle_density_kg_m3 <= 0 or chunk_size < 1:
        parser.error("Density and chunk size must be positive")
    output_dir = args.output_dir or root / "Kinetic_Energy_Results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for shape in shapes:
        codes = sphere_codes if shape == "Spheres" else cube_codes
        rows = []
        fig, ax = plt.subplots(figsize=(9, 5))
        for code in codes:
            folder = root / code / shape
            if not folder.is_dir():
                print(f"Skipping {code}/{shape}: folder not found")
                continue
            required = [
                folder / species / filename
                for species in ("Particle_A", "Particle_B")
                for filename in (
                    "masses.txt", "velocities.txt", "angular_velocities.txt"
                )
            ]
            missing = [path for path in required if not path.is_file()]
            if missing:
                message = (
                    f"{code}/{shape} missing "
                    + ", ".join(str(path.relative_to(folder)) for path in missing)
                )
                if skip_missing_datasets:
                    print(f"Skipping {message}")
                    continue
                raise FileNotFoundError(message)

            series = case_energy(folder, code, shape)
            times, energies = zip(*series)
            ax.plot(times, energies, label=code)
            rows.extend(
                {"shape": shape, "code": code, "time_s": t, "energy_J": e}
                for t, e in series
            )
            print(f"{code}/{shape}: {len(series)} time points")

        if not rows:
            plt.close(fig)
            print(f"No {shape} results found")
            continue
        # ax.set(
        #     title=f"{shape}: total particle kinetic energy",
        #     xlabel="Time (s)",
        #     ylabel="Total particle kinetic energy (J)",
        # )
        # ax.titlefont = titlefont
        ax.set_xlabel("Time [s]", fontsize=labelfont)
        ax.set_ylabel("Total Kinetic Energy [J]", fontsize = labelfont)
        # ax.set_title()
        ax.grid(alpha=0.3)
        ax.legend(fontsize = legendfont)
        ax.tick_params(axis="x", labelsize=labelfont)
        ax.tick_params(axis="y", labelsize=labelfont)
        fig.tight_layout()
        image = output_dir / f"{shape}_kinetic_energy.png"
        table = output_dir / f"{shape}_kinetic_energy.csv"
        fig.savefig(image, dpi=200)
        pd.DataFrame(rows).to_csv(table, index=False)
        plt.close(fig)
        print(f"Saved {image} and {table}")


if __name__ == "__main__":
    main()
