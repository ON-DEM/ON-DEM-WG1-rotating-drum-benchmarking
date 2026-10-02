#!/usr/bin/env python3
"""Calculate and compare the LMI for all configured codes and shapes.

Each dataset is expected at:
    BASE_DIR / code / shape

with:
    Particle_A/positions.txt
    Particle_A/masses.txt
    Particle_B/positions.txt
    Particle_B/masses.txt

Add or remove software by editing only ``sphere_codes`` and ``cube_codes``.
Sphere curves are solid; cube curves are dashed.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ======================================================================
# User configuration
# ======================================================================

base_dir = Path(
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)

shapes = ["Spheres", "Cubes"]

sphere_codes = ["EDEM", "Kratos-Multiphysics", "MercuryDPM", "Liggghts"]
cube_codes = ["EDEM", "YDEM", "Yade"]  # Add "EDEM" here when its cube data is ready.

# Per-code exceptions: (horizontal axis, vertical axis, mirror horizontal)
coordinate_rules = {
    "Yade": ("x", "z", True),
    "YDEM": ("x", "z", True),
    "MercuryDPM": ("x", "y", True),
}

# Source-coordinate centre of each drum cross-section. Codes not listed here
# are already centred on the zero-centred analysis grid.
source_drum_centres = {
    "YDEM": (0.11, 0.11),
}

# The LMI grid also includes the drum thickness. YDEM stores that coordinate
# about y=0.03 m, whereas the analysis grid is centred on zero.
source_thickness_centres = {
    "YDEM": 0.03,
}

# Mass unit used in each code's masses.txt files. Codes not listed here are
# assumed to store kilograms. Both Particle_A and Particle_B use the same unit.
mass_units_by_code = {
    "Kratos-Multiphysics": "g",
}

mass_unit_to_kg = {
    "kg": 1.0,
    "g": 1.0e-3,
    "mg": 1.0e-6,
}

# Grid and LMI settings
cell_size = 0.01
Lx = 0.20
Ly = 0.20
Lz = 0.06
min_particles_per_cell = 30

# Large files are read incrementally to limit memory use.
chunk_size = 1_000_000

# Missing dataset folders are reported and skipped. Set False to stop instead.
skip_missing_datasets = True

# Results are written here.
results_dir = base_dir / "LMI_Results"
show_plot = True


# ======================================================================
# Fixed grid
# ======================================================================

domain = {
    "xmin": -Lx / 2,
    "xmax": Lx / 2,
    "ymin": -Ly / 2,
    "ymax": Ly / 2,
    "zmin": -Lz / 2,
    "zmax": Lz / 2,
}

nx = int(round(Lx / cell_size))
ny = int(round(Ly / cell_size))
nz = int(round(Lz / cell_size))


def dataset_folder(code: str, shape: str) -> Path:
    """Build BASE_DIR/code/shape."""
    return base_dir / code / shape


def configured_datasets() -> list[tuple[str, str]]:
    """Return (code, shape) pairs using the two user-maintained code lists."""
    codes_by_shape = {
        "Spheres": sphere_codes,
        "Cubes": cube_codes,
    }
    unknown_shapes = set(shapes) - set(codes_by_shape)
    if unknown_shapes:
        raise ValueError(f"No code list defined for shape(s): {sorted(unknown_shapes)}")
    return [
        (code, shape)
        for shape in shapes
        for code in codes_by_shape[shape]
    ]


def read_first_mass(path: Path) -> float:
    """Read the first data value from a constant-particle-mass file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing mass file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 3:
                raise ValueError(f"Malformed line {line_number} in {path}")
            mass = float(fields[2])
            if not math.isfinite(mass) or mass <= 0:
                raise ValueError(f"Invalid mass {mass!r} in {path}")
            return mass
    raise ValueError(f"No mass records found in {path}")


def mass_conversion_to_kg(code: str) -> tuple[str, float]:
    """Return the configured input unit and its conversion factor to kg."""
    unit = mass_units_by_code.get(code, "kg").lower()
    if unit not in mass_unit_to_kg:
        raise ValueError(
            f"Unknown mass unit {unit!r} for {code}; choose from "
            f"{sorted(mass_unit_to_kg)}"
        )
    return unit, mass_unit_to_kg[unit]


def positions_in_lmi_axes(chunk: pd.DataFrame, code: str) -> pd.DataFrame:
    """Map source coordinates to analysis X=horizontal, Y=vertical, Z=thickness."""
    horizontal_axis, vertical_axis, mirror_horizontal = coordinate_rules.get(
        code, ("x", "y", False)
    )
    thickness_axis = ({"x", "y", "z"} - {horizontal_axis, vertical_axis}).pop()
    source_h_centre, source_v_centre = source_drum_centres.get(code, (0.0, 0.0))

    horizontal = chunk[horizontal_axis].to_numpy() - source_h_centre
    vertical = chunk[vertical_axis].to_numpy() - source_v_centre
    thickness = (
        chunk[thickness_axis].to_numpy()
        - source_thickness_centres.get(code, 0.0)
    )
    if mirror_horizontal:
        horizontal = -horizontal

    transformed = chunk.copy()
    transformed["x"] = horizontal
    transformed["y"] = vertical
    transformed["z"] = thickness
    return transformed


def count_particles_by_cell(
    positions_path: Path,
    count_column: str,
    code: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, float], int]:
    """Read positions in chunks and aggregate particle counts by time/cell."""
    if not positions_path.is_file():
        raise FileNotFoundError(f"Missing positions file: {positions_path}")

    keys = ["time", "ix", "iy", "iz"]
    count_parts: list[pd.DataFrame] = []
    time_count_parts: list[pd.Series] = []
    total_rows = 0
    excluded_rows = 0
    bounds = {
        "xmin": math.inf,
        "xmax": -math.inf,
        "ymin": math.inf,
        "ymax": -math.inf,
        "zmin": math.inf,
        "zmax": -math.inf,
    }

    reader = pd.read_csv(
        positions_path,
        sep=r"\s+",
        comment="#",
        names=["id", "time", "x", "y", "z"],
        usecols=["time", "x", "y", "z"],
        dtype=np.float64,
        chunksize=chunk_size,
    )

    for chunk in reader:
        if chunk.empty:
            continue
        chunk = positions_in_lmi_axes(chunk, code)
        total_rows += len(chunk)
        time_count_parts.append(chunk.groupby("time", sort=False).size())

        for axis in "xyz":
            values = chunk[axis].to_numpy()
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Non-finite {axis} coordinate in {positions_path}")
            bounds[f"{axis}min"] = min(bounds[f"{axis}min"], float(values.min()))
            bounds[f"{axis}max"] = max(bounds[f"{axis}max"], float(values.max()))

        inside = (
            chunk["x"].ge(domain["xmin"])
            & chunk["x"].lt(domain["xmax"])
            & chunk["y"].ge(domain["ymin"])
            & chunk["y"].lt(domain["ymax"])
            & chunk["z"].ge(domain["zmin"])
            & chunk["z"].lt(domain["zmax"])
        )
        excluded_rows += int((~inside).sum())
        chunk = chunk.loc[inside].copy()
        if chunk.empty:
            continue

        # floor is important: astype(int) alone truncates negative values to zero.
        chunk["ix"] = np.floor(
            (chunk["x"] - domain["xmin"]) / cell_size
        ).astype(np.int32)
        chunk["iy"] = np.floor(
            (chunk["y"] - domain["ymin"]) / cell_size
        ).astype(np.int32)
        chunk["iz"] = np.floor(
            (chunk["z"] - domain["zmin"]) / cell_size
        ).astype(np.int32)

        grouped = (
            chunk.groupby(keys, sort=False)
            .size()
            .rename(count_column)
            .reset_index()
        )
        count_parts.append(grouped)

    if total_rows == 0:
        raise ValueError(f"No position records found in {positions_path}")

    if count_parts:
        counts = (
            pd.concat(count_parts, ignore_index=True)
            .groupby(keys, as_index=False, sort=True)[count_column]
            .sum()
        )
    else:
        counts = pd.DataFrame(columns=keys + [count_column])

    time_counts = pd.concat(time_count_parts).groupby(level=0).sum().sort_index()
    return counts, time_counts, bounds, excluded_rows


def compute_lacey_index(
    count_a: pd.DataFrame,
    count_b: pd.DataFrame,
    mass_a: float,
    mass_b: float,
    target_ratio: np.ndarray,
) -> pd.DataFrame:
    """Compute the same two-material mixing index as the supplied C++ logic."""
    keys = ["time", "ix", "iy", "iz"]
    all_times = np.sort(
        np.union1d(count_a["time"].unique(), count_b["time"].unique())
    )

    cells = count_a.merge(count_b, on=keys, how="outer").fillna(0)
    cells = cells[
        (cells["A_count"] + cells["B_count"]) >= min_particles_per_cell
    ].copy()

    if cells.empty:
        return pd.DataFrame(
            {
                "time": all_times,
                "lacey_index": np.nan,
                "eligible_cells": 0,
            }
        )

    mass_cell_a = cells["A_count"].to_numpy() * mass_a
    mass_cell_b = cells["B_count"].to_numpy() * mass_b
    total_cell_mass = mass_cell_a + mass_cell_b
    ratios = np.column_stack(
        (mass_cell_a / total_cell_mass, mass_cell_b / total_cell_mass)
    )

    factor = target_ratio.max() / target_ratio
    weighted_ratios = ratios * factor
    max_d = weighted_ratios.max(axis=1)
    sum_prob = (weighted_ratios / max_d[:, None]).sum(axis=1)
    cells["cell_lmi"] = sum_prob - 1.0  # n_materials - 1 equals 1 for A/B.

    result = (
        cells.groupby("time", as_index=True)
        .agg(
            lacey_index=("cell_lmi", "mean"),
            eligible_cells=("cell_lmi", "size"),
        )
        .reindex(all_times)
    )
    result.index.name = "time"
    result["eligible_cells"] = result["eligible_cells"].fillna(0).astype(int)
    return result.reset_index()


def process_dataset(code: str, shape: str) -> pd.DataFrame:
    folder = dataset_folder(code, shape)
    print(f"\n=== {code} / {shape} ===")
    print(f"Folder: {folder}")

    raw_mass_a = read_first_mass(folder / "Particle_A" / "masses.txt")
    raw_mass_b = read_first_mass(folder / "Particle_B" / "masses.txt")
    input_mass_unit, to_kg = mass_conversion_to_kg(code)
    mass_a = raw_mass_a * to_kg
    mass_b = raw_mass_b * to_kg
    print(
        f"Mass input unit: {input_mass_unit}; converted masses: "
        f"A={mass_a:.12f} kg, B={mass_b:.12f} kg"
    )

    count_a, times_a, bounds_a, excluded_a = count_particles_by_cell(
        folder / "Particle_A" / "positions.txt", "A_count", code
    )
    count_b, times_b, bounds_b, excluded_b = count_particles_by_cell(
        folder / "Particle_B" / "positions.txt", "B_count", code
    )

    # This is equivalent to the original row-count calculation when every
    # timestep contains the same number of particles, but avoids retaining all
    # position rows in memory.
    total_mass_a = mass_a * int(times_a.sum())
    total_mass_b = mass_b * int(times_b.sum())
    target_ratio = np.array([total_mass_a, total_mass_b], dtype=float)
    target_ratio /= target_ratio.sum()

    lmi = compute_lacey_index(count_a, count_b, mass_a, mass_b, target_ratio)
    lmi.insert(0, "shape", shape)
    lmi.insert(0, "software", code)

    bounds = {
        "xmin": min(bounds_a["xmin"], bounds_b["xmin"]),
        "xmax": max(bounds_a["xmax"], bounds_b["xmax"]),
        "ymin": min(bounds_a["ymin"], bounds_b["ymin"]),
        "ymax": max(bounds_a["ymax"], bounds_b["ymax"]),
        "zmin": min(bounds_a["zmin"], bounds_b["zmin"]),
        "zmax": max(bounds_a["zmax"], bounds_b["zmax"]),
    }
    print(
        "Particle ranges: "
        f"X={bounds['xmin']:.6f}..{bounds['xmax']:.6f}, "
        f"Y={bounds['ymin']:.6f}..{bounds['ymax']:.6f}, "
        f"Z={bounds['zmin']:.6f}..{bounds['zmax']:.6f}"
    )
    print(
        f"Rows: A={int(times_a.sum()):,}, B={int(times_b.sum()):,}; "
        f"outside grid: A={excluded_a:,}, B={excluded_b:,}"
    )
    print(
        f"Target mass fractions: A={target_ratio[0]:.6f}, "
        f"B={target_ratio[1]:.6f}; timesteps={len(lmi)}"
    )
    return lmi


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (software, shape), group in results.groupby(["software", "shape"], sort=False):
        valid = group.dropna(subset=["lacey_index"]).sort_values("time")
        if valid.empty:
            rows.append(
                {
                    "software": software,
                    "shape": shape,
                    "initial_lmi": np.nan,
                    "final_lmi": np.nan,
                    "mean_lmi": np.nan,
                    "maximum_lmi": np.nan,
                    "time_of_maximum": np.nan,
                }
            )
            continue
        maximum_row = valid.loc[valid["lacey_index"].idxmax()]
        rows.append(
            {
                "software": software,
                "shape": shape,
                "initial_lmi": valid.iloc[0]["lacey_index"],
                "final_lmi": valid.iloc[-1]["lacey_index"],
                "mean_lmi": valid["lacey_index"].mean(),
                "maximum_lmi": maximum_row["lacey_index"],
                "time_of_maximum": maximum_row["time"],
            }
        )
    return pd.DataFrame(rows)


def save_tables(results: pd.DataFrame) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    results = results.sort_values(["shape", "software", "time"])
    results.to_csv(results_dir / "lmi_results_long.csv", index=False)

    wide = results.pivot(
        index="time", columns=["software", "shape"], values="lacey_index"
    ).sort_index()
    wide.columns = [f"{software} | {shape}" for software, shape in wide.columns]
    wide.to_csv(results_dir / "lmi_results_wide.csv")

    summary = build_summary(results)
    summary.to_csv(results_dir / "lmi_summary.csv", index=False)
    return summary


def save_figure(results: pd.DataFrame) -> None:
    line_styles = {"Spheres": "-", "Cubes": "--"}
    software_order = list(dict.fromkeys(results["software"]))
    color_map = plt.get_cmap("tab10")
    colors = {
        software: color_map(i % 10)
        for i, software in enumerate(software_order)
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for (software, shape), group in results.groupby(["software", "shape"], sort=False):
        group = group.sort_values("time")
        ax.plot(
            group["time"],
            group["lacey_index"],
            color=colors[software],
            linestyle=line_styles[shape],
            linewidth=2.0,
            label=f"{software} ({shape})",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lacey mixing index (LMI)")
    ax.set_title("Mixing index comparison")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "lmi_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(results_dir / "lmi_comparison.pdf", bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    print(f"Grid: {nx} x {ny} x {nz}; cell size={cell_size:g} m")
    print(f"Minimum particles per included cell: {min_particles_per_cell}")

    all_results = []
    for code, shape in configured_datasets():
        folder = dataset_folder(code, shape)
        required = [
            folder / "Particle_A" / "positions.txt",
            folder / "Particle_A" / "masses.txt",
            folder / "Particle_B" / "positions.txt",
            folder / "Particle_B" / "masses.txt",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            message = (
                f"Skipping {code} / {shape}; missing: "
                + ", ".join(str(path) for path in missing)
            )
            if skip_missing_datasets:
                print(f"WARNING: {message}", file=sys.stderr)
                continue
            raise FileNotFoundError(message)
        all_results.append(process_dataset(code, shape))

    if not all_results:
        print("error: no complete datasets were found", file=sys.stderr)
        return 1

    results = pd.concat(all_results, ignore_index=True)
    summary = save_tables(results)
    save_figure(results)

    print("\nSummary:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nResults written to: {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
