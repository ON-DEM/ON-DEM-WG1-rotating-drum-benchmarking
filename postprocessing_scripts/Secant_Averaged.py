#!/usr/bin/env python3
"""Batch free-surface post-processing for all configured DEM codes.

Folder convention:
    base_dir / code / shape / Particle_A|Particle_B / quantity.txt

Only ``sphere_codes`` and ``cube_codes`` need to change when software is added.
The free-surface calculation requires positions.txt. Kinetic energy is also
calculated when velocity, angular-velocity, and mass data are present.

Both spheres and cubes include translational and rotational kinetic energy.
Their dimensions are inferred from each species mass using a material density
of 2500 kg/m^3: sphere radius r = (3m / (4*pi*rho))^(1/3), and cube side
length a = (m / rho)^(1/3). The corresponding diagonal moments of inertia are
I_sphere = 2*m*r^2/5 and I_cube = m*a^2/6.

All result tables are written as tab-delimited .txt files. The free surface
is time-averaged over ``Time_Range`` before the secant angle is calculated.
"""

from __future__ import annotations

import csv
import math
import sys
from itertools import zip_longest
from pathlib import Path
import traceback

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# ======================================================================
# User configuration
# ======================================================================

base_dir = Path(
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)

shapes = ["Spheres", "Cubes"]
sphere_codes = ["EDEM", "Kratos-Multiphysics", "MercuryDPM", "Liggghts"]
cube_codes = ["YDEM", "Yade"]  # Add "EDEM" when its cube data is ready.

# Inclusive interval used to calculate the average free surface.
Time_Range = [1.0, 3.0]

# Each figure panel uses the available particle frame nearest this time. The
# overlaid secant is still calculated from the full Time_Range above.
snapshot_time_s = Time_Range[1]
snapshot_time_tolerance_s = 1.0e-6
snapshot_dpi = 300
snapshot_secant_color = "tab:green"

# Default cross-section coordinate system.
horizontal_axis = "x"
vertical_axis = "y"

# Per-code exceptions: (horizontal axis, vertical axis, mirror horizontal)
coordinate_rules = {
    "Yade": ("x", "z", True),
    "YDEM": ("x", "z", True),
    "MercuryDPM": ("x", "y", True),
}

# Source-coordinate centre of each drum cross-section.
# Codes not listed here are already centred at drum_centre.
source_drum_centres = {
    "YDEM": (0.11, 0.11),
}


def coordinate_settings(code: str) -> tuple[str, str, bool]:
    return coordinate_rules.get(
        code,
        (horizontal_axis, vertical_axis, False),
    )


def projected_coordinates(
    frame: pd.DataFrame,
    code: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a code's coordinates into the common drum coordinate system."""
    h_axis, v_axis, mirror_horizontal = coordinate_settings(code)

    horizontal = frame[h_axis].to_numpy(dtype=float)
    vertical = frame[v_axis].to_numpy(dtype=float)

    # Translate the source drum centre onto the common analysis centre.
    source_centre = source_drum_centres.get(code, drum_centre)

    horizontal = (
        horizontal
        - source_centre[0]
        + drum_centre[0]
    )
    vertical = (
        vertical
        - source_centre[1]
        + drum_centre[1]
    )

    # Mirror only after recentering.
    if mirror_horizontal:
        horizontal = 2.0 * drum_centre[0] - horizontal

    return horizontal, vertical

drum_centre = (0.0, 0.0)
drum_diameter = 0.20

# Same grid size as the original Fortran program.
grid_size = 0.0032

# Mass units in masses.txt. Unlisted codes default to kg.
mass_units_by_code = {
    "Kratos-Multiphysics": "g",
}
mass_unit_to_kg = {"kg": 1.0, "g": 1.0e-3, "mg": 1.0e-6}

# Material density used to infer sphere radii and cube side lengths from mass.
# Masses are converted to kg before this SI density is applied.
particle_density_kg_m3 = 2500.0

chunk_size = 1_000_000
calculate_kinetic_energy = True
skip_missing_datasets = True
results_root = base_dir / "Free_Surface_Results"


# ======================================================================
# Input definitions
# ======================================================================

POSITION_COLUMNS = ["id", "time", "x", "y", "z"]
VELOCITY_COLUMNS = ["id", "time", "vx", "vy", "vz"]
ANGULAR_COLUMNS = ["id", "time", "wx", "wy", "wz"]
MASS_COLUMNS = ["id", "time", "mass"]

def dataset_folder(code: str, shape: str) -> Path:
    return base_dir / code / shape


def configured_datasets() -> list[tuple[str, str]]:
    codes_by_shape = {"Spheres": sphere_codes, "Cubes": cube_codes}
    unknown = set(shapes) - set(codes_by_shape)
    if unknown:
        raise ValueError(f"No code list is defined for shape(s): {sorted(unknown)}")
    return [(code, shape) for shape in shapes for code in codes_by_shape[shape]]


def text_chunks(
    path: Path,
    columns: list[str],
    usecols: list[str] | None = None,
):
    """Read one standardized text file in bounded-memory chunks."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=columns,
        usecols=usecols,
        chunksize=chunk_size,
    )


def first_value(path: Path, columns: list[str], value_column: str) -> float:
    """Read the first numeric value from a standardized text file."""
    reader = text_chunks(path, columns, [value_column])
    try:
        chunk = next(reader)
    except StopIteration as exc:
        raise ValueError(f"No data rows found in {path}") from exc
    value = float(chunk.iloc[0][value_column])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid {value_column}={value!r} in {path}")
    return value


def mass_factor_to_kg(code: str) -> tuple[str, float]:
    unit = mass_units_by_code.get(code, "kg").lower()
    if unit not in mass_unit_to_kg:
        raise ValueError(
            f"Unknown mass unit {unit!r} for {code}; choose from "
            f"{sorted(mass_unit_to_kg)}"
        )
    return unit, mass_unit_to_kg[unit]


# ======================================================================
# Free-surface calculation
# ======================================================================

def occupied_cells_from_positions(
    path: Path,
    code: str,
) -> tuple[pd.DataFrame, set[float], int]:
    """Return unique occupied grid cells using the code's coordinate rules."""
    origin_h = drum_centre[0] - drum_diameter / 2.0
    origin_v = drum_centre[1] - drum_diameter / 2.0
    grid_count = math.ceil(drum_diameter / grid_size)

    occupied_parts: list[pd.DataFrame] = []
    times: set[float] = set()
    rows = 0

    h_axis, v_axis, mirror_horizontal = coordinate_settings(code)
    needed = list(dict.fromkeys(["time", h_axis, v_axis]))
    start_time, end_time = map(float, Time_Range)

    for chunk in text_chunks(path, POSITION_COLUMNS, needed):
        in_average_range = chunk["time"].between(
            start_time - snapshot_time_tolerance_s,
            end_time + snapshot_time_tolerance_s,
            inclusive="both",
        )
        chunk = chunk.loc[in_average_range]
        if chunk.empty:
            continue

        rows += len(chunk)
        horizontal, vertical = projected_coordinates(chunk, code)
        time_values = chunk["time"].to_numpy(dtype=float)

        values = np.column_stack((time_values, horizontal, vertical))
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite position value in {path}")

        times.update(map(float, chunk["time"].unique()))

        h_index = (
            np.ceil((horizontal - origin_h) / grid_size).astype(np.int64) - 1
        )
        v_index = (
            np.ceil((vertical - origin_v) / grid_size).astype(np.int64) - 1
        )

        invalid = (
            (h_index < 0)
            | (h_index >= grid_count)
            | (v_index < 0)
            | (v_index >= grid_count)
        )

        if invalid.any():
            first = int(np.flatnonzero(invalid)[0])
            h_label = f"mirrored {h_axis}" if mirror_horizontal else h_axis

            raise ValueError(
                f"Particle outside the {h_label}-{v_axis} grid in {path}: "
                f"time={time_values[first]}, "
                f"{h_label}={horizontal[first]}, "
                f"{v_axis}={vertical[first]}"
            )

        occupied_parts.append(
            pd.DataFrame(
                {
                    "time": time_values,
                    "h_index": h_index.astype(np.int32),
                    "v_index": v_index.astype(np.int32),
                }
            ).drop_duplicates()
        )

    if rows == 0:
        raise ValueError(
            f"No positions found in {path} between "
            f"{start_time:g} and {end_time:g} s"
        )

    occupied = pd.concat(
        occupied_parts,
        ignore_index=True,
    ).drop_duplicates()

    return occupied, times, rows


def highest_supported_boundary(occupied: np.ndarray) -> np.ndarray:
    """Highest occupied cell having another occupied cell directly below it."""
    boundary = np.zeros(occupied.shape[0], dtype=np.int64)
    adjacent = occupied[:, 1:] & occupied[:, :-1]
    for h_index in range(occupied.shape[0]):
        candidates = np.flatnonzero(adjacent[h_index])
        if candidates.size:
            boundary[h_index] = int(candidates[-1] + 2)  # 1-based cell row
    return boundary


def zero_below_wall(boundary: np.ndarray) -> np.ndarray:
    """Zero out boundary columns whose height falls below the circular wall.

    This correction is applied after the boundary has been averaged over
    Time_Range, matching the original Fortran workflow.
    """
    grid_count = boundary.size
    radius = drum_diameter / 2.0
    h_local = (np.arange(grid_count, dtype=float) + 0.5) * grid_size
    radicand = radius**2 - (h_local - radius) ** 2
    if (radicand < -1.0e-12).any():
        raise ValueError("A grid-column centre lies outside the circular drum")
    lower_wall = radius - np.sqrt(np.clip(radicand, 0.0, None))
    boundary_local = (boundary.astype(float) - 0.5) * grid_size
    boundary = boundary.copy()
    boundary[boundary_local < lower_wall] = 0
    return boundary


def average_boundary(occupied: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    grid_count = math.ceil(drum_diameter / grid_size)
    times = np.sort(occupied["time"].unique())
    boundary_sum = np.zeros(grid_count, dtype=np.int64)

    for _time, group in occupied.groupby("time", sort=True):
        grid = np.zeros((grid_count, grid_count), dtype=bool)
        grid[
            group["h_index"].to_numpy(dtype=np.int64),
            group["v_index"].to_numpy(dtype=np.int64),
        ] = True
        boundary_sum += highest_supported_boundary(grid)

    # All values are nonnegative; floor(x+0.5) matches Fortran NINT.
    boundary = np.floor(boundary_sum / len(times) + 0.5).astype(np.int64)
    boundary = zero_below_wall(boundary)
    return boundary, times


def fit_boundary(
    boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_local = (np.arange(boundary.size, dtype=float) + 0.5) * grid_size
    v_local = (boundary.astype(float) - 0.5) * grid_size
    valid = boundary != 0
    if np.count_nonzero(valid) < 4:
        raise ValueError("Fewer than four valid boundary points remain")

    polynomial = np.polynomial.Polynomial.fit(
        h_local[valid], v_local[valid], deg=3
    ).convert()
    coefficients = np.asarray(polynomial.coef, dtype=float)
    return coefficients, h_local, v_local, polynomial(h_local)


def secant_information(boundary: np.ndarray) -> tuple[np.ndarray, float]:
    valid_indices = np.flatnonzero(boundary != 0)
    if valid_indices.size < 2:
        raise ValueError("Fewer than two valid boundary points remain")
    values = boundary[valid_indices]
    maximum = values.max()
    minimum = values.min()
    max_index = int(valid_indices[np.flatnonzero(values == maximum)[0]])
    min_index = int(valid_indices[np.flatnonzero(values == minimum)[-1]])

    vector = np.array(
        [
            (min_index - max_index) * grid_size,
            (int(minimum) - int(maximum)) * grid_size,
        ],
        dtype=float,
    )
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("The selected free-surface endpoints are identical")
    vector /= norm
    raw_angle = math.degrees(math.atan2(vector[1], vector[0]))
    # A surface line has no preferred arrow direction. Report the equivalent
    # angle in [-90, 90) so a slope of +0.5 is shown as +26.565 degrees rather
    # than the same line pointing backwards at -153.435 degrees.
    angle_degrees = (raw_angle + 90.0) % 180.0 - 90.0
    return vector, angle_degrees


# ======================================================================
# Optional kinetic-energy calculation
# ======================================================================

def species_mass_and_inertia(
    folder: Path,
    code: str,
    shape: str,
    species: str,
) -> tuple[float, float] | None:
    """Return (mass_kg, trace_of_inertia_kg_m2), or None if unavailable.

    The particle dimension is inferred from mass and material density. Both
    supported shapes have an isotropic inertia tensor, so its trace is three
    times one diagonal component.
    """
    mass_path = folder / species / "masses.txt"
    if not mass_path.is_file():
        return None
    _unit, factor = mass_factor_to_kg(code)
    mass = first_value(mass_path, MASS_COLUMNS, "mass") * factor
    volume = mass / particle_density_kg_m3

    if shape == "Spheres":
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        diagonal_inertia = 2.0 * mass * radius**2 / 5.0
    elif shape == "Cubes":
        side = volume ** (1.0 / 3.0)
        diagonal_inertia = mass * side**2 / 6.0
    else:
        raise ValueError(f"Unsupported shape {shape!r}")

    return mass, 3.0 * diagonal_inertia


def paired_energy(
    velocity_path: Path,
    angular_path: Path,
    mass: float,
    inertia_trace: float,
) -> tuple[float, int]:
    """Calculate energy while validating velocity/angular row alignment.

    This uses the same rotational form as the Fortran source:
    (Ixx + Iyy + Izz) * |omega|^2 / 6. For spheres and cubes the three
    diagonal components are equal, so this is exactly 0.5 * I * |omega|^2.
    """
    velocity_reader = text_chunks(velocity_path, VELOCITY_COLUMNS)
    angular_reader = text_chunks(angular_path, ANGULAR_COLUMNS)
    total = 0.0
    rows = 0

    for velocity, angular in zip_longest(velocity_reader, angular_reader):
        if velocity is None or angular is None or len(velocity) != len(angular):
            raise ValueError(
                f"Velocity/angular files have different lengths: "
                f"{velocity_path}, {angular_path}"
            )
        if not np.array_equal(
            velocity["id"].to_numpy(dtype=np.int64),
            angular["id"].to_numpy(dtype=np.int64),
        ) or not np.allclose(
            velocity["time"].to_numpy(dtype=float),
            angular["time"].to_numpy(dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Velocity/angular rows are not aligned: {velocity_path}, {angular_path}"
            )

        v2 = np.square(velocity[["vx", "vy", "vz"]].to_numpy(dtype=float)).sum(axis=1)
        w2 = np.square(angular[["wx", "wy", "wz"]].to_numpy(dtype=float)).sum(axis=1)
        total += float((0.5 * mass * v2 + inertia_trace * w2 / 6.0).sum())
        rows += len(velocity)
    return total, rows


def calculate_dataset_energy(folder: Path, code: str, shape: str) -> tuple[float, str]:
    if not calculate_kinetic_energy:
        return math.nan, "disabled"

    total = 0.0
    for species in ("Particle_A", "Particle_B"):
        properties = species_mass_and_inertia(folder, code, shape, species)
        velocity_path = folder / species / "velocities.txt"
        angular_path = folder / species / "angular_velocities.txt"
        if properties is None or not velocity_path.is_file() or not angular_path.is_file():
            return math.nan, f"skipped: incomplete mass/velocity data for {species}"
        mass, inertia_trace = properties
        energy, _rows = paired_energy(
            velocity_path, angular_path, mass, inertia_trace
        )
        total += energy

    return total, "calculated (translational + rotational)"


# ======================================================================
# Dataset processing and outputs
# ======================================================================

def write_surface_txt(
    path: Path,
    boundary: np.ndarray,
    h_local: np.ndarray,
    v_local: np.ndarray,
    v_fit_local: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                f"{horizontal_axis}_local",
                f"{vertical_axis}_local",
                f"{vertical_axis}_fit_local",
                f"{horizontal_axis}_global",
                f"{vertical_axis}_global",
                f"{vertical_axis}_fit_global",
                "valid_boundary",
            ]
        )
        for i in range(len(boundary)):
            writer.writerow(
                [
                    h_local[i],
                    v_local[i],
                    v_fit_local[i],
                    h_local[i] + drum_centre[0] - drum_diameter / 2.0,
                    v_local[i] + drum_centre[1] - drum_diameter / 2.0,
                    v_fit_local[i] + drum_centre[1] - drum_diameter / 2.0,
                    int(boundary[i] != 0),
                ]
            )

def inferred_particle_diameter(
    folder: Path,
    code: str,
    shape: str,
    species: str,
) -> float:
    """Infer sphere diameter or cube side length from mass and density."""
    mass_path = folder / species / "masses.txt"
    _unit, factor = mass_factor_to_kg(code)
    mass_kg = first_value(mass_path, MASS_COLUMNS, "mass") * factor
    volume = mass_kg / particle_density_kg_m3

    if shape == "Spheres":
        radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
        return 2.0 * radius
    if shape == "Cubes":
        return volume ** (1.0 / 3.0)

    raise ValueError(f"Unsupported shape {shape!r}")


def read_positions_at_times(
    path: Path,
    actual_times: list[float],
    code: str,
) -> dict[float, pd.DataFrame]:
    """Read and transform the requested position frames."""
    actual_times = sorted(set(actual_times))
    parts: dict[float, list[pd.DataFrame]] = {
        time: [] for time in actual_times
    }

    h_axis, v_axis, _mirror = coordinate_settings(code)
    needed = list(dict.fromkeys(["time", h_axis, v_axis]))

    for chunk in text_chunks(path, POSITION_COLUMNS, needed):
        chunk_times = chunk["time"].to_numpy(dtype=float)

        for actual_time in actual_times:
            mask = np.isclose(
                chunk_times,
                actual_time,
                rtol=0.0,
                atol=snapshot_time_tolerance_s,
            )

            if not mask.any():
                continue

            selected = chunk.loc[mask]
            horizontal, vertical = projected_coordinates(selected, code)

            parts[actual_time].append(
                pd.DataFrame(
                    {
                        "horizontal": horizontal,
                        "vertical": vertical,
                    }
                )
            )

    empty = pd.DataFrame(columns=["horizontal", "vertical"])

    return {
        time: (
            pd.concat(time_parts, ignore_index=True)
            if time_parts
            else empty.copy()
        )
        for time, time_parts in parts.items()
    }


def secant_plot_endpoints(
    boundary: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the global coordinates of the two secant endpoints."""
    valid_indices = np.flatnonzero(boundary != 0)
    if valid_indices.size < 2:
        raise ValueError("Fewer than two valid boundary points remain")

    values = boundary[valid_indices]
    maximum = values.max()
    minimum = values.min()

    max_index = int(valid_indices[np.flatnonzero(values == maximum)[0]])
    min_index = int(valid_indices[np.flatnonzero(values == minimum)[-1]])

    origin_h = drum_centre[0] - drum_diameter / 2.0
    origin_v = drum_centre[1] - drum_diameter / 2.0

    def cell_centre(index: int) -> tuple[float, float]:
        return (
            origin_h + (index + 0.5) * grid_size,
            origin_v + (boundary[index] - 0.5) * grid_size,
        )

    return cell_centre(max_index), cell_centre(min_index)


def prepare_snapshot_record(
    folder: Path,
    code: str,
    shape: str,
    actual_time: float,
    average_boundary_values: np.ndarray,
    average_angle_degrees: float,
) -> dict[str, object]:
    """Load one particle frame and pair it with the averaged secant."""
    species_names = ("Particle_A", "Particle_B")
    positions = {
        species: read_positions_at_times(
            folder / species / "positions.txt",
            [actual_time],
            code,
        )[actual_time]
        for species in species_names
    }

    diameters = {
        species: inferred_particle_diameter(folder, code, shape, species)
        for species in species_names
    }
    ranked_species = sorted(species_names, key=diameters.get)
    small_species = ranked_species[0]
    large_species = ranked_species[-1]
    largest_diameter = max(diameters.values())
    marker_areas = {
        species: max(
            4.0,
            22.0 * (diameters[species] / largest_diameter) ** 2,
        )
        for species in species_names
    }

    return {
        "code": code,
        "shape": shape,
        "actual_time": actual_time,
        "positions": positions,
        "small_species": small_species,
        "large_species": large_species,
        "marker_areas": marker_areas,
        "secant_points": secant_plot_endpoints(average_boundary_values),
        "average_angle_degrees": average_angle_degrees,
    }


def plot_shape_snapshots(records: list[dict[str, object]]) -> Path:
    """Plot one panel per code, grouped into a single figure by shape."""
    if not records:
        raise ValueError("No snapshot records were supplied")

    shape = str(records[0]["shape"])
    if any(record["shape"] != shape for record in records):
        raise ValueError("A grouped snapshot figure must contain one shape only")

    panel_count = len(records)
    column_count = 1 if panel_count == 1 else 2
    row_count = math.ceil(panel_count / column_count)
    marker = "o" if shape == "Spheres" else "s"
    drum_radius = drum_diameter / 2.0

    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 5.2 * row_count),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for axis, record in zip(axes.ravel(), records):
        axis.add_patch(
            Circle(
                drum_centre,
                drum_radius,
                fill=False,
                color="black",
                linewidth=1.5,
                zorder=5,
            )
        )

        positions = record["positions"]
        large_species = str(record["large_species"])
        small_species = str(record["small_species"])
        marker_areas = record["marker_areas"]
        colours = {
            large_species: "tab:blue",
            small_species: "tab:red",
        }

        for species in (large_species, small_species):
            frame = positions[species]
            if frame.empty:
                continue
            axis.scatter(
                frame["horizontal"],
                frame["vertical"],
                s=marker_areas[species],
                marker=marker,
                color=colours[species],
                alpha=0.8,
                linewidths=0,
                rasterized=True,
                zorder=2,
            )

        point_1, point_2 = record["secant_points"]
        axis.plot(
            [point_1[0], point_2[0]],
            [point_1[1], point_2[1]],
            color=snapshot_secant_color,
            linewidth=2.5,
            zorder=6,
        )

        axis.set_title(
            f"{record['code']}\n"
            f"frame t={record['actual_time']:g} s; "
            f"average angle={record['average_angle_degrees']:.2f}°"
        )
        axis.set_aspect("equal")
        axis.set_xlim(
            drum_centre[0] - drum_radius,
            drum_centre[0] + drum_radius,
        )
        axis.set_ylim(
            drum_centre[1] - drum_radius,
            drum_centre[1] + drum_radius,
        )
        axis.set_xlabel("Horizontal position (m)")
        axis.set_ylabel("Vertical position (m)")
        axis.grid(alpha=0.15)

    for unused_axis in axes.ravel()[panel_count:]:
        unused_axis.set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="tab:blue",
            markeredgewidth=0,
            markersize=8,
            label="Large particles",
        ),
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor="tab:red",
            markeredgewidth=0,
            markersize=6,
            label="Small particles",
        ),
        Line2D(
            [0],
            [0],
            color=snapshot_secant_color,
            linewidth=2.5,
            label="Secant of averaged surface",
        ),
    ]

    start_time, end_time = map(float, Time_Range)
    figure.suptitle(
        f"{shape}: particle positions and {start_time:g}–{end_time:g} s "
        "averaged free surface",
        fontsize=15,
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))

    results_root.mkdir(parents=True, exist_ok=True)
    start_label = f"{start_time:g}".replace(".", "p")
    end_label = f"{end_time:g}".replace(".", "p")
    safe_shape = shape.replace(" ", "_")
    output_path = results_root / (
        f"{safe_shape}_averaged_free_surface_{start_label}-{end_label}s.png"
    )
    figure.savefig(
        output_path,
        dpi=snapshot_dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return output_path

def process_dataset(
    code: str,
    shape: str,
) -> tuple[dict[str, object], dict[str, object]]:
    folder = dataset_folder(code, shape)
    print(f"\n=== {code} / {shape} ===")
    occupied_parts = []
    species_times: dict[str, set[float]] = {}
    position_rows = 0

    for species in ("Particle_A", "Particle_B"):
        path = folder / species / "positions.txt"
        occupied, times, rows = occupied_cells_from_positions(path, code)
        occupied_parts.append(occupied)
        species_times[species] = times
        position_rows += rows
        print(f"{species}: {rows:,} position rows, {len(times)} timesteps")

    if species_times["Particle_A"] != species_times["Particle_B"]:
        only_a = len(species_times["Particle_A"] - species_times["Particle_B"])
        only_b = len(species_times["Particle_B"] - species_times["Particle_A"])
        print(
            f"WARNING: unmatched times (A-only={only_a}, B-only={only_b}); "
            "using the union",
            file=sys.stderr,
        )

    occupied = pd.concat(occupied_parts, ignore_index=True).drop_duplicates()
    boundary, times = average_boundary(occupied)
    coefficients, h_local, v_local, v_fit = fit_boundary(boundary)
    secant, angle_degrees = secant_information(boundary)
    energy, energy_status = calculate_dataset_energy(folder, code, shape)

    common_times = np.array(
        sorted(species_times["Particle_A"] & species_times["Particle_B"]),
        dtype=float,
    )
    if common_times.size == 0:
        raise ValueError(
            f"No common Particle_A/Particle_B timestep in Time_Range={Time_Range}"
        )
    actual_snapshot_time = float(
        common_times[np.argmin(np.abs(common_times - snapshot_time_s))]
    )
    snapshot_record = prepare_snapshot_record(
        folder,
        code,
        shape,
        actual_snapshot_time,
        boundary,
        angle_degrees,
    )

    output_dir = results_root / code / shape
    write_surface_txt(
        output_dir / "visu_fit_surf_libre.txt",
        boundary,
        h_local,
        v_local,
        v_fit,
    )
    print(f"Cubic coefficients: {coefficients}")
    print(
        f"Average free-surface angle over {Time_Range[0]:g}–"
        f"{Time_Range[1]:g} s: {angle_degrees:.6f} degrees"
    )
    print(f"Kinetic energy: {energy_status}")
    print(f"Snapshot frame selected: t={actual_snapshot_time:g} s")
    summary = {
        "software": code,
        "shape": shape,
        "average_start_s": float(Time_Range[0]),
        "average_end_s": float(Time_Range[1]),
        "snapshot_time_s": actual_snapshot_time,
        "timesteps": len(times),
        "position_rows": position_rows,
        "valid_boundary_points": int(np.count_nonzero(boundary)),
        "c0": coefficients[0],
        "c1": coefficients[1],
        "c2": coefficients[2],
        "c3": coefficients[3],
        "secant_horizontal": secant[0],
        "secant_vertical": secant[1],
        "surface_angle_degrees": angle_degrees,
        "total_kinetic_energy": energy,
        "energy_status": energy_status,
    }
    return summary, snapshot_record

def write_dataset_diagnostic(
    folder: Path,
    code: str,
    shape: str,
    error_details: str,
) -> Path:
    """Write bounded diagnostics without loading complete data files."""
    results_root.mkdir(parents=True, exist_ok=True)

    report_path = results_root / f"{code}_{shape}_diagnostic.txt"
    h_axis, v_axis, mirror_horizontal = coordinate_settings(code)

    lines = [
        f"Dataset diagnostic: {code} / {shape}",
        f"Dataset folder: {folder}",
        "",
        "Coordinate configuration",
        f"  horizontal source axis: {h_axis}",
        f"  vertical source axis: {v_axis}",
        f"  mirror horizontal: {mirror_horizontal}",
        f"  drum centre: {drum_centre}",
        f"  drum diameter: {drum_diameter}",
        f"  mass unit: {mass_units_by_code.get(code, 'kg')}",
        "",
        "Original exception",
        error_details,
        "",
        "Input-file inspection",
    ]

    file_definitions = {
        "positions.txt": POSITION_COLUMNS,
        "velocities.txt": VELOCITY_COLUMNS,
        "angular_velocities.txt": ANGULAR_COLUMNS,
        "masses.txt": MASS_COLUMNS,
    }

    for species in ("Particle_A", "Particle_B"):
        lines.extend(["", f"[{species}]"])

        for filename, columns in file_definitions.items():
            path = folder / species / filename
            lines.append(f"\n{filename}")
            lines.append(f"  path: {path}")

            if not path.is_file():
                lines.append("  status: MISSING")
                continue

            try:
                lines.append(f"  size: {path.stat().st_size:,} bytes")
            except OSError as exc:
                lines.append(f"  size check failed: {exc}")

            try:
                sample = pd.read_csv(
                    path,
                    sep=r"\s+",
                    comment="#",
                    names=columns,
                    nrows=5,
                )

                lines.append(f"  parsed sample rows: {len(sample)}")

                if sample.empty:
                    lines.append("  status: FILE CONTAINS NO DATA ROWS")
                    continue

                all_nan_columns = [
                    column
                    for column in sample.columns
                    if sample[column].isna().all()
                ]

                if all_nan_columns:
                    lines.append(
                        "  columns containing only NaN: "
                        + ", ".join(all_nan_columns)
                    )

                lines.append("  first parsed rows:")
                sample_text = sample.to_string(index=False)
                lines.extend(
                    f"    {line}" for line in sample_text.splitlines()
                )

                if filename == "positions.txt":
                    try:
                        horizontal, vertical = projected_coordinates(
                            sample,
                            code,
                        )

                        lines.append(
                            "  projected horizontal sample range: "
                            f"{horizontal.min()} to {horizontal.max()}"
                        )
                        lines.append(
                            "  projected vertical sample range: "
                            f"{vertical.min()} to {vertical.max()}"
                        )

                        h_min = drum_centre[0] - drum_diameter / 2.0
                        h_max = drum_centre[0] + drum_diameter / 2.0
                        v_min = drum_centre[1] - drum_diameter / 2.0
                        v_max = drum_centre[1] + drum_diameter / 2.0

                        outside = (
                            (horizontal < h_min)
                            | (horizontal > h_max)
                            | (vertical < v_min)
                            | (vertical > v_max)
                        )

                        lines.append(
                            "  sample particles outside drum grid: "
                            f"{int(np.count_nonzero(outside))}"
                        )
                        lines.append(
                            f"  expected horizontal range: {h_min} to {h_max}"
                        )
                        lines.append(
                            f"  expected vertical range: {v_min} to {v_max}"
                        )

                    except Exception as exc:
                        lines.append(
                            "  coordinate transformation failed: "
                            f"{type(exc).__name__}: {exc}"
                        )

            except Exception as exc:
                lines.append(
                    "  parsing failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    report_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return report_path

def main() -> int:
    if horizontal_axis == vertical_axis or {horizontal_axis, vertical_axis} - {"x", "y", "z"}:
        print("error: cross-section axes must be two different values from x, y, z", file=sys.stderr)
        return 1
    if (
        len(Time_Range) != 2
        or not all(math.isfinite(float(value)) for value in Time_Range)
        or float(Time_Range[0]) > float(Time_Range[1])
    ):
        print(
            "error: Time_Range must contain two finite values with start <= end",
            file=sys.stderr,
        )
        return 1
    if (
        drum_diameter <= 0
        or grid_size <= 0
        or chunk_size < 1
        or particle_density_kg_m3 <= 0
    ):
        print(
            "error: drum_diameter, grid_size, chunk_size, and "
            "particle_density_kg_m3 must be positive",
            file=sys.stderr,
        )
        return 1

    summaries = []
    snapshot_records: list[dict[str, object]] = []
    for code, shape in configured_datasets():
        folder = dataset_folder(code, shape)
        required = [
            folder / "Particle_A" / "positions.txt",
            folder / "Particle_B" / "positions.txt",
        ]
        missing = [path for path in required if not path.is_file()]
        if missing:
            message = f"{code} / {shape} missing " + ", ".join(map(str, missing))
            if skip_missing_datasets:
                print(f"WARNING: skipping {message}", file=sys.stderr)
                continue
            raise FileNotFoundError(message)
        try:
            summary, snapshot_record = process_dataset(code, shape)
            summaries.append(summary)
            snapshot_records.append(snapshot_record)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            message = (
                f"WARNING: skipping {code} / {shape}: "
                f"{type(exc).__name__}: {exc}"
            )
            print(message, file=sys.stderr)
    
            if code == "YDEM":
                traceback_text = traceback.format_exc()
    
                # Print the complete traceback to the console.
                print(
                    "\nFull YDEM traceback:\n"
                    + traceback_text,
                    file=sys.stderr,
                )
    
                try:
                    diagnostic_path = write_dataset_diagnostic(
                        folder,
                        code,
                        shape,
                        traceback_text,
                    )
                    print(
                        f"YDEM diagnostic written to {diagnostic_path}",
                        file=sys.stderr,
                    )
                except Exception as diagnostic_error:
                    print(
                        "Could not write YDEM diagnostic: "
                        f"{type(diagnostic_error).__name__}: "
                        f"{diagnostic_error}",
                        file=sys.stderr,
                    )
    
            if skip_missing_datasets:
                continue
            raise

    if not summaries:
        print("error: no dataset was processed successfully", file=sys.stderr)
        return 1

    results_root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        results_root / "free_surface_summary.txt",
        sep="\t",
        index=False,
        na_rep="nan",
    )
    print("\nSummary:")
    print(summary.to_string(index=False))

    for shape in shapes:
        shape_records = [
            record for record in snapshot_records if record["shape"] == shape
        ]
        if not shape_records:
            continue
        figure_path = plot_shape_snapshots(shape_records)
        print(f"{shape} comparison figure: {figure_path}")

    print(f"\nResults written to {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
