#!/usr/bin/env python3
"""Filling-degree post-processing for all configured rotating-drum DEM codes.

Folder convention (same as the free-surface / secant-timeseries script):
    base_dir / code / shape / Particle_A|Particle_B / positions.txt, masses.txt

For every configured (code, shape) dataset this script:

  1. Reads particle positions at a single "settled" timestep (time_settled_s),
     i.e. a moment before the drum starts rotating, when mass is at rest and
     the free surface is (roughly) flat. Because no particles are created or
     destroyed, the *volumetric* filling degree of the drum is the same at
     every timestep, so any settled frame gives the right answer.
  2. Computes the bed height h0 along the vertical line through the drum
     centre (from the lowest point of the drum, (drum_centre_x, drum_centre_y
     - R), straight up) *analytically* from each particle's centre and
     radius -- no pixel grid / image analysis is involved, which avoids any
     ambiguity from cataracting particles at other times, and avoids
     grid-resolution error at the settled time itself.
  3. Reports:
       - J = h0 / (2R):            the bed-height filling degree, as defined
         in Félix, Falk & D'Ortona-style rotating-drum papers, e.g.
         https://pubs.aip.org/sor/jor/article/64/4/915/241514
       - f_bulk = A_segment(h0)/A_circle: the equivalent *area* (bulk, i.e.
         including voids) filling degree of the drum cross-section, obtained
         from h0 via the standard circular-segment formula. J and f_bulk
         both describe "how full the drum looks", assuming a flat bed top.
       - f_solid = V_particles / V_drum: the *solid* filling degree, i.e.
         the fraction of the drum's total volume that is actually occupied
         by particle material (mass/density), not by voids. This uses
         Sum(mass)/density directly -- exact, and independent of h0.
       - packing fraction = f_solid / f_bulk = V_solid / V_bed_bulk, the
         solids fraction *within* the bed (typically ~0.55-0.64 for random
         packing of monosize spheres).
       - void ratio e = (1 - packing_fraction) / packing_fraction
                       = V_void / V_solid
       - porosity = 1 - packing_fraction = V_void / V_bed_bulk

Converting f_solid to an absolute volume needs the drum's axial length.
Set drum_length_m explicitly if you know it (recommended); otherwise it is
estimated from the axial (out-of-plane) spread of particle centres at
time_settled, plus a margin of drum_length_margin_factor particle diameters
to roughly account for the particles poking out past their own centres.

All result tables are written as tab-delimited .txt files under
results_root. A verification plot per dataset (particles, drum outline,
centreline and h0) is written alongside them so the analytic h0 can be
sanity-checked by eye.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ======================================================================
# User configuration -- keep in sync with WG1_Leoben_Analysis_Secant_Timeseries.py
# ======================================================================
legendfont  = 16
labelfont   = 18
titlefont   = 16

base_dir = Path(
    r"C:\Users\mcvanbenten\OneDrive - Delft University of Technology"
    r"\Documents\ON-DEM\WG1_Benchmark_Rotating_Drum\Particle_Data"
)

shapes = ["Spheres", "Cubes"]
sphere_codes    = []#["EDEM", "Kratos-Multiphysics", "MercuryDPM", "Liggghts"]
cube_codes      = ["EDEM", "YDEM", "Yade"]  # Add "EDEM" when its cube data is ready.

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

drum_centre = (0.0, 0.0)
drum_diameter = 0.20

# Sanity-check tolerance (m) when verifying that projected particle centres
# actually fall inside the configured drum circle for a given code.
drum_centre_check_tolerance_m = 0.002

# Axial (out-of-plane / rotation-axis) length of the drum, in metres.
# Set this explicitly if you know it -- it is the most reliable option.
# Leave as None to estimate it from the axial spread of particle centres at
# time_settled (see drum_length_margin_factor below).
drum_length_m: float | None = None
drum_length_margin_factor = 1.0  # extra length added, in mean particle diameters

# Timestep considered "settled" (drum at rest, before rotation starts).
# Filling degree does not change over time, so any settled frame works;
# override per code if different codes log different settling times.
time_settled_s = 0.2
time_settled_overrides: dict[str, float] = {}  # e.g. {"YDEM": 0.05}
time_settled_tolerance_s = 1.0e-6

# Mass units in masses.txt. Unlisted codes default to kg.
mass_units_by_code = {
    "Kratos-Multiphysics": "g",
}
mass_unit_to_kg = {"kg": 1.0, "g": 1.0e-3, "mg": 1.0e-6}

# Material density used to convert particle mass to volume (assumed the
# same for both species and for the free-surface script).
particle_density_kg_m3 = 2500.0

chunk_size = 1_000_000
skip_missing_datasets = True
make_plots = True
results_root = base_dir / "Filling_Degree_Results"


# ======================================================================
# Input definitions
# ======================================================================

POSITION_COLUMNS = ["id", "time", "x", "y", "z"]
MASS_COLUMNS = ["id", "time", "mass"]

AXES = {"x", "y", "z"}


def coordinate_settings(code: str) -> tuple[str, str, bool]:
    return coordinate_rules.get(code, (horizontal_axis, vertical_axis, False))


def axial_axis_name(code: str) -> str:
    """The one coordinate axis not used for the cross-section (the drum's
    rotation / length axis)."""
    h_axis, v_axis, _mirror = coordinate_settings(code)
    remaining = AXES - {h_axis, v_axis}
    if len(remaining) != 1:
        raise ValueError(f"Could not determine a unique axial axis for {code}")
    return remaining.pop()


def projected_coordinates(frame: pd.DataFrame, code: str) -> tuple[np.ndarray, np.ndarray]:
    """Transform a code's coordinates into the common drum coordinate system.

    Identical convention to the free-surface script, so a drum_centre /
    source_drum_centres fix made there applies here too.
    """
    h_axis, v_axis, mirror_horizontal = coordinate_settings(code)

    horizontal = frame[h_axis].to_numpy(dtype=float)
    vertical = frame[v_axis].to_numpy(dtype=float)

    source_centre = source_drum_centres.get(code, drum_centre)
    horizontal = horizontal - source_centre[0] + drum_centre[0]
    vertical = vertical - source_centre[1] + drum_centre[1]

    if mirror_horizontal:
        horizontal = 2.0 * drum_centre[0] - horizontal

    return horizontal, vertical


def dataset_folder(code: str, shape: str) -> Path:
    return base_dir / code / shape


def configured_datasets() -> list[tuple[str, str]]:
    codes_by_shape = {"Spheres": sphere_codes, "Cubes": cube_codes}
    unknown = set(shapes) - set(codes_by_shape)
    if unknown:
        raise ValueError(f"No code list is defined for shape(s): {sorted(unknown)}")
    return [(code, shape) for shape in shapes for code in codes_by_shape[shape]]


def text_chunks(path: Path, columns: list[str], usecols: list[str] | None = None):
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
            f"Unknown mass unit {unit!r} for {code}; choose from {sorted(mass_unit_to_kg)}"
        )
    return unit, mass_unit_to_kg[unit]


# ======================================================================
# Settled-frame reading
# ======================================================================

def discover_available_times(path: Path) -> np.ndarray:
    times: set[float] = set()
    for chunk in text_chunks(path, POSITION_COLUMNS, ["time"]):
        times.update(map(float, chunk["time"].unique()))
    return np.array(sorted(times))


def nearest_available_time(available: np.ndarray, requested: float) -> float | None:
    if available.size == 0:
        return None
    return float(available[np.argmin(np.abs(available - requested))])


def read_full_frame_at_time(path: Path, actual_time: float) -> pd.DataFrame | None:
    """Read every column for the rows matching actual_time (all particles
    of one species at one timestep)."""
    rows = []
    for chunk in text_chunks(path, POSITION_COLUMNS):
        mask = np.isclose(
            chunk["time"].to_numpy(dtype=float),
            actual_time,
            rtol=0.0,
            atol=max(time_settled_tolerance_s, 1.0e-9),
        )
        if mask.any():
            rows.append(chunk.loc[mask])
    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)


def resolve_time_settled(code: str) -> float:
    return time_settled_overrides.get(code, time_settled_s)


def species_mass_and_volume(folder: Path, code: str, species: str) -> tuple[float, float] | None:
    """Return (mass_per_particle_kg, volume_per_particle_m3), or None if
    masses.txt is missing. Assumes a monodisperse species, as elsewhere."""
    mass_path = folder / species / "masses.txt"
    if not mass_path.is_file():
        return None
    _unit, factor = mass_factor_to_kg(code)
    mass_kg = first_value(mass_path, MASS_COLUMNS, "mass") * factor
    volume_m3 = mass_kg / particle_density_kg_m3
    return mass_kg, volume_m3


def effective_sphere_radius(volume_m3: float) -> float:
    """Volume-equivalent sphere radius. Used for both spheres (exact) and
    cubes (an approximation -- particle orientation isn't in positions.txt,
    so an exact cube footprint along the centreline isn't recoverable)."""
    return (3.0 * volume_m3 / (4.0 * math.pi)) ** (1.0 / 3.0)


# ======================================================================
# Filling-degree geometry
# ======================================================================

def bed_height_at_centreline(particles: pd.DataFrame, centre_h: float) -> tuple[float, int]:
    """Analytic top-of-bed height where the vertical line x=centre_h
    crosses the highest particle it intersects.

    For a particle centred at (h, v) with radius r, the line x=centre_h
    crosses its circle (if it does) at v +/- sqrt(r^2 - (h - centre_h)^2).
    The bed surface at the centreline is the highest such upper crossing
    over every particle the line actually passes through -- exact, and
    independent of any grid or image resolution.
    """
    dx = particles["h"].to_numpy(dtype=float) - centre_h
    r = particles["r"].to_numpy(dtype=float)
    touches_line = np.abs(dx) <= r

    if not touches_line.any():
        # Fall back to the single particle whose circle passes closest to
        # the centreline (a sparse bed or an unlucky packing gap exactly
        # on the line).
        closest = int(np.argmin(np.abs(dx) - r))
        touches_line = np.zeros_like(dx, dtype=bool)
        touches_line[closest] = True

    top_y = particles["v"].to_numpy(dtype=float)[touches_line] + np.sqrt(
        np.clip(r[touches_line] ** 2 - dx[touches_line] ** 2, 0.0, None)
    )
    return float(top_y.max()), int(touches_line.sum())


def segment_area_fraction(h0: float, radius: float) -> float:
    """Area of the circular segment from the bottom of the circle up to
    height h0, divided by the full circle area (the 'bulk', flat-top-bed
    filling degree implied by h0)."""
    h0_clipped = min(max(h0, 0.0), 2.0 * radius)
    theta = 2.0 * math.acos(1.0 - h0_clipped / radius)
    segment_area = radius**2 * (theta - math.sin(theta)) / 2.0
    return segment_area / (math.pi * radius**2)


def verify_drum_centre(all_particles: pd.DataFrame, code: str) -> None:
    """Warn if particle centres fall outside the configured drum circle --
    a sign drum_centre / source_drum_centres needs adjusting for this code."""
    radius = drum_diameter / 2.0
    dist = np.hypot(
        all_particles["h"].to_numpy(dtype=float) - drum_centre[0],
        all_particles["v"].to_numpy(dtype=float) - drum_centre[1],
    )
    allowance = all_particles["r"].to_numpy(dtype=float) + drum_centre_check_tolerance_m
    outside = dist > radius + allowance
    if outside.any():
        worst = float(dist[outside].max())
        print(
            f"WARNING [{code}]: {int(outside.sum())} particle centre(s) fall outside the "
            f"configured drum circle (farthest {worst:.4f} m from drum_centre, drum radius "
            f"is {radius:.4f} m). Check drum_centre / source_drum_centres for {code}.",
            file=sys.stderr,
        )


def resolve_drum_length(axial_values: np.ndarray, mean_diameter: float) -> tuple[float, str]:
    if drum_length_m is not None:
        return drum_length_m, "configured"
    span = float(axial_values.max() - axial_values.min())
    length = span + drum_length_margin_factor * mean_diameter
    return length, "estimated from axial particle spread"


# ======================================================================
# Plotting
# ======================================================================

def plot_filling_degree(
    code: str,
    shape: str,
    species_frames: dict[str, pd.DataFrame],
    h0: float,
) -> Path:
    radius_drum = drum_diameter / 2.0
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.add_patch(
        Circle(drum_centre, radius_drum, fill=False, color="black", linewidth=1.5, zorder=5)
    )

    colours = ["tab:blue", "tab:red"]
    for colour, (species, frame) in zip(colours, species_frames.items()):
        axis.scatter(
            frame["h"],
            frame["v"],
            s=6,
            alpha=0.6,
            color=colour,
            linewidths=0,
            label=species,
            rasterized=True,
            zorder=2,
        )

    bottom_y = drum_centre[1] - radius_drum
    axis.axvline(drum_centre[0], color="grey", linestyle="--", linewidth=1, zorder=3)
    axis.plot(
        [drum_centre[0] - radius_drum, drum_centre[0] + radius_drum],
        [bottom_y + h0, bottom_y + h0],
        color="tab:green",
        linewidth=2,
        zorder=6,
        label=f"h0 = {h0 * 1000:.1f} mm",
    )
    axis.set_aspect("equal")
    axis.set_xlim(drum_centre[0] - radius_drum * 1.05, drum_centre[0] + radius_drum * 1.05)
    axis.set_ylim(drum_centre[1] - radius_drum * 1.05, drum_centre[1] + radius_drum * 1.05)
    axis.set_xlabel("horizontal position (m)")
    axis.set_ylabel("vertical position (m)")
    axis.set_title(f"{code} — {shape}: J = h0/(2R) = {h0 / drum_diameter:.3f}")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(alpha=0.15)

    results_root.mkdir(parents=True, exist_ok=True)
    safe_code = code.replace(" ", "_")
    safe_shape = shape.replace(" ", "_")
    output_path = results_root / f"{safe_code}_{safe_shape}_filling_degree.png"
    figure.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


# ======================================================================
# Dataset processing
# ======================================================================

def process_dataset(code: str, shape: str) -> dict[str, object]:
    folder = dataset_folder(code, shape)
    print(f"\n=== {code} / {shape} ===")

    requested_time = resolve_time_settled(code)
    axial_axis = axial_axis_name(code)

    # Both species must be read at the SAME instant, or the combined bed
    # picture (and hence h0) would mix two different physical configurations.
    # Find the settled time as the match nearest to requested_time within
    # the *intersection* of timesteps available across every species present.
    species_paths = {
        species: folder / species / "positions.txt"
        for species in ("Particle_A", "Particle_B")
        if (folder / species / "positions.txt").is_file()
    }
    if not species_paths:
        raise ValueError(f"No positions.txt found for {code} / {shape}")

    available_by_species = {
        species: discover_available_times(path) for species, path in species_paths.items()
    }
    if len(available_by_species) > 1:
        common = set.intersection(*(set(t) for t in available_by_species.values()))
        if not common:
            print(
                f"WARNING [{code}]: Particle_A and Particle_B share no common logged "
                "timestep; falling back to each species' own nearest match, which may "
                "mix two different physical instants in the combined bed.",
                file=sys.stderr,
            )
            common_times = None
        else:
            common_times = np.array(sorted(common))
    else:
        common_times = next(iter(available_by_species.values()))

    shared_actual_time = (
        nearest_available_time(common_times, requested_time)
        if common_times is not None
        else None
    )

    species_frames: dict[str, pd.DataFrame] = {}
    species_props: dict[str, dict[str, float]] = {}

    for species, pos_path in species_paths.items():
        if shared_actual_time is not None:
            actual_time = shared_actual_time
        else:
            actual_time = nearest_available_time(available_by_species[species], requested_time)
        if actual_time is None:
            print(f"  {species}: positions.txt has no data, skipping species", file=sys.stderr)
            continue

        raw = read_full_frame_at_time(pos_path, actual_time)
        if raw is None or raw.empty:
            continue

        mass_volume = species_mass_and_volume(folder, code, species)
        if mass_volume is None:
            print(f"  {species}: no masses.txt, cannot compute volume, skipping species", file=sys.stderr)
            continue
        mass_kg, volume_m3 = mass_volume
        radius = effective_sphere_radius(volume_m3)

        horizontal, vertical = projected_coordinates(raw, code)
        axial = raw[axial_axis].to_numpy(dtype=float)

        frame = pd.DataFrame({"h": horizontal, "v": vertical, "axial": axial})
        frame["r"] = radius
        species_frames[species] = frame
        species_props[species] = {
            "count": len(frame),
            "mass_per_particle_kg": mass_kg,
            "volume_per_particle_m3": volume_m3,
            "effective_radius_m": radius,
            "actual_time_s": actual_time,
        }

        note = (
            ""
            if math.isclose(actual_time, requested_time, abs_tol=time_settled_tolerance_s)
            else f" (nearest available to requested {requested_time:g} s)"
        )
        print(
            f"  {species}: {len(frame):,} particles at t = {actual_time:g} s{note}, "
            f"r_eff = {radius * 1000:.3f} mm"
        )

    if not species_frames:
        raise ValueError(f"No usable positions/mass data for {code} / {shape}")

    all_particles = pd.concat(species_frames.values(), ignore_index=True)
    verify_drum_centre(all_particles, code)

    radius_drum = drum_diameter / 2.0
    top_y, n_touching = bed_height_at_centreline(all_particles, drum_centre[0])
    bottom_y = drum_centre[1] - radius_drum
    h0 = top_y - bottom_y

    if h0 < 0.0 or h0 > drum_diameter:
        print(
            f"WARNING [{code}]: h0={h0:.5f} m falls outside [0, {drum_diameter:.5f}] m; "
            "clipping to the physical range.",
            file=sys.stderr,
        )
    h0_clipped = min(max(h0, 0.0), drum_diameter)

    J = h0_clipped / drum_diameter
    f_bulk = segment_area_fraction(h0_clipped, radius_drum)

    total_mass_kg = sum(
        props["mass_per_particle_kg"] * props["count"] for props in species_props.values()
    )
    total_solid_volume_m3 = total_mass_kg / particle_density_kg_m3

    counts = [props["count"] for props in species_props.values()]
    diameters = [2.0 * props["effective_radius_m"] for props in species_props.values()]
    mean_diameter = float(np.average(diameters, weights=counts))

    length_m, length_source = resolve_drum_length(
        all_particles["axial"].to_numpy(dtype=float), mean_diameter
    )
    drum_volume_m3 = math.pi * radius_drum**2 * length_m
    bed_bulk_volume_m3 = f_bulk * drum_volume_m3

    f_solid = total_solid_volume_m3 / drum_volume_m3
    if bed_bulk_volume_m3 > 0.0:
        packing_fraction = total_solid_volume_m3 / bed_bulk_volume_m3
        void_ratio = (bed_bulk_volume_m3 - total_solid_volume_m3) / total_solid_volume_m3
        porosity = 1.0 - packing_fraction
    else:
        packing_fraction = void_ratio = porosity = math.nan

    print(f"  h0 = {h0_clipped * 1000:.3f} mm  ->  J = h0/(2R) = {J:.4f}")
    print(f"  bulk (area) filling degree f_bulk = {f_bulk:.4f}")
    print(f"  drum length used: {length_m * 1000:.2f} mm ({length_source})")
    print(f"  solid filling degree f_solid = V_particles/V_drum = {f_solid:.4f}")
    print(
        f"  packing fraction (solid/bulk) = {packing_fraction:.4f}, "
        f"void ratio e = {void_ratio:.4f}, porosity = {porosity:.4f}"
    )

    if make_plots:
        plot_path = plot_filling_degree(code, shape, species_frames, h0_clipped)
        print(f"  Plot: {plot_path}")

    return {
        "software": code,
        "shape": shape,
        "time_settled_requested_s": requested_time,
        "particle_count_total": len(all_particles),
        "centreline_particles_used": n_touching,
        "h0_m": h0_clipped,
        "drum_radius_m": radius_drum,
        "J_filling_degree_h0_over_2R": J,
        "f_bulk_area_filling_degree": f_bulk,
        "drum_length_m": length_m,
        "drum_length_source": length_source,
        "total_solid_volume_m3": total_solid_volume_m3,
        "drum_volume_m3": drum_volume_m3,
        "f_solid_filling_degree": f_solid,
        "packing_fraction": packing_fraction,
        "void_ratio": void_ratio,
        "porosity": porosity,
    }


def main() -> int:
    if drum_diameter <= 0 or particle_density_kg_m3 <= 0 or chunk_size < 1:
        print(
            "error: drum_diameter, particle_density_kg_m3, and chunk_size must be positive",
            file=sys.stderr,
        )
        return 1
    if drum_length_m is not None and drum_length_m <= 0:
        print("error: drum_length_m must be positive if set", file=sys.stderr)
        return 1

    summaries = []
    for code, shape in configured_datasets():
        folder = dataset_folder(code, shape)
        required = [folder / "Particle_A" / "positions.txt", folder / "Particle_A" / "masses.txt"]
        missing = [path for path in required if not path.is_file()]
        if missing:
            message = f"{code} / {shape} missing " + ", ".join(map(str, missing))
            if skip_missing_datasets:
                print(f"WARNING: skipping {message}", file=sys.stderr)
                continue
            raise FileNotFoundError(message)
        try:
            summaries.append(process_dataset(code, shape))
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            message = f"WARNING: skipping {code} / {shape}: {type(exc).__name__}: {exc}"
            print(message, file=sys.stderr)
            if skip_missing_datasets:
                continue
            raise

    if not summaries:
        print("error: no dataset was processed successfully", file=sys.stderr)
        return 1

    results_root.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        results_root / "filling_degree_summary.txt",
        sep="\t",
        index=False,
        na_rep="nan",
    )
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nResults written to {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())