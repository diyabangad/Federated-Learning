"""Compare SGP4 Walker states with a circular two-body TEME-aligned reference."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from math import cos, pi, sin, sqrt
from pathlib import Path

from sgp4.earth_gravity import wgs72
from sgp4.functions import jday
from skyfield.api import load

from fedroute.config import ExperimentConfig
from fedroute.topology import _build_walker_satellites


def _kepler_position_km(
    semi_major_axis_km: float,
    inclination_rad: float,
    raan_rad: float,
    mean_anomaly_rad: float,
) -> tuple[float, float, float]:
    cos_raan = cos(raan_rad)
    sin_raan = sin(raan_rad)
    cos_anomaly = cos(mean_anomaly_rad)
    sin_anomaly = sin(mean_anomaly_rad)
    cos_inclination = cos(inclination_rad)
    sin_inclination = sin(inclination_rad)
    return (
        semi_major_axis_km * (cos_raan * cos_anomaly - sin_raan * sin_anomaly * cos_inclination),
        semi_major_axis_km * (sin_raan * cos_anomaly + cos_raan * sin_anomaly * cos_inclination),
        semi_major_axis_km * sin_anomaly * sin_inclination,
    )


def compare_with_kepler_reference(config_path: Path, offsets_seconds: tuple[int, ...]) -> dict:
    experiment = ExperimentConfig.from_toml(config_path)
    walker = experiment.topology.walker
    if walker is None:
        raise ValueError("The analytic reference currently requires a synthetic Walker config")
    if not offsets_seconds or any(offset < 0 for offset in offsets_seconds):
        raise ValueError("Offsets must be a non-empty sequence of non-negative seconds")

    satellites = _build_walker_satellites(walker, load.timescale())
    satellites_per_plane = walker.total_satellites // walker.planes
    semi_major_axis_km = wgs72.radiusearthkm + walker.altitude_km
    mean_motion_rad_s = sqrt(wgs72.mu / semi_major_axis_km**3)
    inclination_rad = walker.inclination_deg * pi / 180.0
    per_offset = []

    for offset_seconds in offsets_seconds:
        sample_time = walker.epoch_utc + timedelta(seconds=offset_seconds)
        julian_day, julian_fraction = jday(
            sample_time.year,
            sample_time.month,
            sample_time.day,
            sample_time.hour,
            sample_time.minute,
            sample_time.second + sample_time.microsecond / 1_000_000,
        )
        errors_km = []
        for satellite_index, satellite in enumerate(satellites):
            plane = satellite_index // satellites_per_plane
            slot = satellite_index % satellites_per_plane
            raan_rad = 2.0 * pi * plane / walker.planes
            initial_mean_anomaly_rad = (
                2.0 * pi * slot / satellites_per_plane
                + 2.0 * pi * walker.phase_factor * plane / walker.total_satellites
            )
            mean_anomaly_rad = initial_mean_anomaly_rad + mean_motion_rad_s * offset_seconds
            reference = _kepler_position_km(
                semi_major_axis_km,
                inclination_rad,
                raan_rad,
                mean_anomaly_rad,
            )
            error_code, sgp4_position, _ = satellite.model.sgp4(julian_day, julian_fraction)
            if error_code:
                raise RuntimeError(
                    f"SGP4 failed for satellite {satellite_index + 1}: code {error_code}"
                )
            errors_km.append(
                sqrt(sum((sgp4_position[axis] - reference[axis]) ** 2 for axis in range(3)))
            )
        per_offset.append(
            {
                "offset_seconds": offset_seconds,
                "satellite_count": len(satellites),
                "mean_position_error_km": sum(errors_km) / len(errors_km),
                "max_position_error_km": max(errors_km),
            }
        )

    return {
        "experiment_name": experiment.name,
        "parameter_status": experiment.parameter_status,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "epoch_utc": walker.epoch_utc.isoformat(),
        "satellite_count": walker.total_satellites,
        "orbit_altitude_km": walker.altitude_km,
        "sgp4_propagator": "SGP4 implementation from python-sgp4, WGS72 constants, initialized Walker mean elements",
        "reference_propagator": "analytic circular two-body Kepler propagation using WGS72 mu and configured Walker elements",
        "coordinate_frame": "SGP4 output TEME; reference uses the same epoch-aligned inertial axes, treating TEME as inertial over this interval",
        "model_differences": [
            "Reference assumes a perfect circular orbit and central two-body gravity.",
            "SGP4 uses mean elements, e=0.000001, WGS72, and SGP4 periodic/secular corrections.",
            "This is a model comparison, not a claim that either trajectory is ground truth.",
        ],
        "position_error_summary": per_offset,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.template.toml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1-topology/orbit_reference.json"),
    )
    parser.add_argument("--offsets-seconds", type=int, nargs="+", default=[0, 600, 1800, 3600, 6000])
    args = parser.parse_args()
    result = compare_with_kepler_reference(args.config, tuple(args.offsets_seconds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("offset_s mean_error_km max_error_km")
    for sample in result["position_error_summary"]:
        print(
            sample["offset_seconds"],
            f"{sample['mean_position_error_km']:.3f}",
            f"{sample['max_position_error_km']:.3f}",
        )
    print(f"Saved orbital comparison to {args.output}")


if __name__ == "__main__":
    main()
