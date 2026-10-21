"""Compare sampled contact boundaries with continuous-time geometric crossings."""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
from math import sqrt
from pathlib import Path

from skyfield.api import load, wgs84

from fedroute.config import ExperimentConfig, SimulationConfig
from fedroute.topology import _build_walker_satellites, build_topology_trace


def _clears_spherical_earth(position_a, position_b, earth_radius_km: float) -> bool:
    direction = tuple(b - a for a, b in zip(position_a, position_b))
    length_squared = sum(component * component for component in direction)
    if length_squared == 0.0:
        return sum(component * component for component in position_a) >= earth_radius_km**2
    projection = -sum(a * component for a, component in zip(position_a, direction))
    fraction = min(1.0, max(0.0, projection / length_squared))
    closest = tuple(a + fraction * component for a, component in zip(position_a, direction))
    return sum(component * component for component in closest) >= earth_radius_km**2


def _bisect_transition(predicate, left, right, left_active: bool, tolerance_seconds: float = 0.01):
    left_value = predicate(left)
    right_value = predicate(right)
    if left_value != left_active or right_value == left_active:
        raise ValueError("Crossing bracket does not straddle the requested contact transition")
    while (right - left).total_seconds() > tolerance_seconds:
        midpoint = left + (right - left) / 2
        if predicate(midpoint) == left_active:
            left = midpoint
        else:
            right = midpoint
    return left + (right - left) / 2


def validate_crossings(config_path: Path, time_step_seconds: float, duration_seconds: int) -> dict:
    experiment = ExperimentConfig.from_toml(config_path)
    if experiment.topology.walker is None:
        raise ValueError("This crossing validator currently requires synthetic Walker elements")
    simulation = SimulationConfig(
        experiment.simulation.start_utc,
        duration_seconds,
        time_step_seconds,
    )
    trace = build_topology_trace(simulation, experiment.topology)
    time_scale = load.timescale()
    walker = experiment.topology.walker
    satellites = {
        f"sat:{satellite.model.satnum}": satellite
        for satellite in _build_walker_satellites(walker, time_scale)
    }
    observers = {
        f"gs:{station.name}": (station, wgs84.latlon(station.latitude_deg, station.longitude_deg))
        for station in experiment.topology.ground_stations
    }

    def contact_is_active(node_a: str, node_b: str, time) -> bool:
        if node_a.startswith("sat:") and node_b.startswith("sat:"):
            position_a = tuple(satellites[node_a].at(time_scale.from_datetime(time)).position.km)
            position_b = tuple(satellites[node_b].at(time_scale.from_datetime(time)).position.km)
            distance_km = sqrt(sum((a - b) ** 2 for a, b in zip(position_a, position_b)))
            return (
                distance_km <= experiment.topology.max_isl_range_km
                and _clears_spherical_earth(
                    position_a,
                    position_b,
                    experiment.topology.earth_radius_km,
                )
            )

        satellite_node, station_node = (
            (node_a, node_b) if node_a.startswith("sat:") else (node_b, node_a)
        )
        station, observer = observers[station_node]
        altitude, _, _ = (satellites[satellite_node] - observer).at(
            time_scale.from_datetime(time)
        ).altaz()
        return altitude.degrees >= station.minimum_elevation_deg

    boundary_errors = []
    skipped_censored = 0
    for window in trace.contact_windows:
        sample_interval = timedelta(seconds=time_step_seconds)
        transitions = []
        if window.start_censored:
            skipped_censored += 1
        else:
            start_crossing = _bisect_transition(
                lambda time: contact_is_active(window.node_a, window.node_b, time),
                window.start_utc - sample_interval,
                window.start_utc,
                left_active=False,
            )
            transitions.append(("start", window.start_utc, start_crossing))
        if window.end_censored:
            skipped_censored += 1
        else:
            end_crossing = _bisect_transition(
                lambda time: contact_is_active(window.node_a, window.node_b, time),
                window.end_utc - sample_interval,
                window.end_utc,
                left_active=True,
            )
            transitions.append(("end", window.end_utc, end_crossing))
        for boundary, sampled_time, crossing_time in transitions:
            boundary_errors.append(
                {
                    "edge": [window.node_a, window.node_b],
                    "boundary": boundary,
                    "sampled_time_utc": sampled_time.isoformat(),
                    "reference_crossing_utc": crossing_time.isoformat(),
                    "sampled_minus_reference_seconds": (
                        sampled_time - crossing_time
                    ).total_seconds(),
                }
            )

    absolute_errors = [abs(row["sampled_minus_reference_seconds"]) for row in boundary_errors]
    return {
        "experiment_name": experiment.name,
        "parameter_status": experiment.parameter_status,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "validator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "duration_seconds": duration_seconds,
        "time_step_seconds": time_step_seconds,
        "reference_method": "Bisection on direct SGP4 position-derived range/spherical LOS or Skyfield ground elevation, independently of sampled edge window grouping",
        "reference_frame": "SGP4 TEME state vectors for ISLs; Skyfield ground-observer elevation derived from the same propagator for ground contacts",
        "bisection_tolerance_seconds": 0.01,
        "matched_boundary_count": len(boundary_errors),
        "skipped_censored_boundary_count": skipped_censored,
        "max_absolute_sampled_boundary_error_seconds": max(absolute_errors, default=0.0),
        "mean_absolute_sampled_boundary_error_seconds": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
        ),
        "sampling_bound_violations": sum(
            error > time_step_seconds + 0.01 for error in absolute_errors
        ),
        "boundaries": boundary_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.template.toml"))
    parser.add_argument("--time-step-seconds", type=float, default=10.0)
    parser.add_argument("--duration-seconds", type=int, default=6000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase1-topology/contact_crossings.json"),
    )
    args = parser.parse_args()
    result = validate_crossings(args.config, args.time_step_seconds, args.duration_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"boundaries={result['matched_boundary_count']} "
        f"censored_skipped={result['skipped_censored_boundary_count']} "
        f"max_error_s={result['max_absolute_sampled_boundary_error_seconds']:.3f} "
        f"mean_error_s={result['mean_absolute_sampled_boundary_error_seconds']:.3f} "
        f"bound_violations={result['sampling_bound_violations']}"
    )
    print(f"Saved independent crossing comparison to {args.output}")


if __name__ == "__main__":
    main()
