"""Compare topology traces at multiple sampling resolutions and save reproducible artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from math import dist, isclose
from pathlib import Path
import platform
import subprocess
import tracemalloc
from time import perf_counter

import networkx as nx

from fedroute.config import ExperimentConfig
from fedroute.topology import TopologyTrace, build_topology_trace, is_line_of_sight


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _trace_data(trace: TopologyTrace) -> dict:
    graphs = []
    for index, graph in enumerate(trace.graphs):
        graphs.append(
            {
                "time_utc": trace.times_utc[index].isoformat(),
                "nodes": [
                    {"id": node, **attributes}
                    for node, attributes in sorted(graph.nodes(data=True))
                ],
                "edges": [
                    {"source": node_a, "target": node_b, **attributes}
                    for node_a, node_b, attributes in sorted(
                        graph.edges(data=True),
                        key=lambda edge: tuple(sorted((edge[0], edge[1]))),
                    )
                ],
            }
        )
    return {
        "times_utc": [time.isoformat() for time in trace.times_utc],
        "graphs": graphs,
        "contact_windows": [
            {
                "node_a": window.node_a,
                "node_b": window.node_b,
                "start_utc": window.start_utc.isoformat(),
                "end_utc": window.end_utc.isoformat(),
                "duration_seconds": window.duration_seconds,
                "start_censored": window.start_censored,
                "end_censored": window.end_censored,
                "start_uncertainty_seconds": window.start_uncertainty_seconds,
                "end_uncertainty_seconds": window.end_uncertainty_seconds,
            }
            for window in trace.contact_windows
        ],
    }


def _canonical_hash(data: dict) -> str:
    encoded = json.dumps(data, default=_json_default, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _step_label(time_step_seconds: float) -> str:
    if time_step_seconds.is_integer():
        return str(int(time_step_seconds))
    return f"{time_step_seconds:g}"


def _runtime_provenance() -> dict:
    project_root = Path(__file__).resolve().parents[1]
    package_names = ("networkx", "numpy", "scikit-fuzzy", "sgp4", "skyfield", "torch")
    try:
        git_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        working_tree_dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=project_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
        working_tree_dirty = None
    source_paths = (
        Path("pyproject.toml"),
        Path("src/fedroute/config.py"),
        Path("src/fedroute/topology.py"),
        Path("scripts/validate_topology_resolution.py"),
    )
    return {
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in package_names},
        "git_revision": git_revision,
        "working_tree_dirty": working_tree_dirty,
        "source_file_sha256": {
            str(path): hashlib.sha256((project_root / path).read_bytes()).hexdigest()
            for path in source_paths
        },
    }


def _validate_isl_geometry(trace: TopologyTrace, experiment: ExperimentConfig) -> int:
    checked = 0
    earth_radius = experiment.topology.earth_radius_km
    max_range = experiment.topology.max_isl_range_km
    for graph in trace.graphs:
        for node_a, node_b, edge in graph.edges(data=True):
            if edge.get("link_type") != "isl":
                continue
            position_a = graph.nodes[node_a]["position_km"]
            position_b = graph.nodes[node_b]["position_km"]
            actual_range = dist(position_a, position_b)
            if actual_range > max_range + 1e-8:
                raise AssertionError(f"ISL {node_a}-{node_b} exceeds configured range")
            if not isclose(actual_range, edge["range_km"], rel_tol=1e-12, abs_tol=1e-8):
                raise AssertionError(f"ISL {node_a}-{node_b} stored range is inconsistent")
            if not is_line_of_sight(position_a, position_b, earth_radius):
                raise AssertionError(f"ISL {node_a}-{node_b} violates Earth line of sight")
            checked += 1
    return checked


def _snapshot_route_frames(trace: TopologyTrace, source: str, destination: str) -> int:
    return sum(
        source in graph and destination in graph and nx.has_path(graph, source, destination)
        for graph in trace.graphs
    )


def _time_expanded_route_exists(trace: TopologyTrace, source: str, destination: str) -> bool:
    if not trace.graphs or source not in trace.graphs[0]:
        return False
    expanded = nx.DiGraph()
    sink = (destination, "sink")
    for index, graph in enumerate(trace.graphs):
        for node in graph:
            expanded.add_node((node, index))
            if node == destination:
                expanded.add_edge((node, index), sink)
            if index + 1 < len(trace.graphs) and node in trace.graphs[index + 1]:
                expanded.add_edge((node, index), (node, index + 1))
        for node_a, node_b in graph.edges:
            expanded.add_edge((node_a, index), (node_b, index))
            expanded.add_edge((node_b, index), (node_a, index))
    return nx.has_path(expanded, (source, 0), sink) if sink in expanded else False


def _compare_windows(coarse: TopologyTrace, fine: TopologyTrace) -> dict:
    def grouped(trace):
        result: dict[tuple[str, str], list] = {}
        for window in trace.contact_windows:
            result.setdefault((window.node_a, window.node_b), []).append(window)
        for windows in result.values():
            windows.sort(key=lambda window: window.start_utc)
        return result

    coarse_groups = grouped(coarse)
    fine_groups = grouped(fine)
    deltas = []
    matched = 0
    unmatched_coarse = 0
    unmatched_fine = 0
    for edge in set(coarse_groups) | set(fine_groups):
        coarse_windows = coarse_groups.get(edge, [])
        fine_windows = fine_groups.get(edge, [])
        paired = min(len(coarse_windows), len(fine_windows))
        matched += paired
        unmatched_coarse += len(coarse_windows) - paired
        unmatched_fine += len(fine_windows) - paired
        for coarse_window, fine_window in zip(coarse_windows[:paired], fine_windows[:paired]):
            deltas.append(
                {
                    "edge": list(edge),
                    "occurrence": len(deltas),
                    "start_delta_seconds": (
                        fine_window.start_utc - coarse_window.start_utc
                    ).total_seconds(),
                    "end_delta_seconds": (
                        fine_window.end_utc - coarse_window.end_utc
                    ).total_seconds(),
                    "coarse_start_censored": coarse_window.start_censored,
                    "fine_start_censored": fine_window.start_censored,
                    "coarse_end_censored": coarse_window.end_censored,
                    "fine_end_censored": fine_window.end_censored,
                }
            )
    absolute_deltas = [
        abs(value)
        for comparison in deltas
        for value in (comparison["start_delta_seconds"], comparison["end_delta_seconds"])
    ]
    return {
        "coarse_window_count": len(coarse.contact_windows),
        "fine_window_count": len(fine.contact_windows),
        "matched_occurrences": matched,
        "unmatched_coarse_occurrences": unmatched_coarse,
        "unmatched_fine_occurrences": unmatched_fine,
        "max_absolute_boundary_delta_seconds": max(absolute_deltas, default=0.0),
        "mean_absolute_boundary_delta_seconds": (
            sum(absolute_deltas) / len(absolute_deltas) if absolute_deltas else 0.0
        ),
        "boundary_comparisons": deltas,
    }


def _availability_comparison(coarse: TopologyTrace, fine: TopologyTrace) -> dict:
    coarse_by_time = {
        time: {tuple(sorted(edge)) for edge in graph.edges}
        for time, graph in zip(coarse.times_utc, coarse.graphs)
    }
    fine_by_time = {
        time: {tuple(sorted(edge)) for edge in graph.edges}
        for time, graph in zip(fine.times_utc, fine.graphs)
    }
    common_times = sorted(set(coarse_by_time) & set(fine_by_time))
    mismatch_frames = sum(coarse_by_time[time] != fine_by_time[time] for time in common_times)
    return {
        "common_sample_count": len(common_times),
        "mismatch_frame_count": mismatch_frames,
        "availability_mismatch_fraction": (
            mismatch_frames / len(common_times) if common_times else None
        ),
    }


def run_comparison(
    config_path: Path,
    output_dir: Path,
    time_steps: list[int],
    duration_seconds: int | None,
    repeat_check: bool,
) -> dict:
    experiment = ExperimentConfig.from_toml(config_path)
    config_bytes = config_path.read_bytes()
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    provenance = _runtime_provenance()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    runs = []
    traces = []
    for time_step in time_steps:
        duration = duration_seconds or experiment.simulation.duration_seconds
        simulation = replace(
            experiment.simulation,
            duration_seconds=duration,
            time_step_seconds=time_step,
        )
        effective_experiment = replace(experiment, simulation=simulation)
        memory_before_build_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        start = perf_counter()
        trace = build_topology_trace(simulation, experiment.topology)
        build_runtime = perf_counter() - start
        _, peak_python_memory_bytes = tracemalloc.get_traced_memory()
        peak_incremental_python_memory_bytes = max(
            peak_python_memory_bytes - memory_before_build_bytes,
            0,
        )
        checked_isls = _validate_isl_geometry(trace, experiment)
        data = _trace_data(trace)
        trace_hash = _canonical_hash(data)
        repeat_hash = None
        if repeat_check:
            repeated = build_topology_trace(simulation, experiment.topology)
            repeat_hash = _canonical_hash(_trace_data(repeated))
            if repeat_hash != trace_hash:
                raise AssertionError(f"Topology trace was not reproducible for {time_step}s")

        trace_payload = {
            "experiment_name": experiment.name,
            "parameter_status": experiment.parameter_status,
            "source_config_path": str(config_path),
            "source_config_sha256": config_hash,
            "runtime_provenance": provenance,
            "effective_config": asdict(effective_experiment),
            "run_seed": experiment.traffic.seed,
            "sampling_step_seconds": time_step,
            "trace_sha256": trace_hash,
            "trace": data,
        }
        trace_path = output_dir / f"topology_{_step_label(time_step)}s.json"
        trace_path.write_text(
            json.dumps(trace_payload, default=_json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        frames = []
        for graph in trace.graphs:
            satellite_count = sum(data.get("kind") == "satellite" for _, data in graph.nodes(data=True))
            ground_count = sum(data.get("kind") == "ground_station" for _, data in graph.nodes(data=True))
            satellite_edges = sum(edge.get("link_type") == "isl" for _, _, edge in graph.edges(data=True))
            ground_edges = sum(edge.get("link_type") == "ground" for _, _, edge in graph.edges(data=True))
            frames.append(
                {
                    "satellites": satellite_count,
                    "ground_stations": ground_count,
                    "isl_edges": satellite_edges,
                    "ground_edges": ground_edges,
                    "total_edges": graph.number_of_edges(),
                }
            )
        route_source = experiment.metrics.connectivity_source
        route_destination = experiment.metrics.connectivity_destination
        runs.append(
            {
                "time_step_seconds": time_step,
                "duration_seconds": duration,
                "sample_count": len(trace.graphs),
                "contact_window_count": len(trace.contact_windows),
                "checked_isl_edges": checked_isls,
                "snapshot_route_frames": _snapshot_route_frames(
                    trace,
                    route_source,
                    route_destination,
                ),
                "snapshot_route_frame_count": len(trace.graphs),
                "time_expanded_route_exists": _time_expanded_route_exists(
                    trace,
                    route_source,
                    route_destination,
                ),
                "topology_build_runtime_seconds": build_runtime,
                "peak_traced_python_memory_bytes": peak_python_memory_bytes,
                "peak_incremental_traced_python_memory_bytes": (
                    peak_incremental_python_memory_bytes
                ),
                "trace_sha256": trace_hash,
                "repeat_hash_matches": repeat_hash == trace_hash if repeat_check else None,
                "trace_file": str(trace_path),
                "frame_availability": frames,
            }
        )
        traces.append(trace)

    report = {
        "experiment_name": experiment.name,
        "parameter_status": experiment.parameter_status,
        "source_config_path": str(config_path),
        "source_config_sha256": config_hash,
        "runtime_provenance": provenance,
        "duration_seconds": duration_seconds or experiment.simulation.duration_seconds,
        "run_seed": experiment.traffic.seed,
        "repeat_check_enabled": repeat_check,
        "runs": runs,
    }
    ordered = sorted(zip(time_steps, traces), key=lambda pair: pair[0], reverse=True)
    if len(ordered) >= 2:
        adjacent_comparisons = []
        for (coarse_step, coarse_trace), (fine_step, fine_trace) in zip(
            ordered,
            ordered[1:],
        ):
            adjacent_comparisons.append(
                {
                    "coarse_step_seconds": coarse_step,
                    "fine_step_seconds": fine_step,
                    "window_comparison": _compare_windows(coarse_trace, fine_trace),
                    "common_sample_availability": _availability_comparison(
                        coarse_trace,
                        fine_trace,
                    ),
                }
            )
        report["resolution_comparisons"] = adjacent_comparisons
        coarse_step, coarse_trace = ordered[0]
        fine_step, fine_trace = ordered[-1]
        report["coarsest_vs_finest"] = {
            "coarse_step_seconds": coarse_step,
            "fine_step_seconds": fine_step,
            "window_comparison": _compare_windows(coarse_trace, fine_trace),
            "common_sample_availability": _availability_comparison(
                coarse_trace,
                fine_trace,
            ),
        }

    report_path = output_dir / "resolution_report.json"
    report_path.write_text(
        json.dumps(report, default=_json_default, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.template.toml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase1-topology"))
    parser.add_argument("--time-steps", type=float, nargs="+", default=[10.0, 5.0])
    parser.add_argument("--duration-seconds", type=int)
    parser.add_argument("--no-repeat-check", action="store_true")
    args = parser.parse_args()
    if any(step <= 0 for step in args.time_steps):
        parser.error("time steps must be positive")

    report = run_comparison(
        config_path=args.config,
        output_dir=args.output_dir,
        time_steps=args.time_steps,
        duration_seconds=args.duration_seconds,
        repeat_check=not args.no_repeat_check,
    )
    for run in report["runs"]:
        print(
            f"dt={run['time_step_seconds']:g}s samples={run['sample_count']} "
            f"contacts={run['contact_window_count']} "
            f"snapshot_routes={run['snapshot_route_frames']}/{run['snapshot_route_frame_count']} "
            f"contact_plan_route={run['time_expanded_route_exists']} "
            f"build_s={run['topology_build_runtime_seconds']:.3f} "
            f"build_peak_python_MiB="
            f"{run['peak_incremental_traced_python_memory_bytes'] / 1024**2:.2f} "
            f"process_peak_python_MiB="
            f"{run['peak_traced_python_memory_bytes'] / 1024**2:.2f} "
            f"reproducible={run['repeat_hash_matches']}"
        )
    if "resolution_comparisons" in report:
        for comparison in report["resolution_comparisons"]:
            windows = comparison["window_comparison"]
            availability = comparison["common_sample_availability"]
            print(
                f"{comparison['coarse_step_seconds']:g}s vs "
                f"{comparison['fine_step_seconds']:g}s: "
                f"windows={windows['coarse_window_count']}/{windows['fine_window_count']}, "
                f"max_boundary_delta_s={windows['max_absolute_boundary_delta_seconds']:.3f}, "
                f"common_frame_mismatches={availability['mismatch_frame_count']}/"
                f"{availability['common_sample_count']}"
            )
    print(f"Saved validation report to {args.output_dir / 'resolution_report.json'}")


if __name__ == "__main__":
    main()
