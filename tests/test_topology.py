from datetime import datetime, timedelta, timezone
from math import pi, sqrt
from pathlib import Path

import networkx as nx
import pytest
from sgp4.earth_gravity import wgs72

from fedroute.config import (
    ExperimentConfig,
    GroundStationConfig,
    SimulationConfig,
    TopologyConfig,
    WalkerConfig,
)
from fedroute import topology
from fedroute.topology import (
    _build_walker_satellites,
    _contact_windows,
    build_topology_trace,
    is_line_of_sight,
)
from skyfield.api import load, wgs84
from scripts.validate_contact_crossings import _bisect_transition


def test_line_of_sight_rejects_segment_through_earth() -> None:
    assert not is_line_of_sight((7000.0, 0.0, 0.0), (-7000.0, 0.0, 0.0), 6378.0)


def test_line_of_sight_accepts_segment_clear_of_earth() -> None:
    assert is_line_of_sight((7000.0, 0.0, 0.0), (7000.0, 1000.0, 0.0), 6378.0)


def test_contact_windows_include_one_sampling_interval_at_end() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = tuple(start + timedelta(seconds=10 * index) for index in range(4))
    graphs = []
    for is_active in (True, True, False, True):
        graph = nx.Graph()
        if is_active:
            graph.add_edge("sat:a", "sat:b")
        graphs.append(graph)

    windows = _contact_windows(times, tuple(graphs), 10)

    assert [(window.start_utc, window.end_utc) for window in windows] == [
        (times[0], times[2]),
        (times[3], times[3] + timedelta(seconds=10)),
    ]
    assert windows[0].start_censored
    assert not windows[0].end_censored
    assert windows[0].start_uncertainty_seconds is None
    assert windows[0].end_uncertainty_seconds == 10.0
    assert not windows[1].start_censored
    assert windows[1].end_censored
    assert windows[1].start_uncertainty_seconds == 10.0
    assert windows[1].end_uncertainty_seconds is None


def test_contact_windows_bound_analytic_crossing_error_and_miss_short_contacts() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def sampled_window(time_step: float, contact_start: float, contact_end: float, duration: int):
        sample_count = int(duration / time_step)
        times = tuple(
            start + timedelta(seconds=index * time_step)
            for index in range(sample_count + 1)
        )
        graphs = []
        for time in times:
            offset = (time - start).total_seconds()
            graph = nx.Graph()
            if contact_start <= offset < contact_end:
                graph.add_edge("sat:a", "sat:b")
            graphs.append(graph)
        return _contact_windows(times, tuple(graphs), time_step)

    coarse = sampled_window(10.0, 2.3, 6.7, 20)
    medium = sampled_window(5.0, 2.3, 6.7, 20)
    fine = sampled_window(2.5, 2.3, 6.7, 20)
    assert coarse == ()
    assert len(medium) == len(fine) == 1
    assert abs((medium[0].start_utc - start).total_seconds() - 2.3) <= 5.0
    assert abs((medium[0].end_utc - start).total_seconds() - 6.7) <= 5.0
    assert abs((fine[0].start_utc - start).total_seconds() - 2.3) <= 2.5
    assert abs((fine[0].end_utc - start).total_seconds() - 6.7) <= 2.5

    near_boundary = sampled_window(10.0, 10.2, 29.8, 40)
    assert len(near_boundary) == 1
    assert abs((near_boundary[0].start_utc - start).total_seconds() - 10.2) <= 10.0
    assert abs((near_boundary[0].end_utc - start).total_seconds() - 29.8) <= 10.0


def test_contact_crossing_bisection_matches_known_threshold() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    crossing_seconds = 7.3
    crossing = _bisect_transition(
        lambda sample: (sample - start).total_seconds() >= crossing_seconds,
        start,
        start + timedelta(seconds=10),
        left_active=False,
        tolerance_seconds=0.001,
    )

    assert abs((crossing - start).total_seconds() - crossing_seconds) <= 0.001


def test_simulation_requires_utc_and_aligned_duration() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        SimulationConfig(datetime(2026, 1, 1), 60, 10)
    with pytest.raises(ValueError, match="divisible"):
        SimulationConfig(datetime(2026, 1, 1, tzinfo=timezone.utc), 61, 10)


def test_simulation_supports_fractional_time_steps() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    simulation = SimulationConfig(start, duration_seconds=10, time_step_seconds=2.5)

    assert len(simulation.sample_times()) == 5
    assert [
        (right - left).total_seconds()
        for left, right in zip(simulation.sample_times(), simulation.sample_times()[1:])
    ] == [2.5, 2.5, 2.5, 2.5]
    with pytest.raises(ValueError, match="divisible"):
        SimulationConfig(start, duration_seconds=10, time_step_seconds=3.0)


def test_experiment_config_loads_explicit_system_model(tmp_path) -> None:
    config_file = tmp_path / "experiment.toml"
    config_file.write_text(
        """
[experiment]
name = "test"
parameter_status = "provisional"
parameter_notes = "Synthetic test parameters"
run_seeds = [7]
[simulation]
start_utc = "2026-01-01T00:00:00Z"
duration_seconds = 10
time_step_seconds = 10
[topology]
max_isl_range_km = 10000.0
[topology.walker]
total_satellites = 6
planes = 3
phase_factor = 1
altitude_km = 550.0
inclination_deg = 53.0
epoch_utc = "2026-01-01T00:00:00Z"
[[topology.ground_stations]]
name = "gateway"
latitude_deg = 0.0
longitude_deg = 0.0
minimum_elevation_deg = 10.0
[traffic]
arrival_process = "poisson"
seed = 7
[[traffic.flows]]
source = "sat:1"
destination = "gs:gateway"
arrival_rate_packets_per_second = 0.1
packet_bits = 4096
deadline_seconds = 2.0
[energy]
satellite_capacity_j = 100.0
tx_electronics_j_per_bit = 1e-9
tx_amplifier_j_per_bit_km2 = 1e-12
rx_electronics_j_per_bit = 1e-9
fl_training_j_per_transition = 0.01
fl_communication_j_per_byte = 1e-6
[metrics]
lifetime_failure_fractions = [0.01, 0.5, 0.9]
connectivity_source = "sat:1"
connectivity_destination = "gs:gateway"
control_overhead_components = ["routing", "federated"]
residual_energy_fairness = "jain_residual_energy"
convergence_metric = "mean_episode_return"
convergence_window_episodes = 10
convergence_tolerance = 0.01
convergence_stable_windows = 3
"""
    )

    experiment = ExperimentConfig.from_toml(config_file)

    assert experiment.topology.walker.total_satellites == 6
    assert experiment.traffic.flows[0].deadline_seconds == 2.0
    assert experiment.energy.satellite_capacity_j == 100.0
    assert experiment.metrics.lifetime_failure_fractions == (0.01, 0.5, 0.9)
    assert experiment.run_seeds == (7,)


def test_experiment_config_rejects_incomplete_template(tmp_path) -> None:
    config_file = tmp_path / "incomplete.toml"
    config_file.write_text('[experiment]\nname="incomplete"\n')

    with pytest.raises(ValueError, match=r"Missing required \[simulation\]"):
        ExperimentConfig.from_toml(config_file)


def test_experiment_template_is_complete_and_marked_provisional() -> None:
    experiment = ExperimentConfig.from_toml(Path("configs/experiment.template.toml"))

    assert experiment.parameter_status == "provisional"
    assert experiment.traffic.seed == 123
    assert experiment.energy.satellite_capacity_j == 100.0
    assert experiment.metrics.convergence_window_episodes == 10


def test_walker_period_at_780_km_matches_kepler_reference() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    walker = WalkerConfig(
        total_satellites=1,
        planes=1,
        phase_factor=0,
        altitude_km=780.0,
        inclination_deg=53.0,
        epoch_utc=epoch,
    )
    satellite = _build_walker_satellites(walker, load.timescale())[0]

    kepler_period_minutes = (
        2.0 * pi * sqrt((wgs72.radiusearthkm + walker.altitude_km) ** 3 / wgs72.mu) / 60.0
    )
    sgp4_period_minutes = 2.0 * pi / satellite.model.no_kozai

    assert kepler_period_minutes == pytest.approx(100.0, abs=1.0)
    assert sgp4_period_minutes == pytest.approx(kepler_period_minutes, abs=0.01)


def test_24_satellite_pilot_orbital_period_matches_kepler_reference() -> None:
    experiment = ExperimentConfig.from_toml(Path("configs/experiment.template.toml"))
    walker = experiment.topology.walker
    assert walker is not None
    satellites = _build_walker_satellites(walker, load.timescale())
    expected_period_minutes = (
        2.0
        * pi
        * sqrt((wgs72.radiusearthkm + walker.altitude_km) ** 3 / wgs72.mu)
        / 60.0
    )
    generated_periods = [2.0 * pi / satellite.model.no_kozai for satellite in satellites]

    assert len(satellites) == 24
    assert expected_period_minutes == pytest.approx(95.7, abs=0.2)
    assert all(
        period == pytest.approx(expected_period_minutes, abs=0.01)
        for period in generated_periods
    )


def test_ground_contact_metadata_matches_skyfield_observer_check() -> None:
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    walker = WalkerConfig(
        total_satellites=1,
        planes=1,
        phase_factor=0,
        altitude_km=550.0,
        inclination_deg=53.0,
        epoch_utc=epoch,
    )
    time_scale = load.timescale()
    satellite = _build_walker_satellites(walker, time_scale)[0]
    skyfield_time = time_scale.from_datetime(epoch)
    subpoint = wgs84.subpoint(satellite.at(skyfield_time))
    station = GroundStationConfig(
        "underpass",
        latitude_deg=subpoint.latitude.degrees + 0.2,
        longitude_deg=subpoint.longitude.degrees,
        minimum_elevation_deg=0.0,
    )
    observer = wgs84.latlon(station.latitude_deg, station.longitude_deg)
    expected_altitude, _, expected_range = (satellite - observer).at(skyfield_time).altaz()

    trace = build_topology_trace(
        SimulationConfig(epoch, 10, 10),
        TopologyConfig(
            max_isl_range_km=10_000.0,
            walker=walker,
            ground_stations=(station,),
        ),
    )
    edge = trace.graphs[0]["sat:1"]["gs:underpass"]

    assert edge["elevation_deg"] == pytest.approx(expected_altitude.degrees, abs=1e-8)
    assert edge["range_km"] == pytest.approx(expected_range.km, abs=1e-8)

    at_mask = GroundStationConfig(
        station.name,
        station.latitude_deg,
        station.longitude_deg,
        minimum_elevation_deg=expected_altitude.degrees,
    )
    above_mask = GroundStationConfig(
        station.name,
        station.latitude_deg,
        station.longitude_deg,
        minimum_elevation_deg=expected_altitude.degrees + 1e-6,
    )
    at_mask_trace = build_topology_trace(
        SimulationConfig(epoch, 10, 10),
        TopologyConfig(10_000.0, walker=walker, ground_stations=(at_mask,)),
    )
    above_mask_trace = build_topology_trace(
        SimulationConfig(epoch, 10, 10),
        TopologyConfig(10_000.0, walker=walker, ground_stations=(above_mask,)),
    )
    assert at_mask_trace.graphs[0].has_edge("sat:1", "gs:underpass")
    assert not above_mask_trace.graphs[0].has_edge("sat:1", "gs:underpass")


def test_walker_configuration_propagates_with_sgp4() -> None:
    epoch = datetime(2026, 10, 16, tzinfo=timezone.utc)
    walker = WalkerConfig(
        total_satellites=6,
        planes=3,
        phase_factor=1,
        altitude_km=550.0,
        inclination_deg=53.0,
        epoch_utc=epoch,
    )

    satellites = _build_walker_satellites(walker, load.timescale())

    assert len(satellites) == 6
    assert {satellite.model.satnum for satellite in satellites} == set(range(1, 7))
    assert all(
        len(satellite.at(load.timescale().from_datetime(epoch)).position.km) == 3
        for satellite in satellites
    )


def test_topology_trace_builds_graphs_and_contact_metadata(monkeypatch, tmp_path) -> None:
    class Position:
        def __init__(self, km):
            self.km = km

    class Sample:
        def __init__(self, km):
            self.position = Position(km)

    class Satellite:
        def __init__(self, name, catalog_number, position):
            self.name = name
            self.model = type("Model", (), {"satnum": catalog_number})()
            self.position = position

        def at(self, _time):
            return Sample(self.position)

    class TimeScale:
        def from_datetime(self, value):
            return value

    satellites = (
        Satellite("duplicate", 1, (7000.0, 0.0, 0.0)),
        Satellite("duplicate", 2, (7000.0, 100.0, 0.0)),
    )
    monkeypatch.setattr(topology.load, "timescale", lambda: TimeScale())
    monkeypatch.setattr(
        topology.load,
        "tle_file",
        lambda _path, ts: satellites,
    )
    tle_file = tmp_path / "catalog.tle"
    tle_file.write_text("test fixture")
    simulation = SimulationConfig(datetime(2026, 1, 1, tzinfo=timezone.utc), 10, 10)
    settings = topology.TopologyConfig(
        max_isl_range_km=100.0,
        tle_file=tle_file,
    )

    trace = build_topology_trace(simulation, settings)

    assert len(trace.graphs) == 2
    assert set(trace.graphs[0].nodes) == {"sat:1", "sat:2"}
    assert trace.graphs[0]["sat:1"]["sat:2"]["range_km"] == pytest.approx(100.0)
    assert trace.graphs[0]["sat:1"]["sat:2"]["contact_duration_s"] == 20.0
    assert trace.graphs[0]["sat:1"]["sat:2"]["remaining_contact_s"] == 20.0
    assert trace.graphs[1]["sat:1"]["sat:2"]["remaining_contact_s"] == 10.0

    below_cutoff = topology.TopologyConfig(
        max_isl_range_km=99.999,
        tle_file=tle_file,
    )
    excluded_trace = build_topology_trace(simulation, below_cutoff)
    assert not excluded_trace.graphs[0].has_edge("sat:1", "sat:2")