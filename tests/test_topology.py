from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest

from fedroute.config import SimulationConfig, WalkerConfig
from fedroute import topology
from fedroute.topology import (
    _build_walker_satellites,
    _contact_windows,
    build_topology_trace,
    is_line_of_sight,
)
from skyfield.api import load


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


def test_simulation_requires_utc_and_aligned_duration() -> None:
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        SimulationConfig(datetime(2026, 1, 1), 60, 10)
    with pytest.raises(ValueError, match="divisible"):
        SimulationConfig(datetime(2026, 1, 1, tzinfo=timezone.utc), 61, 10)


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
        max_isl_range_km=200.0,
        tle_file=tle_file,
    )

    trace = build_topology_trace(simulation, settings)

    assert len(trace.graphs) == 2
    assert set(trace.graphs[0].nodes) == {"sat:1", "sat:2"}
    assert trace.graphs[0]["sat:1"]["sat:2"]["contact_duration_s"] == 20.0
    assert trace.graphs[0]["sat:1"]["sat:2"]["remaining_contact_s"] == 20.0
    assert trace.graphs[1]["sat:1"]["sat:2"]["remaining_contact_s"] == 10.0