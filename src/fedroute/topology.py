"""Generate sampled satellite and ground-station contact graphs from TLEs."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import combinations
from math import pi, sqrt

import networkx as nx
from skyfield.api import EarthSatellite, load, wgs84
from sgp4.api import Satrec, WGS72
from sgp4.earth_gravity import wgs72 as earth_gravity
from sgp4.functions import jday

from fedroute.config import SimulationConfig, TopologyConfig, WalkerConfig


@dataclass(frozen=True)
class ContactWindow:
    node_a: str
    node_b: str
    start_utc: datetime
    end_utc: datetime

    @property
    def duration_seconds(self) -> float:
        return (self.end_utc - self.start_utc).total_seconds()


@dataclass(frozen=True)
class TopologyTrace:
    times_utc: tuple[datetime, ...]
    graphs: tuple[nx.Graph, ...]
    contact_windows: tuple[ContactWindow, ...]


def is_line_of_sight(
    position_a_km: tuple[float, float, float],
    position_b_km: tuple[float, float, float],
    earth_radius_km: float,
) -> bool:
    """Check whether the segment between two geocentric positions clears Earth."""
    segment = tuple(b - a for a, b in zip(position_a_km, position_b_km))
    segment_length_squared = sum(component * component for component in segment)
    if segment_length_squared == 0.0:
        return sum(component * component for component in position_a_km) >= earth_radius_km**2

    projection = -sum(a * direction for a, direction in zip(position_a_km, segment))
    fraction = min(1.0, max(0.0, projection / segment_length_squared))
    closest = tuple(a + fraction * direction for a, direction in zip(position_a_km, segment))
    closest_distance_squared = sum(component * component for component in closest)
    return closest_distance_squared >= earth_radius_km**2


def _contact_windows(
    times_utc: tuple[datetime, ...],
    graphs: tuple[nx.Graph, ...],
    time_step_seconds: int,
) -> tuple[ContactWindow, ...]:
    active_edges = [
        {frozenset(edge) for edge in graph.edges}
        for graph in graphs
    ]
    all_edges = set().union(*active_edges)
    windows: list[ContactWindow] = []

    for edge in all_edges:
        node_a, node_b = sorted(edge)
        start_index: int | None = None
        for index, edges in enumerate(active_edges + [set()]):
            if edge in edges:
                if start_index is None:
                    start_index = index
            elif start_index is not None:
                windows.append(
                    ContactWindow(
                        node_a=node_a,
                        node_b=node_b,
                        start_utc=times_utc[start_index],
                        end_utc=times_utc[index - 1]
                        + timedelta(seconds=time_step_seconds),
                    )
                )
                start_index = None

    return tuple(sorted(windows, key=lambda item: (item.start_utc, item.node_a, item.node_b)))


def _satellite_node(satellite: EarthSatellite) -> str:
    return f"sat:{satellite.model.satnum}"


def _build_walker_satellites(
    walker: WalkerConfig,
    time_scale,
) -> tuple[EarthSatellite, ...]:
    """Build circular Walker-Delta elements and initialize each with SGP4."""
    epoch_jd, epoch_fraction = jday(
        walker.epoch_utc.year,
        walker.epoch_utc.month,
        walker.epoch_utc.day,
        walker.epoch_utc.hour,
        walker.epoch_utc.minute,
        walker.epoch_utc.second + walker.epoch_utc.microsecond / 1_000_000,
    )
    epoch_days = epoch_jd + epoch_fraction - 2433281.5
    satellites_per_plane = walker.total_satellites // walker.planes
    semimajor_axis_km = earth_gravity.radiusearthkm + walker.altitude_km
    mean_motion_rad_per_minute = sqrt(
        earth_gravity.mu / semimajor_axis_km**3
    ) * 60.0
    inclination = walker.inclination_deg * pi / 180.0

    satellites = []
    for plane in range(walker.planes):
        raan = 2.0 * pi * plane / walker.planes
        for slot in range(satellites_per_plane):
            satellite_number = plane * satellites_per_plane + slot + 1
            mean_anomaly = (
                2.0 * pi * slot / satellites_per_plane
                + 2.0 * pi * walker.phase_factor * plane / walker.total_satellites
            ) % (2.0 * pi)
            record = Satrec()
            record.sgp4init(
                WGS72,
                "i",
                satellite_number,
                epoch_days,
                0.0,
                0.0,
                0.0,
                0.000001,
                0.0,
                inclination,
                mean_anomaly,
                mean_motion_rad_per_minute,
                raan,
            )
            satellite = EarthSatellite.from_satrec(record, time_scale)
            satellite.name = f"walker-{satellite_number:03d}"
            satellites.append(satellite)
    return tuple(satellites)


def build_topology_trace(
    simulation: SimulationConfig,
    topology: TopologyConfig,
) -> TopologyTrace:
    """Propagate TLE satellites and build one availability graph per sample."""
    times_utc = simulation.sample_times()
    time_scale = load.timescale()
    if topology.walker is not None:
        satellites = _build_walker_satellites(topology.walker, time_scale)
    else:
        satellites = tuple(load.tle_file(str(topology.tle_file), ts=time_scale))
    if not satellites:
        raise ValueError(f"No satellites found in TLE file: {topology.tle_file}")

    observers = {
        f"gs:{station.name}": (
            station,
            wgs84.latlon(station.latitude_deg, station.longitude_deg),
        )
        for station in topology.ground_stations
    }
    graphs: list[nx.Graph] = []

    for sample_time in times_utc:
        skyfield_time = time_scale.from_datetime(sample_time)
        positions = {
            _satellite_node(satellite): tuple(satellite.at(skyfield_time).position.km)
            for satellite in satellites
        }
        graph = nx.Graph(time_utc=sample_time)
        satellite_names = {
            _satellite_node(satellite): satellite.name for satellite in satellites
        }
        for node, position in positions.items():
            graph.add_node(
                node,
                kind="satellite",
                name=satellite_names[node],
                position_km=position,
            )
        for node, (station, _) in observers.items():
            graph.add_node(
                node,
                kind="ground_station",
                latitude_deg=station.latitude_deg,
                longitude_deg=station.longitude_deg,
            )

        for node_a, node_b in combinations(positions, 2):
            position_a = positions[node_a]
            position_b = positions[node_b]
            range_km = sum(
                (a - b) ** 2 for a, b in zip(position_a, position_b)
            ) ** 0.5
            if (
                range_km <= topology.max_isl_range_km
                and is_line_of_sight(position_a, position_b, topology.earth_radius_km)
            ):
                graph.add_edge(node_a, node_b, range_km=range_km, link_type="isl")

        for satellite in satellites:
            satellite_node = _satellite_node(satellite)
            for station_node, (station, observer) in observers.items():
                difference = satellite - observer
                altitude, _, range_to_station = difference.at(skyfield_time).altaz()
                elevation_deg = altitude.degrees
                if elevation_deg >= station.minimum_elevation_deg:
                    graph.add_edge(
                        satellite_node,
                        station_node,
                        range_km=range_to_station.km,
                        elevation_deg=elevation_deg,
                        link_type="ground",
                    )

        graphs.append(graph)

    graph_tuple = tuple(graphs)
    windows = _contact_windows(
        times_utc,
        graph_tuple,
        simulation.time_step_seconds,
    )
    windows_by_edge: dict[frozenset[str], list[ContactWindow]] = {}
    for window in windows:
        windows_by_edge.setdefault(
            frozenset((window.node_a, window.node_b)), []
        ).append(window)

    for index, graph in enumerate(graph_tuple):
        for node_a, node_b in graph.edges:
            window = next(
                window
                for window in windows_by_edge[frozenset((node_a, node_b))]
                if window.start_utc <= times_utc[index] < window.end_utc
            )
            graph[node_a][node_b]["contact_duration_s"] = window.duration_seconds
            graph[node_a][node_b]["remaining_contact_s"] = (
                window.end_utc - times_utc[index]
            ).total_seconds()

    return TopologyTrace(times_utc, graph_tuple, windows)