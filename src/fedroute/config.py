"""Validated inputs for simulation sampling and link-topology generation."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isclose, isfinite
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class GroundStationConfig:
    name: str
    latitude_deg: float
    longitude_deg: float
    minimum_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Ground-station name cannot be empty")
        if not all(
            isfinite(value)
            for value in (self.latitude_deg, self.longitude_deg, self.minimum_elevation_deg)
        ):
            raise ValueError("Ground-station coordinates and elevation must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("Ground-station latitude must be in [-90, 90]")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise ValueError("Ground-station longitude must be in [-180, 180]")
        if not 0.0 <= self.minimum_elevation_deg <= 90.0:
            raise ValueError("Minimum elevation must be in [0, 90] degrees")


@dataclass(frozen=True)
class SimulationConfig:
    start_utc: datetime
    duration_seconds: int
    time_step_seconds: float

    def __post_init__(self) -> None:
        if self.start_utc.tzinfo is None or self.start_utc.utcoffset() != timedelta(0):
            raise ValueError("start_utc must be timezone-aware UTC")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not isfinite(self.time_step_seconds) or self.time_step_seconds <= 0:
            raise ValueError("time_step_seconds must be finite and positive")
        sample_intervals = self.duration_seconds / self.time_step_seconds
        if not isclose(sample_intervals, round(sample_intervals), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("duration_seconds must be divisible by time_step_seconds")

    def sample_times(self) -> tuple[datetime, ...]:
        """Return UTC sample instants including both simulation endpoints."""
        interval_count = round(self.duration_seconds / self.time_step_seconds)
        return tuple(
            self.start_utc + timedelta(seconds=index * self.time_step_seconds)
            for index in range(interval_count + 1)
        )


@dataclass(frozen=True)
class WalkerConfig:
    total_satellites: int
    planes: int
    phase_factor: int
    altitude_km: float
    inclination_deg: float
    epoch_utc: datetime

    def __post_init__(self) -> None:
        if self.total_satellites <= 0 or self.planes <= 0:
            raise ValueError("Walker satellite and plane counts must be positive")
        if self.total_satellites % self.planes != 0:
            raise ValueError("total_satellites must be divisible by planes")
        if not 0 <= self.phase_factor < self.planes:
            raise ValueError("phase_factor must be in [0, planes)")
        if not isfinite(self.altitude_km) or self.altitude_km <= 0.0:
            raise ValueError("altitude_km must be finite and positive")
        if not isfinite(self.inclination_deg) or not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must be in [0, 180]")
        if self.epoch_utc.tzinfo is None or self.epoch_utc.utcoffset() != timedelta(0):
            raise ValueError("epoch_utc must be timezone-aware UTC")


@dataclass(frozen=True)
class TopologyConfig:
    max_isl_range_km: float
    tle_file: Path | None = None
    walker: WalkerConfig | None = None
    earth_radius_km: float = 6378.137
    ground_stations: tuple[GroundStationConfig, ...] = ()

    def __post_init__(self) -> None:
        if (self.tle_file is None) == (self.walker is None):
            raise ValueError("Provide exactly one of tle_file or walker")
        if self.tle_file is not None and not self.tle_file.is_file():
            raise ValueError(f"TLE file does not exist: {self.tle_file}")
        if not isfinite(self.max_isl_range_km) or self.max_isl_range_km <= 0.0:
            raise ValueError("max_isl_range_km must be finite and positive")
        if not isfinite(self.earth_radius_km) or self.earth_radius_km <= 0.0:
            raise ValueError("earth_radius_km must be finite and positive")
        station_names = [station.name for station in self.ground_stations]
        if len(station_names) != len(set(station_names)):
            raise ValueError("Ground-station names must be unique")


@dataclass(frozen=True)
class TrafficFlowConfig:
    source: str
    destination: str
    arrival_rate_packets_per_second: float
    packet_bits: int
    deadline_seconds: float

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.destination.strip():
            raise ValueError("Traffic flow source and destination cannot be empty")
        if self.source == self.destination:
            raise ValueError("Traffic flow source and destination must differ")
        if not isfinite(self.arrival_rate_packets_per_second) or self.arrival_rate_packets_per_second <= 0:
            raise ValueError("Traffic arrival rate must be finite and positive")
        if self.packet_bits <= 0:
            raise ValueError("Traffic packet size must be positive")
        if not isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
            raise ValueError("Traffic deadline must be finite and positive")


@dataclass(frozen=True)
class TrafficConfig:
    arrival_process: str
    seed: int
    flows: tuple[TrafficFlowConfig, ...]

    def __post_init__(self) -> None:
        if self.arrival_process not in {"periodic", "poisson"}:
            raise ValueError("arrival_process must be 'periodic' or 'poisson'")
        if self.seed < 0:
            raise ValueError("Traffic seed must be non-negative")
        if not self.flows:
            raise ValueError("At least one traffic flow is required")


@dataclass(frozen=True)
class EnergyModelConfig:
    satellite_capacity_j: float
    tx_electronics_j_per_bit: float
    tx_amplifier_j_per_bit_km2: float
    rx_electronics_j_per_bit: float
    fl_training_j_per_transition: float
    fl_communication_j_per_byte: float

    def __post_init__(self) -> None:
        if not isfinite(self.satellite_capacity_j) or self.satellite_capacity_j <= 0:
            raise ValueError("Satellite energy capacity must be finite and positive")
        for name, value in (
            ("tx_electronics_j_per_bit", self.tx_electronics_j_per_bit),
            ("tx_amplifier_j_per_bit_km2", self.tx_amplifier_j_per_bit_km2),
            ("rx_electronics_j_per_bit", self.rx_electronics_j_per_bit),
            ("fl_training_j_per_transition", self.fl_training_j_per_transition),
            ("fl_communication_j_per_byte", self.fl_communication_j_per_byte),
        ):
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class MetricDefinitions:
    lifetime_failure_fractions: tuple[float, float, float]
    connectivity_source: str
    connectivity_destination: str
    control_overhead_components: tuple[str, ...]
    residual_energy_fairness: str
    convergence_metric: str
    convergence_window_episodes: int
    convergence_tolerance: float
    convergence_stable_windows: int

    def __post_init__(self) -> None:
        if len(self.lifetime_failure_fractions) != 3:
            raise ValueError("Specify failure fractions for T1, T50, and T90")
        if any(
            not isfinite(fraction) or not 0.0 < fraction <= 1.0
            for fraction in self.lifetime_failure_fractions
        ):
            raise ValueError("Lifetime failure fractions must be in (0, 1]")
        if tuple(sorted(self.lifetime_failure_fractions)) != self.lifetime_failure_fractions:
            raise ValueError("Lifetime failure fractions must be ordered T1, T50, T90")
        if not self.connectivity_source.strip() or not self.connectivity_destination.strip():
            raise ValueError("Connectivity source and destination cannot be empty")
        if self.connectivity_source == self.connectivity_destination:
            raise ValueError("Connectivity source and destination must differ")
        if not self.control_overhead_components or any(
            not component.strip() for component in self.control_overhead_components
        ):
            raise ValueError("At least one named control-overhead component is required")
        if self.residual_energy_fairness != "jain_residual_energy":
            raise ValueError("residual_energy_fairness must be 'jain_residual_energy'")
        if self.convergence_metric != "mean_episode_return":
            raise ValueError("convergence_metric must be 'mean_episode_return'")
        if self.convergence_window_episodes <= 0 or self.convergence_stable_windows <= 0:
            raise ValueError("Convergence window sizes must be positive")
        if not isfinite(self.convergence_tolerance) or self.convergence_tolerance < 0:
            raise ValueError("convergence_tolerance must be finite and non-negative")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    parameter_status: str
    parameter_notes: str
    run_seeds: tuple[int, ...]
    simulation: SimulationConfig
    topology: TopologyConfig
    traffic: TrafficConfig
    energy: EnergyModelConfig
    metrics: MetricDefinitions

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment name cannot be empty")
        if self.parameter_status not in {"provisional", "calibrated"}:
            raise ValueError("parameter_status must be 'provisional' or 'calibrated'")
        if not self.parameter_notes.strip():
            raise ValueError("parameter_notes must document parameter provenance")
        if not self.run_seeds or any(seed < 0 for seed in self.run_seeds):
            raise ValueError("run_seeds must contain non-negative seeds")
        if len(set(self.run_seeds)) != len(self.run_seeds):
            raise ValueError("run_seeds must be unique")
        if self.topology.walker is not None:
            satellite_nodes = {
                f"sat:{catalog_number}"
                for catalog_number in range(1, self.topology.walker.total_satellites + 1)
            }
            ground_station_nodes = {
                f"gs:{station.name}" for station in self.topology.ground_stations
            }
            for flow in self.traffic.flows:
                if flow.source not in satellite_nodes or flow.destination not in ground_station_nodes:
                    raise ValueError(
                        "Walker traffic flows must reference configured satellite sources "
                        "and ground-station destinations"
                    )
            if (
                self.metrics.connectivity_source not in satellite_nodes
                or self.metrics.connectivity_destination not in ground_station_nodes
            ):
                raise ValueError(
                    "Walker connectivity endpoints must reference a configured satellite "
                    "and ground station"
                )

    @classmethod
    def from_toml(cls, path: Path) -> "ExperimentConfig":
        """Load a complete experiment config; unresolved template fields fail early."""
        config_path = path.resolve()
        with config_path.open("rb") as config_file:
            raw = tomllib.load(config_file)

        experiment = _required_section(raw, "experiment")
        simulation_data = _required_section(raw, "simulation")
        topology_data = _required_section(raw, "topology")
        traffic_data = _required_section(raw, "traffic")
        energy_data = _required_section(raw, "energy")
        metrics_data = _required_section(raw, "metrics")
        has_walker = "walker" in topology_data
        has_tle_file = "tle_file" in topology_data
        if has_walker == has_tle_file:
            raise ValueError("Provide exactly one of topology.walker or topology.tle_file")

        simulation = SimulationConfig(
            start_utc=_parse_datetime(_required_value(simulation_data, "start_utc")),
            duration_seconds=int(_required_value(simulation_data, "duration_seconds")),
            time_step_seconds=float(_required_value(simulation_data, "time_step_seconds")),
        )
        stations = tuple(
            GroundStationConfig(
                name=str(_required_value(station, "name")),
                latitude_deg=float(_required_value(station, "latitude_deg")),
                longitude_deg=float(_required_value(station, "longitude_deg")),
                minimum_elevation_deg=float(_required_value(station, "minimum_elevation_deg")),
            )
            for station in topology_data.get("ground_stations", [])
        )
        if "walker" in topology_data:
            walker_data = topology_data["walker"]
            walker = WalkerConfig(
                total_satellites=int(_required_value(walker_data, "total_satellites")),
                planes=int(_required_value(walker_data, "planes")),
                phase_factor=int(_required_value(walker_data, "phase_factor")),
                altitude_km=float(_required_value(walker_data, "altitude_km")),
                inclination_deg=float(_required_value(walker_data, "inclination_deg")),
                epoch_utc=_parse_datetime(_required_value(walker_data, "epoch_utc")),
            )
            tle_file = None
        else:
            walker = None
            configured_tle = _required_value(topology_data, "tle_file")
            tle_file = (config_path.parent / str(configured_tle)).resolve()
        topology = TopologyConfig(
            max_isl_range_km=float(_required_value(topology_data, "max_isl_range_km")),
            tle_file=tle_file,
            walker=walker,
            earth_radius_km=float(topology_data.get("earth_radius_km", 6378.137)),
            ground_stations=stations,
        )

        flows = tuple(
            TrafficFlowConfig(
                source=str(_required_value(flow, "source")),
                destination=str(_required_value(flow, "destination")),
                arrival_rate_packets_per_second=float(
                    _required_value(flow, "arrival_rate_packets_per_second")
                ),
                packet_bits=int(_required_value(flow, "packet_bits")),
                deadline_seconds=float(_required_value(flow, "deadline_seconds")),
            )
            for flow in traffic_data.get("flows", [])
        )
        traffic = TrafficConfig(
            arrival_process=str(_required_value(traffic_data, "arrival_process")),
            seed=int(_required_value(traffic_data, "seed")),
            flows=flows,
        )
        energy = EnergyModelConfig(
            satellite_capacity_j=float(_required_value(energy_data, "satellite_capacity_j")),
            tx_electronics_j_per_bit=float(_required_value(energy_data, "tx_electronics_j_per_bit")),
            tx_amplifier_j_per_bit_km2=float(
                _required_value(energy_data, "tx_amplifier_j_per_bit_km2")
            ),
            rx_electronics_j_per_bit=float(_required_value(energy_data, "rx_electronics_j_per_bit")),
            fl_training_j_per_transition=float(
                _required_value(energy_data, "fl_training_j_per_transition")
            ),
            fl_communication_j_per_byte=float(
                _required_value(energy_data, "fl_communication_j_per_byte")
            ),
        )
        metrics = MetricDefinitions(
            lifetime_failure_fractions=tuple(
                float(value)
                for value in _required_value(metrics_data, "lifetime_failure_fractions")
            ),
            connectivity_source=str(_required_value(metrics_data, "connectivity_source")),
            connectivity_destination=str(
                _required_value(metrics_data, "connectivity_destination")
            ),
            control_overhead_components=tuple(
                str(value)
                for value in _required_value(metrics_data, "control_overhead_components")
            ),
            residual_energy_fairness=str(
                _required_value(metrics_data, "residual_energy_fairness")
            ),
            convergence_metric=str(_required_value(metrics_data, "convergence_metric")),
            convergence_window_episodes=int(
                _required_value(metrics_data, "convergence_window_episodes")
            ),
            convergence_tolerance=float(
                _required_value(metrics_data, "convergence_tolerance")
            ),
            convergence_stable_windows=int(
                _required_value(metrics_data, "convergence_stable_windows")
            ),
        )
        return cls(
            name=str(_required_value(experiment, "name")),
            parameter_status=str(_required_value(experiment, "parameter_status")),
            parameter_notes=str(_required_value(experiment, "parameter_notes")),
            run_seeds=tuple(int(seed) for seed in _required_value(experiment, "run_seeds")),
            simulation=simulation,
            topology=topology,
            traffic=traffic,
            energy=energy,
            metrics=metrics,
        )


def _required_section(data: dict, name: str) -> dict:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing required [{name}] configuration section")
    return section


def _required_value(data: dict, name: str):
    if name not in data:
        raise ValueError(f"Missing required configuration value: {name}")
    return data[name]


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO-8601 strings")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


UTC = timezone.utc