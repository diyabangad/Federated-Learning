"""Validated inputs for simulation sampling and link-topology generation."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class GroundStationConfig:
    name: str
    latitude_deg: float
    longitude_deg: float
    minimum_elevation_deg: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Ground-station name cannot be empty")
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
    time_step_seconds: int

    def __post_init__(self) -> None:
        if self.start_utc.tzinfo is None or self.start_utc.utcoffset() != timedelta(0):
            raise ValueError("start_utc must be timezone-aware UTC")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.time_step_seconds <= 0:
            raise ValueError("time_step_seconds must be positive")
        if self.duration_seconds % self.time_step_seconds != 0:
            raise ValueError("duration_seconds must be divisible by time_step_seconds")

    def sample_times(self) -> tuple[datetime, ...]:
        """Return UTC sample instants including both simulation endpoints."""
        return tuple(
            self.start_utc + timedelta(seconds=offset)
            for offset in range(0, self.duration_seconds + 1, self.time_step_seconds)
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
        if not 0.0 <= self.inclination_deg <= 180.0:
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
        if self.max_isl_range_km <= 0.0:
            raise ValueError("max_isl_range_km must be positive")
        if self.earth_radius_km <= 0.0:
            raise ValueError("earth_radius_km must be positive")
        station_names = [station.name for station in self.ground_stations]
        if len(station_names) != len(set(station_names)):
            raise ValueError("Ground-station names must be unique")


UTC = timezone.utc