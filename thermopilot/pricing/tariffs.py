"""
Electricity Tariff Models (Time-of-Use & Critical Peak Pricing).

Provides realistic rate structures representing real-world utilities (e.g. PG&E EV2-A,
Ontario TOU, and dynamic peak dispatch pricing).
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class ElectricityTariff:
    name: str
    off_peak_rate: float      # $/kWh (Night / early morning)
    mid_peak_rate: float      # $/kWh (Daytime regular)
    on_peak_rate: float       # $/kWh (Evening peak, typically 16:00 - 21:00)
    on_peak_start_hour: int = 16
    on_peak_end_hour: int = 21
    mid_peak_start_hour: int = 7
    mid_peak_end_hour: int = 23


TARIFF_DEFAULT_TOU = ElectricityTariff(
    name="Standard Commercial/Residential TOU",
    off_peak_rate=0.12,
    mid_peak_rate=0.24,
    on_peak_rate=0.52,
    on_peak_start_hour=16,
    on_peak_end_hour=21,
    mid_peak_start_hour=7,
    mid_peak_end_hour=23
)

TARIFF_AGGRESSIVE_PEAK = ElectricityTariff(
    name="Critical Peak Dynamic",
    off_peak_rate=0.10,
    mid_peak_rate=0.22,
    on_peak_rate=0.85,
    on_peak_start_hour=16,
    on_peak_end_hour=20,
    mid_peak_start_hour=8,
    mid_peak_end_hour=22
)


class TOUTariffSchedule:
    """
    Generates time-series vector of electricity prices for simulation horizons.
    """

    def __init__(self, tariff: ElectricityTariff = TARIFF_DEFAULT_TOU):
        self.tariff = tariff

    def get_price_at_hour(self, hour_of_day: float, is_weekend: bool = False) -> float:
        """Determines $/kWh for given hour."""
        # Many utilities treat weekends as off-peak or mid-peak throughout
        if is_weekend:
            return self.tariff.off_peak_rate

        h = hour_of_day % 24.0
        if self.tariff.on_peak_start_hour <= h < self.tariff.on_peak_end_hour:
            return self.tariff.on_peak_rate
        elif self.tariff.mid_peak_start_hour <= h < self.tariff.mid_peak_end_hour:
            return self.tariff.mid_peak_rate
        else:
            return self.tariff.off_peak_rate

    def generate_schedule(self, total_steps: int, dt_hours: float,
                          start_hour_offset: float = 0.0) -> np.ndarray:
        """
        Builds complete 1D numpy array of electricity rates ($/kWh) for total_steps.
        """
        schedule = np.zeros(total_steps, dtype=float)
        for step in range(total_steps):
            total_hour = start_hour_offset + step * dt_hours
            day_idx = int(total_hour // 24)
            is_weekend = (day_idx % 7) in [5, 6]  # Sat/Sun
            schedule[step] = self.get_price_at_hour(total_hour, is_weekend=is_weekend)
        return schedule
