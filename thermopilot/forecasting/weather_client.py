"""
Open-Meteo Weather API Client with local caching and offline fallback.

Fetches outdoor temperature, relative humidity, and solar irradiance.
Caches JSON responses on disk to prevent API rate limits, ensure test
determinism, and provide smooth offline operation.
"""

import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import requests
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WeatherDataPoint:
    timestamp_hour: float       # Hour index relative to simulation start [hours]
    temp_out: float             # Outdoor ambient temperature [°C]
    humidity: float             # Relative humidity [%]
    solar_irradiance: float     # Direct normal solar irradiance [kW/m²]


class WeatherClient:
    """
    Client for Open-Meteo weather forecasts with caching and synthetic fallback.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = 5):
        self.cache_dir = cache_dir or Path(__file__).resolve().parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def get_forecast(self, latitude: float = 37.7749, longitude: float = -122.4194,
                     days: int = 7, dt_hours: float = 0.5) -> List[WeatherDataPoint]:
        """
        Retrieves weather forecast for given coordinate, interpolated to dt_hours.
        Falls back to cache or diurnal physics generator if network is unavailable.
        """
        cache_key = f"forecast_{latitude:.3f}_{longitude:.3f}_{days}d.json"
        cache_file = self.cache_dir / cache_key

        data = None
        # Try disk cache first
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info(f"Loaded weather forecast from local cache: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")

        # Fetch from Open-Meteo if not in cache
        if data is None:
            try:
                params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "hourly": ["temperature_2m", "relative_humidity_2m", "direct_normal_irradiance"],
                    "forecast_days": min(days, 14),
                    "timezone": "auto"
                }
                resp = requests.get(self.BASE_URL, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2)
                    logger.info("Fetched and cached weather from Open-Meteo API.")
                else:
                    logger.warning(f"Open-Meteo returned status {resp.status_code}. Falling back to synthetic.")
            except Exception as ex:
                logger.warning(f"Weather API request failed ({ex}). Utilizing realistic diurnal profile.")

        # Fallback to realistic diurnal physics if API unavailable
        if data is None or "hourly" not in data:
            return self.generate_synthetic_weather(days=days, dt_hours=dt_hours)

        return self._interpolate_hourly(data["hourly"], dt_hours=dt_hours, days=days)

    def _interpolate_hourly(self, hourly: dict, dt_hours: float, days: int) -> List[WeatherDataPoint]:
        raw_temps = hourly.get("temperature_2m", [])
        raw_hum = hourly.get("relative_humidity_2m", [])
        raw_solar = hourly.get("direct_normal_irradiance", [])

        total_hours = min(len(raw_temps), days * 24)
        if total_hours == 0:
            return self.generate_synthetic_weather(days=days, dt_hours=dt_hours)

        raw_x = np.arange(total_hours)
        interp_x = np.arange(0, total_hours, dt_hours)

        interp_temps = np.interp(interp_x, raw_x, raw_temps[:total_hours])
        interp_hum = np.interp(interp_x, raw_x, raw_hum[:total_hours]) if raw_hum else np.full_like(interp_x, 50.0)
        # Convert W/m² to kW/m²
        raw_solar_kw = [s / 1000.0 for s in raw_solar[:total_hours]] if raw_solar else [0.0] * total_hours
        interp_solar = np.interp(interp_x, raw_x, raw_solar_kw)

        result = []
        for t, temp, hum, sol in zip(interp_x, interp_temps, interp_hum, interp_solar):
            result.append(WeatherDataPoint(
                timestamp_hour=float(t),
                temp_out=float(temp),
                humidity=float(hum),
                solar_irradiance=max(0.0, float(sol))
            ))
        return result

    @staticmethod
    def generate_synthetic_weather(days: int = 7, dt_hours: float = 0.5,
                                   t_mean: float = 16.0, t_amplitude: float = 8.0) -> List[WeatherDataPoint]:
        """
        Generates realistic multi-day diurnal weather profile with day/night oscillations
        and realistic solar noon peak.
        """
        total_steps = int(round((days * 24.0) / dt_hours))
        result = []

        for step in range(total_steps):
            t_hour = step * dt_hours
            hour_of_day = t_hour % 24.0

            # Diurnal temperature cycle: lowest at 05:00, peak at 15:00
            temp = t_mean + t_amplitude * math.sin((hour_of_day - 9.0) * math.pi / 12.0)

            # Inverse humidity cycle (higher at night, lower at mid-day heat)
            humidity = 60.0 - 25.0 * math.sin((hour_of_day - 9.0) * math.pi / 12.0)

            # Solar irradiance (kW/m²), strictly positive between 06:00 and 18:00
            solar = 0.0
            if 6.0 <= hour_of_day <= 18.0:
                solar = 0.85 * math.sin((hour_of_day - 6.0) * math.pi / 12.0)

            result.append(WeatherDataPoint(
                timestamp_hour=float(t_hour),
                temp_out=round(temp, 2),
                humidity=round(humidity, 1),
                solar_irradiance=round(max(0.0, solar), 3)
            ))

        return result
