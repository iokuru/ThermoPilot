import pytest
import numpy as np

from thermopilot.config import PRESET_RESIDENTIAL
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState
from thermopilot.forecasting.weather_client import WeatherClient, WeatherDataPoint
from thermopilot.forecasting.model import ThermalForecaster


def test_weather_client_synthetic_diurnal():
    """Verify synthetic weather generator produces realistic physical ranges."""
    series = WeatherClient.generate_synthetic_weather(days=3, dt_hours=0.5, t_mean=18.0, t_amplitude=6.0)
    assert len(series) == 3 * 48

    temps = [p.temp_out for p in series]
    solar = [p.solar_irradiance for p in series]

    assert min(temps) >= 12.0
    assert max(temps) <= 24.0
    assert min(solar) >= 0.0
    assert max(solar) > 0.5


def test_thermal_forecaster_benchmarks_superior_to_naive():
    """Verify HistGradientBoosting drift model significantly outperforms naive persistence baseline."""
    physics = BuildingThermalModel(PRESET_RESIDENTIAL, dt_hours=0.5)
    weather = WeatherClient.generate_synthetic_weather(days=5, dt_hours=0.5)

    forecaster = ThermalForecaster(random_state=42)
    bench = forecaster.train_and_benchmark(
        physics, weather, train_samples=1000, test_samples=300
    )

    # The ML forecaster should outperform the naive baseline by at least 60%
    assert bench.model_rmse < bench.baseline_rmse
    assert bench.rmse_reduction_pct > 60.0
    assert bench.model_rmse < 0.15, f"Model RMSE {bench.model_rmse:.4f} °C is unexpectedly high"


def test_predict_next_temp_direction():
    """Verify trained model correctly infers heating increases temperature."""
    physics = BuildingThermalModel(PRESET_RESIDENTIAL, dt_hours=0.5)
    weather = WeatherClient.generate_synthetic_weather(days=3, dt_hours=0.5)

    forecaster = ThermalForecaster(random_state=42)
    forecaster.train_and_benchmark(physics, weather, train_samples=800, test_samples=200)

    st = ThermalState(temp_in=21.0, temp_wall=21.0)
    w = WeatherDataPoint(timestamp_hour=12.0, temp_out=15.0, humidity=50.0, solar_irradiance=0.0)

    t_next_heat = forecaster.predict_next_temp(st, q_hvac=5.0, weather=w)
    t_next_cool = forecaster.predict_next_temp(st, q_hvac=-5.0, weather=w)

    assert t_next_heat > st.temp_in
    assert t_next_cool < st.temp_in
