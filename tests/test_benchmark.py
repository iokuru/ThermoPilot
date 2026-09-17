import pytest
from thermopilot.config import PRESET_RESIDENTIAL, SimulationConfig, ComfortBounds
from thermopilot.pricing.tariffs import TARIFF_DEFAULT_TOU
from thermopilot.benchmark.evaluator import BenchmarkEvaluator
from thermopilot.forecasting.weather_client import WeatherClient


def test_comparative_benchmark_mpc_savings():
    """Verify MPC reduces total electricity cost and sheds peak load vs rule-based thermostat."""
    config = SimulationConfig(dt_hours=0.5, horizon_hours=24, simulation_days=2)
    comfort = ComfortBounds(temp_min=20.0, temp_max=23.5, slack_penalty=50.0)

    evaluator = BenchmarkEvaluator(
        building=PRESET_RESIDENTIAL,
        comfort=comfort,
        config=config,
        tariff=TARIFF_DEFAULT_TOU
    )

    weather = WeatherClient.generate_synthetic_weather(
        days=4, dt_hours=0.5, t_mean=27.0, t_amplitude=7.0
    )

    report = evaluator.run_benchmark(weather_series=weather)

    # Verifications
    assert report.baseline_cost > 0.0
    assert report.mpc_cost < report.baseline_cost, (
        f"MPC (${report.mpc_cost:.2f}) did not achieve lower cost than baseline (${report.baseline_cost:.2f})"
    )
    assert report.cost_savings_pct > 5.0, (
        f"Expected at least 5% cost savings, got {report.cost_savings_pct:.1f}%"
    )
    assert report.peak_reduction_pct > 40.0, (
        f"Expected at least 40% peak load reduction, got {report.peak_reduction_pct:.1f}%"
    )
    assert report.mpc_comfort_violation_ch < 1.0, (
        f"MPC comfort violations should be negligible, got {report.mpc_comfort_violation_ch:.2f} °C·hr"
    )
