"""
CLI Runner for ThermoPilot Predictive HVAC Simulation.

Executes a 7-day comparative evaluation of Model Predictive Control vs
Rule-Based Thermostat Baseline, logging exact quantified metrics.
"""

import sys
import argparse
from thermopilot.config import (
    BUILDING_PRESETS,
    SimulationConfig,
    ComfortBounds
)
from thermopilot.pricing.tariffs import TARIFF_DEFAULT_TOU, TARIFF_AGGRESSIVE_PEAK
from thermopilot.benchmark.evaluator import BenchmarkEvaluator
from thermopilot.forecasting.model import ThermalForecaster
from thermopilot.forecasting.weather_client import WeatherClient
from thermopilot.physics.rc_model import BuildingThermalModel


def main():
    parser = argparse.ArgumentParser(description="ThermoPilot Simulation & Benchmark CLI")
    parser.add_argument("--building", choices=["residential", "commercial", "high_thermal_mass"],
                        default="residential", help="Building thermal profile")
    parser.add_argument("--days", type=int, default=5, help="Simulation evaluation days")
    parser.add_argument("--comfort-slack", type=float, default=50.0, help="Comfort violation penalty weight")
    parser.add_argument("--tracking-weight", type=float, default=0.02, help="Secondary target setpoint tracking weight")
    args = parser.parse_args()

    building_params = BUILDING_PRESETS[args.building]
    sim_config = SimulationConfig(dt_hours=0.5, horizon_hours=24, simulation_days=args.days)
    comfort = ComfortBounds(slack_penalty=args.comfort_slack)

    print("================================================================")
    print("  ThermoPilot: Predictive HVAC Optimization Engine")
    print(f"  Target: {building_params.name} | Duration: {args.days} Days")
    print("================================================================\n")

    # 1. Forecasting Layer Benchmark
    print("[1/3] Benchmarking Gradient-Boosted Thermal Forecaster vs Naive Baseline...")
    physics = BuildingThermalModel(building_params, dt_hours=sim_config.dt_hours)
    weather_client = WeatherClient()
    weather = weather_client.get_forecast(days=args.days + 2, dt_hours=sim_config.dt_hours)

    forecaster = ThermalForecaster(random_state=42)
    fb = forecaster.train_and_benchmark(physics, weather, train_samples=2500, test_samples=600)
    print(f"  -> Model RMSE: {fb.model_rmse:.4f} °C (Persistence Baseline: {fb.baseline_rmse:.4f} °C)")
    print(f"  -> Forecast Improvement: {fb.rmse_reduction_pct:.1f}% error reduction\n")

    # 2. Multi-Day MPC vs Thermostat Benchmark
    print(f"[2/3] Running {args.days}-Day Receding-Horizon MPC vs Rule-Based Thermostat...")
    evaluator = BenchmarkEvaluator(
        building=building_params,
        comfort=comfort,
        config=sim_config,
        tariff=TARIFF_DEFAULT_TOU
    )
    report = evaluator.run_benchmark(
        weather_series=weather,
        target_tracking_weight=args.tracking_weight
    )

    # 3. Print Results
    print("\n[3/3] Simulation Complete! Results:")
    print(report.format_table())

    print("\n----------------------------------------------------------------")
    print("  QUANTIFIED RESUME BULLET POINTS:")
    print(f"  • Reduced electricity costs by {report.cost_savings_pct:.1f}% (${report.cost_savings_usd:.2f} savings)")
    print(f"    over a {args.days}-day horizon vs. rule-based thermostat baseline")
    print(f"  • Shifted {report.peak_reduction_pct:.1f}% of peak-hour HVAC electrical load to cheaper off-peak windows")
    print(f"  • Forecasting model achieved {fb.model_rmse:.2f}°C RMSE ({fb.rmse_reduction_pct:.1f}% reduction vs baseline)")
    print("----------------------------------------------------------------\n")


if __name__ == "__main__":
    main()
