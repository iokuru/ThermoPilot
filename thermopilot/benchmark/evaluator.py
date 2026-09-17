"""
Comparative Benchmark Evaluator.

Executes side-by-side simulations of:
  1. Predictive Model Predictive Control (MPC)
  2. Rule-Based Hysteretic Thermostat Baseline

Under identical thermal models, weather inputs, and Time-of-Use electricity tariffs.
Computes quantifiable performance KPIs for resume metrics and portfolio validation.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import pandas as pd

from thermopilot.config import BuildingParameters, ComfortBounds, SimulationConfig, PRESET_RESIDENTIAL
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState
from thermopilot.forecasting.weather_client import WeatherClient, WeatherDataPoint
from thermopilot.pricing.tariffs import ElectricityTariff, TOUTariffSchedule, TARIFF_DEFAULT_TOU
from thermopilot.mpc.optimizer import MPCOptimizer
from thermopilot.mpc.controller import RecedingHorizonController, SimulationTelemetry
from thermopilot.baseline.thermostat import RuleBasedThermostat


@dataclass
class ComparativeBenchmarkReport:
    mpc_telemetry: SimulationTelemetry
    baseline_telemetry: SimulationTelemetry
    building_name: str
    simulation_days: float

    # Quantified Summary KPIs
    mpc_cost: float
    baseline_cost: float
    cost_savings_usd: float
    cost_savings_pct: float

    mpc_energy_kwh: float
    baseline_energy_kwh: float
    energy_savings_pct: float

    mpc_peak_kwh: float
    baseline_peak_kwh: float
    peak_reduction_pct: float

    mpc_comfort_violation_ch: float
    baseline_comfort_violation_ch: float

    def format_table(self) -> str:
        s = []
        s.append(f"================================================================")
        s.append(f"  ThermoPilot Benchmark Evaluation Report: {self.building_name}")
        s.append(f"  Duration: {self.simulation_days:.1f} Days")
        s.append(f"================================================================")
        s.append(f"{'Metric':<32} | {'Rule-Based':<12} | {'MPC Pilot':<12} | {'Delta':<10}")
        s.append(f"---------------------------------+--------------+--------------+-----------")
        s.append(f"{'Total Cost ($)':<32} | ${self.baseline_cost:<11.2f} | ${self.mpc_cost:<11.2f} | {self.cost_savings_pct:+.1f}%")
        s.append(f"{'Total Electricity (kWh)':<32} | {self.baseline_energy_kwh:<12.1f} | {self.mpc_energy_kwh:<12.1f} | {self.energy_savings_pct:+.1f}%")
        s.append(f"{'Peak-Hours Energy (kWh)':<32} | {self.baseline_peak_kwh:<12.1f} | {self.mpc_peak_kwh:<12.1f} | {self.peak_reduction_pct:+.1f}%")
        s.append(f"{'Comfort Violations (°C·hr)':<32} | {self.baseline_comfort_violation_ch:<12.2f} | {self.mpc_comfort_violation_ch:<12.2f} | {self.mpc_comfort_violation_ch - self.baseline_comfort_violation_ch:+.2f}")
        s.append(f"================================================================")
        return "\n".join(s)


class BenchmarkEvaluator:
    """
    Orchestrates comparative benchmarking between MPC and the baseline thermostat.
    """

    def __init__(self, building: BuildingParameters = PRESET_RESIDENTIAL,
                 comfort: ComfortBounds = ComfortBounds(),
                 config: SimulationConfig = SimulationConfig(),
                 tariff: ElectricityTariff = TARIFF_DEFAULT_TOU):
        self.building = building
        self.comfort = comfort
        self.config = config
        self.tariff = tariff

        self.physics = BuildingThermalModel(building, dt_hours=config.dt_hours)
        self.tariff_schedule_gen = TOUTariffSchedule(tariff)

    def run_benchmark(self, weather_series: Optional[List[WeatherDataPoint]] = None,
                      initial_state: Optional[ThermalState] = None,
                      target_tracking_weight: float = 0.02) -> ComparativeBenchmarkReport:
        """
        Executes both controllers under identical inputs.
        """
        dt = self.config.dt_hours
        total_steps_needed = self.config.total_steps + self.config.horizon_steps + 10

        if weather_series is None or len(weather_series) < total_steps_needed:
            # Generate realistic multi-day weather
            weather_client = WeatherClient()
            weather_series = weather_client.get_forecast(
                days=self.config.simulation_days + 2,
                dt_hours=dt
            )

        tariffs = self.tariff_schedule_gen.generate_schedule(
            total_steps=len(weather_series),
            dt_hours=dt
        )

        if initial_state is None:
            initial_state = ThermalState(
                temp_in=self.comfort.target_temp,
                temp_wall=self.comfort.target_temp - 0.5
            )

        # 1. Run Rule-Based Thermostat Baseline
        thermostat = RuleBasedThermostat(
            self.physics, self.comfort,
            cooling_setpoint=self.comfort.temp_max - 1.0,
            heating_setpoint=self.comfort.temp_min + 1.0,
            deadband=0.3
        )
        baseline_tel = thermostat.run_simulation(
            initial_state=initial_state,
            full_weather=weather_series,
            full_tariffs=tariffs,
            config=self.config
        )

        # 2. Run MPC Controller
        optimizer = MPCOptimizer(
            physics_model=self.physics,
            comfort=self.comfort,
            config=self.config,
            target_tracking_weight=target_tracking_weight
        )
        mpc_controller = RecedingHorizonController(
            optimizer=optimizer,
            plant_physics=self.physics,
            config=self.config
        )
        mpc_tel = mpc_controller.run(
            initial_state=initial_state,
            full_weather=weather_series,
            full_tariffs=tariffs
        )

        # 3. Calculate comparative metrics
        base_cost = baseline_tel.total_cost_usd
        mpc_cost = mpc_tel.total_cost_usd
        cost_savings = base_cost - mpc_cost
        cost_savings_pct = (cost_savings / base_cost) * 100.0 if base_cost > 0 else 0.0

        base_energy = baseline_tel.total_energy_kwh
        mpc_energy = mpc_tel.total_energy_kwh
        energy_savings_pct = ((base_energy - mpc_energy) / base_energy) * 100.0 if base_energy > 0 else 0.0

        # Peak hours analysis (where tariff == on_peak_rate)
        is_peak = mpc_tel.electricity_price >= (self.tariff.on_peak_rate - 1e-4)
        mpc_peak_kwh = float(np.sum(mpc_tel.p_elec[is_peak]) * dt)
        base_peak_kwh = float(np.sum(baseline_tel.p_elec[is_peak]) * dt)
        peak_reduction_pct = ((base_peak_kwh - mpc_peak_kwh) / base_peak_kwh) * 100.0 if base_peak_kwh > 0 else 0.0

        return ComparativeBenchmarkReport(
            mpc_telemetry=mpc_tel,
            baseline_telemetry=baseline_tel,
            building_name=self.building.name,
            simulation_days=self.config.simulation_days,
            mpc_cost=mpc_cost,
            baseline_cost=base_cost,
            cost_savings_usd=cost_savings,
            cost_savings_pct=cost_savings_pct,
            mpc_energy_kwh=mpc_energy,
            baseline_energy_kwh=base_energy,
            energy_savings_pct=energy_savings_pct,
            mpc_peak_kwh=mpc_peak_kwh,
            baseline_peak_kwh=base_peak_kwh,
            peak_reduction_pct=peak_reduction_pct,
            mpc_comfort_violation_ch=mpc_tel.comfort_violation_c_hours,
            baseline_comfort_violation_ch=baseline_tel.comfort_violation_c_hours
        )
