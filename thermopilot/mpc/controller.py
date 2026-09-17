"""
Receding Horizon Controller (Closed-Loop MPC).

Executes the continuous receding-horizon loop:
  1. Samples current true plant state x_t from physical environment.
  2. Extracts rolling forecast window [t, t + H] for weather and electricity tariffs.
  3. Re-solves the convex optimization problem over horizon H.
  4. Injects only the first optimal control action u*_0 into the physical plant.
  5. Advances simulation forward by dt and records telemetry.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from thermopilot.config import SimulationConfig, ComfortBounds
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState
from thermopilot.forecasting.weather_client import WeatherDataPoint
from thermopilot.mpc.optimizer import MPCOptimizer, MPCSolution


@dataclass
class SimulationTelemetry:
    time_hours: np.ndarray
    temp_in: np.ndarray
    temp_wall: np.ndarray
    temp_out: np.ndarray
    solar_irradiance: np.ndarray
    q_hvac: np.ndarray
    p_elec: np.ndarray
    electricity_price: np.ndarray
    step_cost: np.ndarray
    cumulative_cost: np.ndarray
    comfort_violation_c_hours: float
    total_energy_kwh: float
    total_cost_usd: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "hour": self.time_hours,
            "temp_in": self.temp_in,
            "temp_wall": self.temp_wall,
            "temp_out": self.temp_out,
            "solar_kw": self.solar_irradiance,
            "q_hvac_kw": self.q_hvac,
            "p_elec_kw": self.p_elec,
            "price_per_kwh": self.electricity_price,
            "step_cost": self.step_cost,
            "cumulative_cost": self.cumulative_cost
        })


class RecedingHorizonController:
    """
    Closed-loop receding horizon control runner.
    """

    def __init__(self, optimizer: MPCOptimizer,
                 plant_physics: BuildingThermalModel,
                 config: SimulationConfig):
        self.optimizer = optimizer
        self.plant = plant_physics
        self.config = config
        self.dt = config.dt_hours

    def run(self, initial_state: ThermalState,
            full_weather: List[WeatherDataPoint],
            full_tariffs: np.ndarray,
            process_noise_std: float = 0.05) -> SimulationTelemetry:
        """
        Runs receding horizon loop across the entire multi-day simulation.

        Args:
            initial_state: Building state at t=0
            full_weather: Full sequence of weather data points
            full_tariffs: Full sequence of electricity tariffs
            process_noise_std: Realistic sensor/process thermal noise standard deviation
        """
        dt = self.dt
        H = self.config.horizon_steps
        total_steps = min(self.config.total_steps, len(full_weather) - H, len(full_tariffs) - H)
        assert total_steps > 0, "Weather/tariff series shorter than rolling horizon H."

        # Arrays for logging telemetry
        log_time = np.zeros(total_steps)
        log_tin = np.zeros(total_steps)
        log_twall = np.zeros(total_steps)
        log_tout = np.zeros(total_steps)
        log_solar = np.zeros(total_steps)
        log_qhvac = np.zeros(total_steps)
        log_pelec = np.zeros(total_steps)
        log_price = np.zeros(total_steps)
        log_step_cost = np.zeros(total_steps)
        log_cum_cost = np.zeros(total_steps)

        current_state = initial_state
        prev_u = 0.0
        cum_cost = 0.0
        comfort_violation_deg_hours = 0.0

        rng = np.random.RandomState(42)

        for step in range(total_steps):
            t_hour = step * dt
            weather_now = full_weather[step]
            price_now = full_tariffs[step]

            # 1. Extract horizon forecast matrices
            tariff_window = full_tariffs[step: step + H]
            weather_window = np.zeros((H, 3), dtype=float)
            for k in range(H):
                w_pt = full_weather[step + k]
                weather_window[k, 0] = w_pt.temp_out
                weather_window[k, 1] = w_pt.solar_irradiance
                weather_window[k, 2] = self.plant.params.internal_gain_base

            # 2. Re-solve MPC optimization over rolling horizon
            sol: MPCSolution = self.optimizer.solve(
                current_state=current_state,
                tariff_schedule=tariff_window,
                weather_forecast_w=weather_window,
                previous_u=prev_u
            )

            # 3. Extract first control action
            u_0 = float(sol.q_hvac_schedule[0])
            p_0 = float(sol.p_elec_schedule[0])

            # 4. Step true continuous physical plant (using RK4 ground truth)
            # Add mild stochastic disturbance to simulate plant-model mismatch
            next_state = self.plant.step_rk4(
                current_state,
                q_hvac=u_0,
                t_out=weather_now.temp_out,
                q_solar=weather_now.solar_irradiance,
                q_int=self.plant.params.internal_gain_base,
                dt=dt
            )
            if process_noise_std > 0.0:
                noise = rng.normal(0.0, process_noise_std)
                next_state.temp_in += noise

            # 5. Compute accounting metrics
            step_cost = p_0 * price_now * dt
            cum_cost += step_cost

            # Check comfort violation degree-hours
            t_min = self.optimizer.comfort.temp_min
            t_max = self.optimizer.comfort.temp_max
            violation = 0.0
            if current_state.temp_in < t_min:
                violation = (t_min - current_state.temp_in) * dt
            elif current_state.temp_in > t_max:
                violation = (current_state.temp_in - t_max) * dt
            comfort_violation_deg_hours += violation

            # Record telemetry
            log_time[step] = t_hour
            log_tin[step] = current_state.temp_in
            log_twall[step] = current_state.temp_wall
            log_tout[step] = weather_now.temp_out
            log_solar[step] = weather_now.solar_irradiance
            log_qhvac[step] = u_0
            log_pelec[step] = p_0
            log_price[step] = price_now
            log_step_cost[step] = step_cost
            log_cum_cost[step] = cum_cost

            # Advance state
            current_state = next_state
            prev_u = u_0

        total_energy = float(np.sum(log_pelec) * dt)

        return SimulationTelemetry(
            time_hours=log_time,
            temp_in=log_tin,
            temp_wall=log_twall,
            temp_out=log_tout,
            solar_irradiance=log_solar,
            q_hvac=log_qhvac,
            p_elec=log_pelec,
            electricity_price=log_price,
            step_cost=log_step_cost,
            cumulative_cost=log_cum_cost,
            comfort_violation_c_hours=comfort_violation_deg_hours,
            total_energy_kwh=total_energy,
            total_cost_usd=cum_cost
        )
