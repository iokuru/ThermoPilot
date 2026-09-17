"""
Rule-Based Dual-Setpoint Hysteresis Thermostat Baseline.

Simulates the standard industry-standard HVAC thermostat controller:
  - Bang-bang / hysteretic control around heating and cooling setpoints.
  - Price-agnostic: reacts only to immediate temperature boundary exceedances.
  - Generates realistic baseline telemetry for direct comparison against MPC.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from thermopilot.config import BuildingParameters, ComfortBounds, SimulationConfig
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState
from thermopilot.forecasting.weather_client import WeatherDataPoint
from thermopilot.mpc.controller import SimulationTelemetry


class ThermostatMode(str, Enum):
    OFF = "OFF"
    HEATING = "HEATING"
    COOLING = "COOLING"


class RuleBasedThermostat:
    """
    Standard hysteretic deadband thermostat.
    """

    def __init__(self, physics: BuildingThermalModel,
                 comfort: ComfortBounds,
                 cooling_setpoint: float = 22.5,
                 heating_setpoint: float = 21.0,
                 deadband: float = 0.3):
        self.physics = physics
        self.comfort = comfort
        self.cooling_setpoint = cooling_setpoint
        self.heating_setpoint = heating_setpoint
        self.deadband = deadband
        self.mode = ThermostatMode.OFF

    def compute_action(self, current_temp: float) -> float:
        """
        Determines thermal HVAC power [kW].
        Positive for heating, negative for cooling, zero for off.
        """
        max_power = self.physics.params.max_hvac_thermal_kw

        # Heating logic with hysteresis deadband
        if current_temp < (self.heating_setpoint - self.deadband):
            self.mode = ThermostatMode.HEATING
        elif current_temp >= self.heating_setpoint and self.mode == ThermostatMode.HEATING:
            self.mode = ThermostatMode.OFF

        # Cooling logic with hysteresis deadband
        if current_temp > (self.cooling_setpoint + self.deadband):
            self.mode = ThermostatMode.COOLING
        elif current_temp <= self.cooling_setpoint and self.mode == ThermostatMode.COOLING:
            self.mode = ThermostatMode.OFF

        if self.mode == ThermostatMode.HEATING:
            deficit = self.heating_setpoint - current_temp
            power = min(max_power, max(2.5, deficit * 6.0))
            return float(power)
        elif self.mode == ThermostatMode.COOLING:
            excess = current_temp - self.cooling_setpoint
            power = min(max_power, max(2.5, excess * 6.0))
            return float(-power)
        else:
            return 0.0

    def run_simulation(self, initial_state: ThermalState,
                       full_weather: List[WeatherDataPoint],
                       full_tariffs: np.ndarray,
                       config: SimulationConfig,
                       process_noise_std: float = 0.05) -> SimulationTelemetry:
        """
        Executes multi-day simulation under rule-based thermostat control.
        """
        dt = config.dt_hours
        H = config.horizon_steps
        total_steps = min(config.total_steps, len(full_weather) - H, len(full_tariffs) - H)

        cop_h = self.physics.params.cop_heating
        cop_c = self.physics.params.cop_cooling

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
        cum_cost = 0.0
        comfort_violation_deg_hours = 0.0
        rng = np.random.RandomState(42)

        for step in range(total_steps):
            t_hour = step * dt
            weather_now = full_weather[step]
            price_now = full_tariffs[step]

            # 1. Thermostat decides instantaneous control action
            q_hvac = self.compute_action(current_state.temp_in)

            # 2. Electrical power consumption
            if q_hvac > 0:
                p_elec = q_hvac / cop_h
            elif q_hvac < 0:
                p_elec = abs(q_hvac) / cop_c
            else:
                p_elec = 0.0

            # 3. Step physical plant (RK4)
            next_state = self.physics.step_rk4(
                current_state,
                q_hvac=q_hvac,
                t_out=weather_now.temp_out,
                q_solar=weather_now.solar_irradiance,
                q_int=self.physics.params.internal_gain_base,
                dt=dt
            )
            if process_noise_std > 0.0:
                noise = rng.normal(0.0, process_noise_std)
                next_state.temp_in += noise

            # 4. Accounting & metrics
            step_cost = p_elec * price_now * dt
            cum_cost += step_cost

            t_min = self.comfort.temp_min
            t_max = self.comfort.temp_max
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
            log_qhvac[step] = q_hvac
            log_pelec[step] = p_elec
            log_price[step] = price_now
            log_step_cost[step] = step_cost
            log_cum_cost[step] = cum_cost

            current_state = next_state

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
