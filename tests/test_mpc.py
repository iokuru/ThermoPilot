import pytest
import numpy as np

from thermopilot.config import PRESET_RESIDENTIAL, ComfortBounds, SimulationConfig
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState
from thermopilot.forecasting.weather_client import WeatherDataPoint, WeatherClient
from thermopilot.pricing.tariffs import TARIFF_DEFAULT_TOU, TOUTariffSchedule
from thermopilot.mpc.optimizer import MPCOptimizer, MPCSolution
from thermopilot.mpc.controller import RecedingHorizonController


@pytest.fixture
def mpc_setup():
    physics = BuildingThermalModel(PRESET_RESIDENTIAL, dt_hours=0.5)
    comfort = ComfortBounds(temp_min=20.0, temp_max=23.5, target_temp=21.75)
    config = SimulationConfig(dt_hours=0.5, horizon_hours=24)
    optimizer = MPCOptimizer(physics, comfort, config)
    return physics, comfort, config, optimizer


def test_mpc_single_shot_optimality(mpc_setup):
    """Verify single-shot QP solves to OPTIMAL status and respects comfort bounds."""
    physics, comfort, config, optimizer = mpc_setup
    state = ThermalState(temp_in=21.5, temp_wall=21.0)

    # 48 steps (24 hours)
    H = config.horizon_steps
    tariffs = np.full(H, 0.20)
    # Afternoon price spike
    tariffs[20:30] = 0.60

    weather = np.zeros((H, 3))
    weather[:, 0] = 26.0  # Warm outside (cooling needed)
    weather[:, 1] = 0.5   # Moderate solar
    weather[:, 2] = 0.6   # Base internal gain

    sol: MPCSolution = optimizer.solve(state, tariffs, weather)

    assert sol.success, f"MPC failed with status: {sol.status}"
    assert len(sol.q_hvac_schedule) == H
    assert len(sol.temp_in_predicted) == H + 1

    # Predicted temperature should remain within comfort limits [20, 23.5]
    assert np.all(sol.temp_in_predicted >= comfort.temp_min - 0.2)
    assert np.all(sol.temp_in_predicted <= comfort.temp_max + 0.2)


def test_mpc_anticipatory_precooling(mpc_setup):
    """Verify MPC pre-cools thermal mass before an expensive price surge."""
    physics, comfort, config, optimizer = mpc_setup
    state = ThermalState(temp_in=22.5, temp_wall=22.5)

    H = config.horizon_steps
    # Cheap off-peak price, then severe price spike at k=16..24
    tariffs = np.full(H, 0.10)
    tariffs[16:24] = 0.80

    weather = np.zeros((H, 3))
    weather[:, 0] = 28.0  # Hot summer ambient (needs cooling)
    weather[:, 1] = 0.2
    weather[:, 2] = 0.5

    sol: MPCSolution = optimizer.solve(state, tariffs, weather)
    assert sol.success

    # Pre-cooling: cooling should be active before k=16 (negative q_hvac)
    pre_spike_cooling = np.sum(sol.q_hvac_schedule[10:16] < -0.5)
    assert pre_spike_cooling > 0, "MPC should pre-cool building before price spike"

    # During peak tariff (k=16..24), cooling should be curtailed / throttled
    avg_peak_cooling = np.mean(np.abs(sol.q_hvac_schedule[16:24]))
    avg_pre_cooling = np.mean(np.abs(sol.q_hvac_schedule[10:16]))
    assert avg_peak_cooling <= avg_pre_cooling, "MPC should curtail HVAC power during peak price spike"


def test_receding_horizon_controller_execution(mpc_setup):
    """Verify closed-loop receding horizon controller executes smoothly over multiple days."""
    physics, comfort, config, optimizer = mpc_setup
    config.simulation_days = 1  # 1 day fast integration test

    controller = RecedingHorizonController(optimizer, physics, config)
    weather = WeatherClient.generate_synthetic_weather(days=3, dt_hours=config.dt_hours)
    tariff_gen = TOUTariffSchedule(TARIFF_DEFAULT_TOU)
    tariffs = tariff_gen.generate_schedule(total_steps=len(weather), dt_hours=config.dt_hours)

    init_state = ThermalState(temp_in=21.5, temp_wall=21.5)
    telemetry = controller.run(init_state, weather, tariffs)

    assert len(telemetry.time_hours) > 0
    assert telemetry.total_energy_kwh > 0.0
    assert telemetry.total_cost_usd > 0.0
    assert telemetry.comfort_violation_c_hours < 5.0, "Violations should remain minimal"
