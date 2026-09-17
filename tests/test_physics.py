import pytest
import numpy as np

from thermopilot.config import PRESET_RESIDENTIAL, PRESET_COMMERCIAL, PRESET_HIGH_THERMAL_MASS
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState


@pytest.fixture
def residential_model():
    return BuildingThermalModel(PRESET_RESIDENTIAL, dt_hours=0.5)


def test_thermal_stability_eigenvalues(residential_model):
    """Verify system is strictly Hurwitz stable (all continuous eigenvalues negative)."""
    eigvals = np.linalg.eigvals(residential_model.A_c)
    assert np.all(np.real(eigvals) < 0.0), "Continuous system matrix A_c must be strictly stable"

    tau_fast, tau_slow = residential_model.compute_time_constants()
    assert 0.5 < tau_fast < 5.0, f"Fast time constant {tau_fast:.2f}h outside realistic range (0.5 - 5h)"
    assert 10.0 < tau_slow < 100.0, f"Slow time constant {tau_slow:.2f}h outside realistic range (10 - 100h)"


def test_passive_equilibrium_convergence(residential_model):
    """Under zero internal/solar/HVAC load, temperature must converge to outdoor ambient."""
    t_out = 5.0  # Cold winter day
    ss = residential_model.compute_steady_state(t_out=t_out, q_hvac=0.0, q_solar=0.0, q_int=0.0)

    assert pytest.approx(ss.temp_in, rel=1e-3) == t_out
    assert pytest.approx(ss.temp_wall, rel=1e-3) == t_out


def test_decay_toward_ambient_monotonically(residential_model):
    """Hot initial building with no heat input must monotonically cool towards ambient."""
    state = ThermalState(temp_in=25.0, temp_wall=24.0)
    t_out = 10.0

    temperatures = [state.temp_in]
    for _ in range(48):  # 24 hours (48 * 0.5h)
        state = residential_model.step_rk4(state, q_hvac=0.0, t_out=t_out, q_solar=0.0, q_int=0.0)
        temperatures.append(state.temp_in)

    # Must decrease monotonically
    diffs = np.diff(temperatures)
    assert np.all(diffs <= 0.0), "Passive cooling must be monotonically decreasing"
    assert state.temp_in < 25.0
    assert state.temp_in > t_out


def test_discrete_vs_rk4_consistency(residential_model):
    """Exact matrix exponential ZOH should closely match RK4 continuous simulation."""
    state = ThermalState(temp_in=20.0, temp_wall=19.5)
    t_out = 2.0
    q_hvac = 3.5
    q_solar = 0.4
    q_int = 0.6

    rk4_next = residential_model.step_rk4(state, q_hvac, t_out, q_solar, q_int)
    disc_next = residential_model.step_discrete(state, q_hvac, t_out, q_solar, q_int)

    assert pytest.approx(rk4_next.temp_in, abs=0.05) == disc_next.temp_in
    assert pytest.approx(rk4_next.temp_wall, abs=0.05) == disc_next.temp_wall


def test_hvac_heating_and_cooling_directionality(residential_model):
    """Positive HVAC power must raise indoor temp; negative power must lower it."""
    init_state = ThermalState(temp_in=20.0, temp_wall=20.0)
    t_out = 20.0

    heat_state = residential_model.step_rk4(init_state, q_hvac=4.0, t_out=t_out, q_solar=0.0, q_int=0.0)
    cool_state = residential_model.step_rk4(init_state, q_hvac=-4.0, t_out=t_out, q_solar=0.0, q_int=0.0)

    assert heat_state.temp_in > init_state.temp_in
    assert cool_state.temp_in < init_state.temp_in
