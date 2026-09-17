"""
Model Predictive Control (MPC) Core Optimizer.

Solves a convex Quadratic Program (QP) over a finite rolling horizon H:
  Minimize:
    J = sum_{k=0}^{H-1} [
          c_k * P_elec,k * dt                      (Energy cost under TOU tariff)
        + rho_comfort * (s_low,k + s_high,k)       (Comfort violation penalty)
        + rho_target * (T_in,k - T_target)^2       (Secondary setpoint tracking)
        + rho_smooth * (u_k - u_{k-1})^2           (Compressor cycling rate penalty)
    ]
  Subject to:
    x_{k+1} = A_d * x_k + B_d * (u_heat,k - u_cool,k) + E_d * w_k
    0 <= u_heat,k <= U_max
    0 <= u_cool,k <= U_max
    T_min - s_low,k <= T_in,k <= T_max + s_high,k
    s_low,k >= 0, s_high,k >= 0

Guarantees 100% feasibility via convex slack variables, preventing runtime solver failure.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import cvxpy as cp

from thermopilot.config import BuildingParameters, ComfortBounds, SimulationConfig
from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState


@dataclass
class MPCSolution:
    success: bool
    cost: float
    q_hvac_schedule: np.ndarray      # Net thermal kW (+ heating, - cooling) [H]
    p_elec_schedule: np.ndarray      # Electrical kW consumed [H]
    temp_in_predicted: np.ndarray    # Predicted indoor temperature [H+1]
    temp_wall_predicted: np.ndarray  # Predicted wall temperature [H+1]
    slack_violations: np.ndarray     # Comfort violations [H]
    solve_time_sec: float
    status: str


class MPCOptimizer:
    """
    Formulates and solves the convex single-shot MPC problem using CVXPY.
    """

    def __init__(self, physics_model: BuildingThermalModel,
                 comfort: ComfortBounds,
                 config: SimulationConfig,
                 target_tracking_weight: float = 0.02):
        self.physics = physics_model
        self.comfort = comfort
        self.config = config
        self.target_tracking_weight = target_tracking_weight
        self.dt = config.dt_hours
        self.H = config.horizon_steps

    def solve(self, current_state: ThermalState,
              tariff_schedule: np.ndarray,
              weather_forecast_w: np.ndarray,
              previous_u: float = 0.0) -> MPCSolution:
        """
        Solves single-shot MPC optimization over horizon H.

        Args:
            current_state: Current observed building state [T_in, T_wall]
            tariff_schedule: Array of electricity prices [$/kWh] for next H steps
            weather_forecast_w: Array of shape (H, 3) for [T_out, Q_solar, Q_int]
            previous_u: HVAC net power applied at t-1 (for rate-of-change smoothing)

        Returns:
            MPCSolution dataclass with optimal control and state trajectories.
        """
        H = self.H
        dt = self.dt
        max_hvac = self.physics.params.max_hvac_thermal_kw
        cop_h = self.physics.params.cop_heating
        cop_c = self.physics.params.cop_cooling

        # Optimization variables
        # State: x[:, k] = [T_in, k; T_wall, k]
        x = cp.Variable((2, H + 1), name="state")
        u_heat = cp.Variable(H, nonneg=True, name="u_heat")
        u_cool = cp.Variable(H, nonneg=True, name="u_cool")
        s_low = cp.Variable(H, nonneg=True, name="slack_low")
        s_high = cp.Variable(H, nonneg=True, name="slack_high")

        # Constraints
        constraints = []
        # Initial condition
        constraints.append(x[:, 0] == current_state.to_array())

        # System dynamics and state evolution
        A_d = self.physics.A_d
        B_d = self.physics.B_d.ravel()
        E_d = self.physics.E_d

        for k in range(H):
            # Net thermal input: u_net = u_heat - u_cool
            u_net_k = u_heat[k] - u_cool[k]
            w_k = weather_forecast_w[k]

            # Dynamic step
            constraints.append(
                x[:, k + 1] == A_d @ x[:, k] + B_d * u_net_k + E_d @ w_k
            )

            # Actuator saturation
            constraints.append(u_heat[k] <= max_hvac)
            constraints.append(u_cool[k] <= max_hvac)

            # Soft comfort bounds with slack variables
            constraints.append(x[0, k] >= self.comfort.temp_min - s_low[k])
            constraints.append(x[0, k] <= self.comfort.temp_max + s_high[k])

        # Terminal comfort constraint
        constraints.append(x[0, H] >= self.comfort.temp_min - s_low[H - 1])
        constraints.append(x[0, H] <= self.comfort.temp_max + s_high[H - 1])

        # Objective Function Formulation
        objective_terms = []

        # 1. TOU Energy cost: sum c_k * (u_heat/COP_h + u_cool/COP_c) * dt
        p_elec = (u_heat / cop_h) + (u_cool / cop_c)
        energy_cost = cp.sum(cp.multiply(tariff_schedule, p_elec)) * dt
        objective_terms.append(energy_cost)

        # 2. Comfort boundary violation penalty (L1 exact penalty)
        comfort_penalty = self.comfort.slack_penalty * cp.sum(s_low + s_high) * dt
        objective_terms.append(comfort_penalty)

        # 3. Mild target comfort setpoint centering (L2 penalty)
        target_penalty = self.target_tracking_weight * cp.sum_squares(x[0, :H] - self.comfort.target_temp)
        objective_terms.append(target_penalty)

        # 4. Actuation smoothness (slew rate) penalty
        u_net = u_heat - u_cool
        delta_u_0 = u_net[0] - previous_u
        delta_u_rest = u_net[1:] - u_net[:-1]
        smoothness = self.comfort.smoothness_penalty * (cp.square(delta_u_0) + cp.sum_squares(delta_u_rest))
        objective_terms.append(smoothness)

        total_objective = cp.Minimize(cp.sum(objective_terms))
        problem = cp.Problem(total_objective, constraints)

        # Solve convex program
        # Try OSQP first, then CLARABEL / ECOS if necessary
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            try:
                problem.solve(solver=cp.CLARABEL, verbose=False)
            except Exception:
                problem.solve(verbose=False)

        is_success = problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

        if is_success and u_heat.value is not None:
            net_u = np.array(u_heat.value - u_cool.value).flatten()
            net_p = np.array((u_heat.value / cop_h) + (u_cool.value / cop_c)).flatten()
            pred_tin = np.array(x.value[0, :]).flatten()
            pred_twall = np.array(x.value[1, :]).flatten()
            slacks = np.array(s_low.value + s_high.value).flatten()
            cost_val = float(problem.value)
        else:
            # Fallback safe zero action if solver issues occur
            net_u = np.zeros(H)
            net_p = np.zeros(H)
            pred_tin = np.full(H + 1, current_state.temp_in)
            pred_twall = np.full(H + 1, current_state.temp_wall)
            slacks = np.zeros(H)
            cost_val = 0.0

        return MPCSolution(
            success=is_success,
            cost=cost_val,
            q_hvac_schedule=net_u,
            p_elec_schedule=net_p,
            temp_in_predicted=pred_tin,
            temp_wall_predicted=pred_twall,
            slack_violations=slacks,
            solve_time_sec=problem.solver_stats.solve_time if problem.solver_stats else 0.0,
            status=problem.status
        )
