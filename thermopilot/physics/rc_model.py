"""
Building Thermal Physics Simulation (2R2C Model).

Models the thermal dynamics of a building as a 2-Resistor 2-Capacitor lumped network:
  - C_air: Indoor air and immediate thermal mass (fast time constant).
  - C_wall: Building structural envelope mass (slow time constant).
  - R_int: Internal heat transfer resistance between air and wall mass.
  - R_envelope: Exterior insulation thermal resistance to outdoor ambient.

Provides exact discrete-time state-space matrices via matrix exponential (ZOH)
as well as Runge-Kutta 4th Order (RK4) continuous integration for physical ground truth.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.linalg import expm

from thermopilot.config import BuildingParameters


@dataclass
class ThermalState:
    temp_in: float      # Indoor air temperature [°C]
    temp_wall: float    # Envelope wall mass temperature [°C]

    def to_array(self) -> np.ndarray:
        return np.array([self.temp_in, self.temp_wall], dtype=float)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ThermalState":
        return cls(temp_in=float(arr[0]), temp_wall=float(arr[1]))


class BuildingThermalModel:
    """
    Continuous and discrete state-space implementation of 2R2C building thermal physics.

    State:
        x = [T_in, T_wall]^T
    Input:
        u = Q_hvac (kW thermal: positive = heating, negative = cooling)
    Disturbances:
        w = [T_out, Q_solar, Q_int]^T
    """

    def __init__(self, params: BuildingParameters, dt_hours: float = 0.5):
        self.params = params
        self.dt = dt_hours

        # Compute continuous-time state-space matrices
        self.A_c, self.B_c, self.E_c = self._build_continuous_matrices()

        # Compute exact discrete-time state-space matrices using ZOH
        self.A_d, self.B_d, self.E_d = self._discretize_matrices(self.dt)

    def _build_continuous_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        c_a = self.params.c_air
        c_w = self.params.c_wall
        r_env = self.params.r_envelope
        r_int = self.params.r_int

        # Continuous state matrix A_c: dx/dt = A_c * x + B_c * u + E_c * w
        # dT_in/dt   = -1/(R_int * C_air) * T_in + 1/(R_int * C_air) * T_wall + Q_hvac/C_air + (Q_sol_in + Q_int)/C_air
        # dT_wall/dt = 1/(R_int * C_wall) * T_in - (1/R_env + 1/R_int)/C_wall * T_wall + T_out/(R_env * C_wall) + Q_sol_w/C_wall
        A_c = np.array([
            [-1.0 / (r_int * c_a), 1.0 / (r_int * c_a)],
            [1.0 / (r_int * c_w), - (1.0 / (r_env * c_w) + 1.0 / (r_int * c_w))]
        ], dtype=float)

        # HVAC thermal input matrix B_c (acts directly on indoor air node)
        B_c = np.array([
            [1.0 / c_a],
            [0.0]
        ], dtype=float)

        # Disturbance matrix E_c for w = [T_out, Q_solar_global, Q_internal]
        # Solar radiation splits between direct window indoor gain (70%) and exterior wall absorption (30%)
        solar_ratio_in = 0.7 * self.params.solar_gain_coeff
        solar_ratio_wall = 0.3 * self.params.solar_gain_coeff

        E_c = np.array([
            [0.0, solar_ratio_in / c_a, 1.0 / c_a],
            [1.0 / (r_env * c_w), solar_ratio_wall / c_w, 0.0]
        ], dtype=float)

        return A_c, B_c, E_c

    def _discretize_matrices(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Exact zero-order hold discretization via matrix exponential block form:
          M = expm([[A_c, B_all], [0, 0]] * dt)
        """
        B_all = np.hstack([self.B_c, self.E_c])  # shape (2, 4)
        n = self.A_c.shape[0]
        m = B_all.shape[1]

        augmented = np.zeros((n + m, n + m), dtype=float)
        augmented[:n, :n] = self.A_c
        augmented[:n, n:] = B_all

        phi = expm(augmented * dt)
        A_d = phi[:n, :n]
        gamma_all = phi[:n, n:]

        B_d = gamma_all[:, :1]
        E_d = gamma_all[:, 1:]

        return A_d, B_d, E_d

    def derivative(self, state: ThermalState, q_hvac: float, t_out: float,
                   q_solar: float = 0.0, q_int: Optional[float] = None) -> np.ndarray:
        """
        Computes continuous time derivative dx/dt = [dT_in/dt, dT_wall/dt]^T.
        """
        if q_int is None:
            q_int = self.params.internal_gain_base

        x = state.to_array()
        u = np.array([q_hvac], dtype=float)
        w = np.array([t_out, q_solar, q_int], dtype=float)

        dxdt = self.A_c @ x + (self.B_c @ u).ravel() + (self.E_c @ w).ravel()
        return dxdt

    def step_rk4(self, state: ThermalState, q_hvac: float, t_out: float,
                 q_solar: float = 0.0, q_int: Optional[float] = None,
                 dt: Optional[float] = None) -> ThermalState:
        """
        Integrates system forward by dt using Runge-Kutta 4th Order.
        Ground truth physical integrator for simulation environment.
        """
        h = dt if dt is not None else self.dt
        if q_int is None:
            q_int = self.params.internal_gain_base

        x0 = state.to_array()

        def f(x_eval: np.ndarray) -> np.ndarray:
            st = ThermalState.from_array(x_eval)
            return self.derivative(st, q_hvac=q_hvac, t_out=t_out, q_solar=q_solar, q_int=q_int)

        k1 = f(x0)
        k2 = f(x0 + 0.5 * h * k1)
        k3 = f(x0 + 0.5 * h * k2)
        k4 = f(x0 + h * k3)

        x_next = x0 + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return ThermalState.from_array(x_next)

    def step_discrete(self, state: ThermalState, q_hvac: float, t_out: float,
                      q_solar: float = 0.0, q_int: Optional[float] = None) -> ThermalState:
        """
        Fast discrete-time step x_{k+1} = A_d * x_k + B_d * u_k + E_d * w_k.
        Used for internal predictions and optimization constraints.
        """
        if q_int is None:
            q_int = self.params.internal_gain_base

        x = state.to_array()
        u = np.array([q_hvac], dtype=float)
        w = np.array([t_out, q_solar, q_int], dtype=float)

        x_next = self.A_d @ x + (self.B_d @ u).ravel() + (self.E_d @ w).ravel()
        return ThermalState.from_array(x_next)

    def compute_steady_state(self, t_out: float, q_hvac: float = 0.0,
                             q_solar: float = 0.0, q_int: Optional[float] = None) -> ThermalState:
        """
        Analytically computes thermal equilibrium state where dx/dt = 0:
          x_ss = -A_c^{-1} (B_c * u + E_c * w)
        """
        if q_int is None:
            q_int = self.params.internal_gain_base

        u = np.array([q_hvac], dtype=float)
        w = np.array([t_out, q_solar, q_int], dtype=float)

        rhs = (self.B_c @ u).ravel() + (self.E_c @ w).ravel()
        x_ss = np.linalg.solve(-self.A_c, rhs)
        return ThermalState.from_array(x_ss)

    def compute_time_constants(self) -> Tuple[float, float]:
        """
        Returns the two physical thermal time constants tau_fast, tau_slow [hours]
        derived from the eigenvalues of continuous A_c.
        """
        eigvals = np.linalg.eigvals(self.A_c)
        taus = -1.0 / np.real(eigvals)
        return float(min(taus)), float(max(taus))
