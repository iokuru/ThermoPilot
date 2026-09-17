"""
Short-horizon thermal drift forecasting model.

Uses HistGradientBoostingRegressor to predict indoor temperature changes:
  Delta T_in = T_in(t+1) - T_in(t)
conditioned on current thermal state, planned HVAC power, and upcoming weather.

Logs RMSE and MAE against a naive persistence baseline (T_{t+1} = T_t).
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np
import math
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from thermopilot.physics.rc_model import BuildingThermalModel, ThermalState
from thermopilot.forecasting.weather_client import WeatherDataPoint


@dataclass
class ForecastBenchmarkResult:
    model_rmse: float
    model_mae: float
    baseline_rmse: float
    baseline_mae: float
    rmse_reduction_pct: float
    mae_reduction_pct: float

    def summary(self) -> str:
        return (
            f"Forecaster Benchmark:\n"
            f"  Model RMSE:    {self.model_rmse:.4f} °C (vs Baseline {self.baseline_rmse:.4f} °C) -> "
            f"{self.rmse_reduction_pct:+.1f}% improvement\n"
            f"  Model MAE:     {self.model_mae:.4f} °C (vs Baseline {self.baseline_mae:.4f} °C) -> "
            f"{self.mae_reduction_pct:+.1f}% improvement"
        )


class ThermalForecaster:
    """
    Gradient-boosted decision tree model predicting indoor thermal drift.
    """

    def __init__(self, random_state: int = 42):
        self.model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            random_state=random_state
        )
        self.is_trained = False

    @staticmethod
    def extract_features(state: ThermalState, q_hvac: float,
                         weather: WeatherDataPoint) -> np.ndarray:
        """
        Engineers physically meaningful feature vector:
          [T_in, T_wall, T_out, delta_T_ambient, Q_hvac, solar, sin_hour, cos_hour]
        """
        delta_t_amb = weather.temp_out - state.temp_in
        hour = weather.timestamp_hour % 24.0
        sin_hr = math.sin(2.0 * math.pi * hour / 24.0)
        cos_hr = math.cos(2.0 * math.pi * hour / 24.0)

        return np.array([
            state.temp_in,
            state.temp_wall,
            weather.temp_out,
            delta_t_amb,
            q_hvac,
            weather.solar_irradiance,
            sin_hr,
            cos_hr
        ], dtype=float)

    def generate_training_data(self, physics_model: BuildingThermalModel,
                               weather_series: List[WeatherDataPoint],
                               n_samples: int = 2000,
                               random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates diverse training dataset using physics simulator across randomized
        HVAC power states and realistic weather conditions.
        """
        rng = np.random.RandomState(random_seed)
        X_list, y_list = [], []

        weather_len = len(weather_series)
        max_hvac = physics_model.params.max_hvac_thermal_kw

        # Sample trajectories across initial conditions
        for _ in range(n_samples):
            w_idx = rng.randint(0, weather_len - 1)
            weather = weather_series[w_idx]

            # Realistic temperature ranges
            t_in = rng.uniform(18.0, 25.0)
            t_wall = t_in + rng.uniform(-1.5, 1.5)
            state = ThermalState(temp_in=t_in, temp_wall=t_wall)

            # Random HVAC excitation in cooling/heating/off
            mode = rng.choice(["off", "cool", "heat"])
            if mode == "cool":
                q_hvac = -rng.uniform(0.5, max_hvac)
            elif mode == "heat":
                q_hvac = rng.uniform(0.5, max_hvac)
            else:
                q_hvac = 0.0

            q_int = rng.uniform(0.2, 1.0)
            q_solar = weather.solar_irradiance

            # True physics ground truth step
            next_state = physics_model.step_rk4(
                state, q_hvac=q_hvac, t_out=weather.temp_out,
                q_solar=q_solar, q_int=q_int
            )

            feat = self.extract_features(state, q_hvac, weather)
            drift = next_state.temp_in - state.temp_in

            X_list.append(feat)
            y_list.append(drift)

        return np.array(X_list, dtype=float), np.array(y_list, dtype=float)

    def train_and_benchmark(self, physics_model: BuildingThermalModel,
                            weather_series: List[WeatherDataPoint],
                            train_samples: int = 3000,
                            test_samples: int = 800) -> ForecastBenchmarkResult:
        """
        Trains the gradient-boosted forecaster and benchmarks strictly against
        the naive persistence baseline (predicts drift = 0).
        """
        X_train, y_train = self.generate_training_data(
            physics_model, weather_series, n_samples=train_samples, random_seed=42
        )
        X_test, y_test = self.generate_training_data(
            physics_model, weather_series, n_samples=test_samples, random_seed=123
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        # Naive persistence: predicted drift is zero (T_{t+1} == T_t)
        y_naive = np.zeros_like(y_test)

        model_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        model_mae = float(mean_absolute_error(y_test, y_pred))

        baseline_rmse = float(np.sqrt(mean_squared_error(y_test, y_naive)))
        baseline_mae = float(mean_absolute_error(y_test, y_naive))

        rmse_red = ((baseline_rmse - model_rmse) / baseline_rmse) * 100.0
        mae_red = ((baseline_mae - model_mae) / baseline_mae) * 100.0

        return ForecastBenchmarkResult(
            model_rmse=model_rmse,
            model_mae=model_mae,
            baseline_rmse=baseline_rmse,
            baseline_mae=baseline_mae,
            rmse_reduction_pct=rmse_red,
            mae_reduction_pct=mae_red
        )

    def predict_next_temp(self, state: ThermalState, q_hvac: float,
                          weather: WeatherDataPoint) -> float:
        """Predicts indoor temperature at t + dt."""
        if not self.is_trained:
            raise RuntimeError("Forecaster has not been trained yet.")
        feat = self.extract_features(state, q_hvac, weather).reshape(1, -1)
        drift = self.model.predict(feat)[0]
        return state.temp_in + drift
