from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class BuildingParameters:
    """
    Physical parameters for 2R2C (Resistor-Capacitor) building thermal network.

    State variables:
      T_in:  Indoor air temperature [°C]
      T_wall: Structural envelope wall temperature [°C]

    Parameters:
      c_air:       Thermal capacitance of indoor air & light furnishings [kWh/°C]
      c_wall:      Thermal capacitance of building envelope / masonry [kWh/°C]
      r_envelope:  Thermal resistance of exterior wall to ambient [°C/kW]
      r_int:       Thermal resistance between wall mass and indoor air [°C/kW]
      solar_gain_coeff: Effective window solar aperture area [m² * SHGC]
      internal_gain_base: Base internal heat load from appliances/occupants [kW]
      cop_cooling: Coefficient of Performance for HVAC in cooling mode [dimensionless]
      cop_heating: Coefficient of Performance for HVAC in heating mode [dimensionless]
      max_hvac_thermal_kw: Max HVAC thermal power capacity [kW_thermal]
    """
    name: str = "Residential Single-Family"
    c_air: float = 2.5            # kWh/°C (fast thermal response)
    c_wall: float = 15.0          # kWh/°C (slow structural envelope mass)
    r_envelope: float = 4.0       # °C/kW (exterior insulation resistance)
    r_int: float = 1.2            # °C/kW (indoor-wall heat exchange)
    solar_gain_coeff: float = 2.5 # m² effective solar aperture
    internal_gain_base: float = 0.5 # kW base occupants/plug loads
    cop_cooling: float = 3.5      # Typical modern inverter heat pump
    cop_heating: float = 3.2
    max_hvac_thermal_kw: float = 8.0 # kW thermal capacity


@dataclass(frozen=True)
class ComfortBounds:
    """
    Occupant comfort configuration and penalty coefficients.
    """
    temp_min: float = 20.0        # Minimum permissible indoor temp [°C]
    temp_max: float = 23.5        # Maximum permissible indoor temp [°C]
    target_temp: float = 21.75    # Neutral comfort midpoint [°C]
    slack_penalty: float = 50.0   # $/°C-hour penalty for boundary violation
    smoothness_penalty: float = 0.05 # Penalty on HVAC actuation rate of change (u_k - u_{k-1})²


@dataclass
class SimulationConfig:
    """
    Temporal discretization and optimization horizon parameters.
    """
    dt_hours: float = 0.5         # 30-minute discretization steps
    horizon_hours: int = 24       # 24-hour rolling MPC horizon
    simulation_days: int = 7      # Default multi-day evaluation duration

    @property
    def horizon_steps(self) -> int:
        return int(round(self.horizon_hours / self.dt_hours))

    @property
    def total_steps(self) -> int:
        return int(round((self.simulation_days * 24.0) / self.dt_hours))


# Presets for benchmarking across varied thermal dynamics
PRESET_RESIDENTIAL = BuildingParameters(
    name="Single-Family Home",
    c_air=2.5,
    c_wall=15.0,
    r_envelope=4.0,
    r_int=1.2,
    solar_gain_coeff=2.5,
    internal_gain_base=0.6,
    cop_cooling=3.5,
    cop_heating=3.2,
    max_hvac_thermal_kw=8.0,
)

PRESET_COMMERCIAL = BuildingParameters(
    name="Commercial Office",
    c_air=8.0,
    c_wall=35.0,
    r_envelope=2.2,
    r_int=0.8,
    solar_gain_coeff=6.0,
    internal_gain_base=2.5,
    cop_cooling=4.0,
    cop_heating=3.6,
    max_hvac_thermal_kw=25.0,
)

PRESET_HIGH_THERMAL_MASS = BuildingParameters(
    name="Heavy Masonry",
    c_air=3.0,
    c_wall=30.0,
    r_envelope=5.0,
    r_int=1.5,
    solar_gain_coeff=2.0,
    internal_gain_base=0.4,
    cop_cooling=3.5,
    cop_heating=3.2,
    max_hvac_thermal_kw=10.0,
)

BUILDING_PRESETS: Dict[str, BuildingParameters] = {
    "residential": PRESET_RESIDENTIAL,
    "commercial": PRESET_COMMERCIAL,
    "high_thermal_mass": PRESET_HIGH_THERMAL_MASS,
}
