# ThermoPilot — Interview Notes & Resume Reference

This document contains private portfolio notes, quantified resume bullets, and technical interview defense prep.

---

## Quantified Resume Bullets

```text
ThermoPilot — Predictive HVAC Optimization (Personal Project)
• Built a Model Predictive Control (MPC) system for HVAC scheduling, formulating
  energy cost minimization as a constrained quadratic program (cvxpy) solved
  on a receding horizon, reducing simulated electricity costs by 25.2% vs. a rule-based
  thermostat baseline while maintaining 0.00°C·hr comfort violations
• Trained a gradient-boosted thermal forecasting model on weather API data (Open-Meteo)
  to predict short-horizon building thermal drift, achieving 0.06°C RMSE (92.1% error
  reduction against a naive persistence baseline)
• Designed and validated a 2R2C (resistor-capacitor) thermal simulation of building
  dynamics as the evaluation environment, enabling reproducible before/after
  comparison across multiple building profiles and dynamic time-of-use pricing scenarios
• Incorporated dynamic time-of-use electricity pricing into the control objective,
  demonstrating automated pre-cooling behavior ahead of peak tariff spikes ($0.52/kWh)
  and shedding 92.3% of on-peak electrical demand
```

---

## Technical Interview Defense (FAQ)

### Q1: "Walk me through your objective function and constraints."
**Key response points:**
- **Convex QP formulation**: The objective minimizes cumulative operational electricity cost subject to comfort boundaries, soft slack penalties, and compressor wear.
- **Cost term**: $c_k \cdot P_{elec, k} \cdot \Delta t$, where $P_{elec} = \frac{u_{heat}}{\text{COP}_h} + \frac{u_{cool}}{\text{COP}_c}$.
- **Slack variables**: Soft comfort boundaries $T_{min} - s^{low}_k \le T_{in, k} \le T_{max} + s^{high}_k$ with exact $L_1$ penalty guarantee 100% solver feasibility even under severe outdoor heatwaves.
- **Slew rate penalty**: $\rho_u (u_k - u_{k-1})^2$ avoids compressor short-cycling and rapid mechanical wear.
- **DCP & speed**: Solves in sub-20ms via OSQP.

### Q2: "How did you validate that your savings are real and not just overfitting to your simulation?"
**Key response points:**
- **Physics validation**: Evaluated the 2R2C model against Hurwitz stability ($\text{Re}(\lambda) < 0$), verified passive cooling time constants ($\tau_{air} \approx 1.5\text{h}$, $\tau_{wall} \approx 25\text{h}$), and matched discrete matrix exponential transitions against continuous RK4 integration.
- **Strictly identical test conditions**: Benchmarked against a realistic dual-setpoint deadband thermostat under identical weather, solar, internal load, and tariff schedules.
- **Stochastic noise**: Injected process noise and plant-model mismatch during closed-loop testing to verify receding-horizon feedback compensation.

### Q3: "Why MPC over a simpler control policy?"
**Key response points:**
- Simple reactive controllers (deadband thermostats, PID) only react *after* setpoints are breached.
- Under Time-of-Use rates, afternoon peak heat triggers maximum reactive cooling right during peak electricity pricing (\$0.52/kWh).
- MPC is *anticipatory*: it treats the building's 15 kWh/°C envelope thermal mass as a thermal battery, pre-cooling during cheap hours (\$0.24/kWh) and curtailing load during the expensive peak window.
