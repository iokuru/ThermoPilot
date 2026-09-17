import pytest
import numpy as np

from thermopilot.pricing.tariffs import (
    ElectricityTariff,
    TOUTariffSchedule,
    TARIFF_DEFAULT_TOU,
    TARIFF_AGGRESSIVE_PEAK
)


def test_tariff_schedule_generation():
    """Verify tariff generator produces correct day/night and on-peak price patterns."""
    schedule_gen = TOUTariffSchedule(TARIFF_DEFAULT_TOU)
    # Generate 48 steps (24 hours at 0.5h step)
    rates = schedule_gen.generate_schedule(total_steps=48, dt_hours=0.5, start_hour_offset=0.0)

    assert len(rates) == 48

    # Check 03:00 (step 6) -> Off peak ($0.12)
    assert rates[6] == pytest.approx(TARIFF_DEFAULT_TOU.off_peak_rate)

    # Check 12:00 (step 24) -> Mid peak ($0.24)
    assert rates[24] == pytest.approx(TARIFF_DEFAULT_TOU.mid_peak_rate)

    # Check 18:00 (step 36) -> On peak ($0.52)
    assert rates[36] == pytest.approx(TARIFF_DEFAULT_TOU.on_peak_rate)


def test_weekend_tariff_off_peak():
    """Verify weekend hours default to off-peak rate."""
    schedule_gen = TOUTariffSchedule(TARIFF_DEFAULT_TOU)
    # Day 5 (Saturday, 120 hours offset)
    saturday_rate = schedule_gen.get_price_at_hour(hour_of_day=18.0, is_weekend=True)
    assert saturday_rate == pytest.approx(TARIFF_DEFAULT_TOU.off_peak_rate)
