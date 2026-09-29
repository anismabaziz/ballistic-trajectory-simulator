import numpy as np
import pytest

from physics import BallisticPhysics

GRAVITY = 9.81
LAUNCH_SPEED = 100.0
MAX_STEP = 0.05
ANGLES = [30.0, 45.0, 60.0]

# Far looser than what the integrator currently delivers, which is around 5e-13
# on range and 5e-14 on time. solve_ivp takes an adaptive step capped at
# MAX_STEP, so the error is a function of the integrator's internals rather than
# a fixed quantity, and it will move between scipy releases. These bounds are
# set loose enough that such a move will not turn this test red, and tight
# enough that a genuine sign or axis error still fails by orders of magnitude.
RANGE_TOLERANCE = 1e-3
TIME_TOLERANCE = 1e-6


def vacuum_physics():
    """Ballistics with drag, wind, and the Earth's rotation all switched off.

    Every value is passed explicitly. Falling back on the defaults in config
    would drag in air density, a wind table, and a latitude, and the closed-form
    comparison below only means anything in a vacuum.
    """
    return BallisticPhysics(
        mass=1.0,
        gravity=GRAVITY,
        rho=0.0,
        drag_coefficient=0.0,
        area=0.0,
        latitude=0.0,
        alt_levels=[0.0],
        wind_x_vals=[0.0],
        wind_z_vals=[0.0],
        wind_vertical_vals=[0.0],
    )


def closed_form_range(angle_deg):
    return LAUNCH_SPEED**2 * np.sin(2.0 * np.radians(angle_deg)) / GRAVITY


def closed_form_flight_time(angle_deg):
    return 2.0 * LAUNCH_SPEED * np.sin(np.radians(angle_deg)) / GRAVITY


@pytest.mark.parametrize("angle_deg", ANGLES)
def test_vacuum_range_matches_closed_form(angle_deg):
    _xs, _ys, _zs, ground_range, _flight_time, _peak = vacuum_physics().trajectory_3d(
        LAUNCH_SPEED, angle_deg, max_step=MAX_STEP, t_final=200.0
    )

    assert ground_range == pytest.approx(closed_form_range(angle_deg), abs=RANGE_TOLERANCE)


@pytest.mark.parametrize("angle_deg", ANGLES)
def test_vacuum_flight_time_matches_closed_form(angle_deg):
    _xs, _ys, _zs, _ground_range, flight_time, _peak = vacuum_physics().trajectory_3d(
        LAUNCH_SPEED, angle_deg, max_step=MAX_STEP, t_final=200.0
    )

    assert flight_time == pytest.approx(closed_form_flight_time(angle_deg), abs=TIME_TOLERANCE)


@pytest.mark.parametrize("azimuth_deg", [30.0, 60.0, 90.0])
def test_vacuum_horizontal_range_is_azimuth_independent(azimuth_deg):
    xs, _ys, zs, _ground_range, _flight_time, _peak = vacuum_physics().trajectory_3d(
        LAUNCH_SPEED, 45.0, azimuth_deg=azimuth_deg, max_step=MAX_STEP, t_final=200.0
    )

    horizontal_range = np.hypot(xs[-1], zs[-1])

    assert horizontal_range == pytest.approx(closed_form_range(45.0), abs=RANGE_TOLERANCE)
