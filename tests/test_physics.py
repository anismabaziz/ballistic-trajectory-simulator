"""The physics model, checked against things that can be derived by hand.

Two claims, in order. The integrator reproduces a closed-form vacuum trajectory,
which the vacuum tests below cover. And the atmosphere it integrates through does
what the model says it does: drag acts on velocity relative to the air, the wind
table is interpolated between its levels, and the Coriolis term bends a shot the
right way and the other way across the equator.

Every construction passes its values explicitly. Reading the defaults in config
would be a test of the defaults, which are air density and a wind table, and
neither claim below holds in still air.
"""

import numpy as np
import pytest

from physics import BallisticPhysics

GRAVITY = 9.81
LAUNCH_SPEED = 100.0
MAX_STEP = 0.05
ANGLES = [30.0, 45.0, 60.0]

MASS = 10.0
RHO = 1.225
DRAG_COEFFICIENT = 0.47
AREA = 0.01
LAUNCH_ANGLE = 45.0

# Altitudes the wind tests sample, all inside the table's range so the
# interpolation is exercised rather than a table lookup landing on a row.
ALT_LEVELS = [0.0, 500.0, 1000.0, 2000.0, 3000.0]
NO_WIND = [0.0, 0.0, 0.0, 0.0, 0.0]

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


def atmosphere(latitude=0.0, alt_levels=ALT_LEVELS, wind_x_vals=NO_WIND):
    """A drag-bearing atmosphere built from the constants above, not from config.

    The other two wind arrays are built to the length of the altitude table,
    because the constructor does not pad them: a wind array shorter than the
    altitude table fails inside `np.interp` rather than quietly.
    """
    return BallisticPhysics(
        mass=MASS,
        gravity=GRAVITY,
        rho=RHO,
        drag_coefficient=DRAG_COEFFICIENT,
        area=AREA,
        latitude=latitude,
        alt_levels=alt_levels,
        wind_x_vals=wind_x_vals,
        wind_z_vals=[0.0] * len(alt_levels),
        wind_vertical_vals=[0.0] * len(alt_levels),
    )


def shoot(physics, azimuth_deg=0.0):
    """The standard shot, at whatever the atmosphere does to it.

    100 m/s at 45 degrees, integrated tightly enough that the range is good to
    about 1e-12. Azimuth 0 is down the x axis, which is where the wind tests
    measure range, and azimuth 90 is due north up the z axis, which is where the
    Coriolis tests measure drift.
    """
    return physics.trajectory_3d(
        LAUNCH_SPEED, LAUNCH_ANGLE, azimuth_deg=azimuth_deg, max_step=MAX_STEP, t_final=200.0
    )


def lateral_drift(physics):
    """How far east of the meridian it left from, a due-north shot lands, in m.

    The launch has no x velocity of its own, so anything that shows up in x on
    landing was put there by the Coriolis term.
    """
    xs, _ys, _zs, _range, _flight_time, _peak = shoot(physics, azimuth_deg=90.0)
    return xs[-1]


def wind_speed_at_rest(physics, altitude):
    """The wind speed a ground-frame-stationary projectile infers, in m/s.

    A still projectile's relative airspeed is the wind speed and nothing else, so
    the drag acceleration it picks up is 0.5 rho Cd A w^2 / m pointing downwind.
    That is the only way to read the wind table through this interface, and it
    reads w rather than w^2.
    """
    ax, _ay, _az = physics.compute_acceleration(altitude, 0.0, 0.0, 0.0)
    return np.sqrt(2.0 * MASS * ax / (RHO * DRAG_COEFFICIENT * AREA))


def test_zero_relative_airspeed_gives_no_drag():
    """A projectile flying with the air feels nothing but gravity.

    The relative airspeed is what drag is built from, so it reaching zero is the
    one case where the drag term drops out entirely. This guards the branch at
    `physics.py:55`.

    The wind table is sheared, 8 m/s at the ground to 16 m/s at 3 km, and the
    velocity checked against it is the wind at that altitude rather than one
    number for all three. A model that read the wind at the launch row and reused
    it for the whole flight     would look up 8 m/s everywhere, and at 3 km this would be an 8 m/s relative
    airspeed of real drag rather than none.

    """
    physics = atmosphere(wind_x_vals=[8.0, 10.0, 12.0, 14.0, 16.0])

    for altitude, wind_at_altitude in ((0.0, 8.0), (750.0, 11.0), (3000.0, 16.0)):
        assert physics.compute_acceleration(altitude, wind_at_altitude, 0.0, 0.0) == (
            0.0,
            -GRAVITY,
            0.0,
        )


def test_drag_acts_on_relative_not_ground_velocity():
    """A projectile hanging still in the ground frame is still pushed by the wind.

    This is the bug the wind table exists to make visible. A model that built
    drag from ground velocity sees a stationary projectile and applies no drag at
    all, which quietly under-draws every shot flown through moving air. The
    magnitude below is the textbook 0.5 rho Cd A v^2 / m, with v the wind speed,
    not the projectile's speed over the ground, which is zero here.
    """
    wind_speed = 10.0
    still_in_wind = atmosphere(wind_x_vals=[wind_speed] + NO_WIND[1:])
    still_in_calm = atmosphere()

    ax, ay, az = still_in_wind.compute_acceleration(0.0, 0.0, 0.0, 0.0)
    expected = 0.5 * RHO * DRAG_COEFFICIENT * AREA * wind_speed**2 / MASS

    assert (ax, ay, az) == (pytest.approx(expected), -GRAVITY, 0.0)
    assert still_in_calm.compute_acceleration(0.0, 0.0, 0.0, 0.0) == (0.0, -GRAVITY, 0.0)


def test_wind_table_interpolates_linearly():
    """Between two rows of the wind table, the wind is the midpoint of the two.

    The wind is read back through a still projectile rather than off the arrays
    themselves, which is roundabout but is the only readout the interface has,
    and reading the arrays would only be a second assertion about `np.interp`.

    The levels are spaced so the answer is checkable by eye. 500 m is halfway
    between the 0 m and 1000 m rows, so the wind there is the average of 0 and
    10, which is also the value a linear interpolation has to give. 1500 m is
    halfway between 1000 m and 2000 m, where the average of 10 and 30 is 20, and
    the two halves of the table climb at different rates, 0.01 m/s per m against
    0.02, so a nearest-row lookup or a single constant gradient would miss one of
    the two.
    """
    physics = atmosphere(alt_levels=[0.0, 1000.0, 2000.0], wind_x_vals=[0.0, 10.0, 30.0])

    for altitude, expected_wind in ((500.0, 5.0), (1500.0, 20.0)):
        assert wind_speed_at_rest(physics, altitude) == pytest.approx(expected_wind, rel=1e-9)


def test_wind_shear_extends_downrange_range():
    """Wind that blows downrange carries the projectile further downrange.

    The wind table here is sheared, rising from 5 m/s at the ground to 40 m/s at
    3 km, so most of the benefit arrives in the stretched part of the flight
    where the projectile is slowest and the wind is strongest. The shot lands at
    835 m in still air and 856 m in the shear, a gap of 21 m: thirteen orders of
    magnitude above the integrator's own error, and a long way from a sign that
    could have gone the other way.
    """
    calm = atmosphere()
    tailwind = atmosphere(wind_x_vals=[5.0, 10.0, 20.0, 30.0, 40.0])

    _calm_x, _calm_y, _calm_z, calm_range, _calm_t, _calm_h = shoot(calm)
    _x, _y, _z, sheared_range, _t, _h = shoot(tailwind)

    assert sheared_range > calm_range + 10.0


def test_coriolis_deflects_right_in_northern_hemisphere():
    """A shot fired due north at 45 N lands east of the line it left on.

    The model reads its horizontal axes as x east and z north, which is what puts
    the Coriolis term on the correct side: for a northward velocity the eastward
    acceleration is +2 omega sin(latitude) v, which is positive in the northern
    hemisphere and to the right of somebody walking north. Nothing in the
    codebase names those axes, so the claim is written out here rather than left
    implied: a term that put the same deflection on z would pass every other test
    in this file and put every long shot in the wrong hemisphere.

    The constructor takes latitude in radians, so the test does the conversion.
    Handing it 45.0 raw is a real latitude, seven and some radians past the pole,
    and sin() would not complain.

    The drift itself is small, and that is the claim worth making. The estimate
    2 omega sin(45) v T^2 / 2 puts it near a metre for this shot and the
    trajectory delivers 0.56 m of it, the difference being the drag that the
    estimate ignores. The band below is loose around that on purpose: it pins the
    sign and the order of magnitude, not a fitted number.
    """
    physics = atmosphere(latitude=np.radians(45.0))

    xs, _ys, _zs, _range, _flight_time, _peak = shoot(physics, azimuth_deg=90.0)

    assert 0.2 < xs[-1] < 1.5


def test_coriolis_reverses_across_the_equator():
    """The same due-north shot bends the other way at 45 S, and not at all on it.

    One hemisphere on its own would pass against a term that had lost its sign on
    the way out of the formula. Two at once pin it: the southern drift is the
    northern one mirrored, to the integrator's own precision. On the equator,
    where sin(latitude) is zero, the lateral drift comes out at 5e-14 m against
    the 0.56 m either side of it.
    """
    north = atmosphere(latitude=np.radians(45.0))
    south = atmosphere(latitude=np.radians(-45.0))
    equator = atmosphere(latitude=0.0)

    northward = lateral_drift(north)
    southward = lateral_drift(south)

    assert northward == pytest.approx(-southward, abs=1e-9)
    assert abs(lateral_drift(equator)) < 1e-9
