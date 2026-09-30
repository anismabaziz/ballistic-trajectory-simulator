"""How far apart the project's two integrators land, and how that scales.

The solver integrates with adaptive RK45 and the real-time renderer steps with
a fixed step. Both call `compute_acceleration`, so the only thing separating
their answers is the step rule, which is exactly what the numbers below isolate.

Three claims. The rule the renderer shipped with, semi-implicit Euler, is first
order, so halving the frame step halves the landing error rather than squaring
it. The fourth-order rule costs four acceleration evaluations per step and lands
within a hundredth of a millimetre of the adaptive solver at one step per frame,
which is what makes it the right rule to render with. And at that frame step the
rule that shipped lands inside the renderer's target radius but nowhere near the
solver's answer, which is the gap the project reports rather than hides.
"""

import numpy as np

from ballistics.physics import BallisticPhysics

# The launch the study and the figure both use: the renderer's defaults.
LAUNCH_SPEED = 300.0
LAUNCH_ELEVATION_DEG = 35.0
FRAME_STEP = 1.0 / 60.0

# The renderer's default target radius. The Euler error below is about a quarter
# of it, which is the whole point: the two integrators agree well enough for the
# sandbox to look right and badly enough for the gap to be worth publishing. The
# bound is half the radius so an integrator change shows up here rather than in
# a screenshot.
TARGET_RADIUS = 20.0
LANDING_TOLERANCE_M = 0.5 * TARGET_RADIUS


def launch_state():
    physics = BallisticPhysics()
    angle = np.radians(LAUNCH_ELEVATION_DEG)
    return physics, [0.0, 0.0, 0.0, LAUNCH_SPEED * np.cos(angle), LAUNCH_SPEED * np.sin(angle), 0.0]


def landing_error(physics, state, step, rule):
    """Distance between where the fixed-step rule lands and where RK45 lands.

    Both are compared at the flight time the adaptive integration reports, so
    this is the distance between the two answers rather than the distance
    between two slightly different impact points.
    """
    xs, ys, zs, t_arr, _range, flight_time, _ = physics.trajectory_3d(
        LAUNCH_SPEED, LAUNCH_ELEVATION_DEG, return_time=True
    )
    stepped = physics.integrate_fixed_step(state, step, flight_time, rule=rule)
    return float(np.linalg.norm(np.asarray(stepped[:3]) - np.asarray([xs[-1], ys[-1], zs[-1]])))


def test_euler_landing_error_halves_with_the_step():
    physics, state = launch_state()

    coarse = landing_error(physics, state, FRAME_STEP * 2.0, "euler")
    fine = landing_error(physics, state, FRAME_STEP, "euler")

    # First order means the ratio sits at 2. It lands a little above, because
    # the second-order term is not negligible yet at these step sizes.
    assert 1.8 < coarse / fine < 2.4


def test_euler_landing_error_at_sixty_fps_is_within_the_target_radius():
    physics, state = launch_state()

    error = landing_error(physics, state, FRAME_STEP, "euler")

    assert error < LANDING_TOLERANCE_M


def test_rk4_at_sixty_fps_lands_more_than_a_hundred_times_closer_than_euler():
    physics, state = launch_state()

    rk4_error = landing_error(physics, state, FRAME_STEP, "rk4")
    euler_error = landing_error(physics, state, FRAME_STEP, "euler")

    # 6.2e-6 m measured against 5.1 m. The absolute bound leaves room for the
    # atmosphere table changing; the ratio is the claim the decision rests on.
    assert rk4_error < 0.01
    assert rk4_error < 0.01 * euler_error


def test_substepping_euler_reaches_a_metre_of_the_adaptive_solution():
    physics, state = launch_state()

    error = landing_error(physics, state, FRAME_STEP / 16.0, "euler")

    assert error < 1.0
