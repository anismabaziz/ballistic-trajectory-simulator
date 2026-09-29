"""What the miss distance metric actually means.

`check_collision` and `closest_approach_between_trajectories` are the two
functions every launch solution gets scored by, and a launch solution is only as
trustworthy as its score. These tests pin the two claims the glossary in
CONTEXT.md makes about them. A miss distance is a distance between a trajectory
and a target at one shared instant, not a distance between two paths treated as
geometry. And a hit is the first time the projectile is inside the target, not
the closest it ever comes.

The shots are flown through the physics module rather than fabricated as point
arrays, so each case is a trajectory the integrator actually produced. Every
atmospheric value is passed explicitly, for the reason given in `test_physics.py`:
the defaults in config carry a wind table, and a wind table would move the
numbers asserted here.
"""

import numpy as np
import pytest

from physics import BallisticPhysics
from targets import Target, check_collision, closest_approach_between_trajectories

GRAVITY = 9.81

# The step size the whole file flies at. Tight enough that the geometry below is
# sampled rather than stepped over, and pinned rather than left to a default
# because two of these tests compare distances of a few metres.
MAX_STEP = 0.01

# The 45 degree shot used for the centre-crossing and the unreachable target
# lands at v^2/g = 1019 m and is in the air for about 14 s.
STANDARD_SPEED_MPS = 100.0
STANDARD_ANGLE_DEG = 45.0
STANDARD_FLIGHT_T = 200.0

# Against a distance of hundreds of metres this is zero, and it is eight orders
# looser than the integrator's own 1e-12 error on position.
COINCIDENT_TOLERANCE_M = 1e-6

# The target parked at its own launch point is 115 m from the trajectory at its
# closest, which is the whole altitude the moving target climbs through in the
# seconds before the crossing. Asserted as a floor, well under the 255 m, to
# catch a metric that quietly compared against the launch position.
PARKED_MISS_FLOOR_M = 100.0

# The 45 degree shot lands at v^2 sin(2*45)/g = 1019.37 m. Sitting 500 m past
# that leaves the final sample as the closest the trajectory ever gets, so the
# miss distance is the overshoot exactly.
OVERSHOOT_M = 500.0

# The 75 degree shot climbs through 255 m on the way up and comes back down
# through it, so a target flying level at 255 m sees two crossings. These bound
# each one, and the two are well separated in time, which is what makes the
# reported hit index unambiguous.
ASCENDING_CROSSING_S = (3.0, 3.3)
DESCENDING_CROSSING_S = (16.4, 16.7)

# Inside the 10 m radius the two passes miss by 7.2 m and 4.6 m respectively.
# The descending pass is the closer one, so a metric that reported the closest
# approach as the hit would name the wrong end of the flight.
ASCENDING_MISS_M = 7.2
DESCENDING_MISS_M = 4.6
CROSSING_TOLERANCE_M = 0.5


def vacuum(angle_deg=STANDARD_ANGLE_DEG, t_final=STANDARD_FLIGHT_T, speed=STANDARD_SPEED_MPS):
    """A vacuum shot, integrated tightly enough to trust to a few parts in 1e12.

    Drag, wind, and the Earth's rotation are all off, so the trajectory is the
    parabola the closed forms in `test_physics.py` describe and nothing else.
    The extra digits earn their place here in a way they do not there: these
    tests compare a distance of a few metres against a target's own position, and
    a metre is small enough that a loose integration would show.
    """
    physics = BallisticPhysics(
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
    xs, ys, zs, t_array, ground_range, flight_time, peak = physics.trajectory_3d(
        speed,
        angle_deg,
        return_time=True,
        max_step=MAX_STEP,
        t_final=t_final,
    )
    return xs, ys, zs, t_array, ground_range, flight_time, peak


def distances_to(xs, ys, zs, target, t_array):
    """How far the target is from each trajectory sample, at that sample's own time.

    `check_collision` reduces exactly this to a minimum and an argmin. The tests
    below use it to say which samples are inside the target and when, which is
    what lets them claim something about the reported indices beyond the fact
    that two integers came back.
    """
    trajectory = np.column_stack((xs, ys, zs))
    return np.linalg.norm(trajectory - target.positions_over_time(t_array), axis=1)


def within(seconds, moment):
    return seconds[0] <= moment <= seconds[1]


def test_a_moving_target_is_met_at_its_own_time_not_at_its_launch_point():
    """A trajectory through a moving target's centre reports no miss.

    One target, one trajectory, two velocities. Moving, it is arranged to sit
    exactly on the projectile at the apex of the shot, and the metric reports a
    hit with no miss distance. Parked at the position it launched from, the same
    target misses by 115 m.

    That contrast is the claim. A miss distance is a distance at a shared
    instant, and the only thing separating the two answers is whether the target
    was where it actually was at the time being scored. A metric that compared
    the trajectory against a target's launch position would report 115 m for the
    moving case too, and every interceptor solution in the project would be
    scored against a target that had not moved yet.

    The radius is 0.05 m, and exactly one sample of the trajectory falls inside
    it. A looser radius would swallow the neighbouring samples and turn the zero
    into an average of several near misses rather than the coincidence it is.
    """
    xs, ys, zs, t_array, _range, _flight_time, _peak = vacuum()

    apex_index = int(np.argmax(ys))
    apex = np.array([xs[apex_index], ys[apex_index], zs[apex_index]])
    velocity = np.array([30.0, -10.0, 5.0])
    launch_position = apex - velocity * t_array[apex_index]

    moving = Target(
        launch_position[0],
        y=launch_position[1],
        z=launch_position[2],
        radius=0.05,
        vx=velocity[0],
        vy=velocity[1],
        vz=velocity[2],
    )
    parked = Target(
        launch_position[0], y=launch_position[1], z=launch_position[2], radius=0.05
    )

    hit, hit_index, miss_distance, closest_index = check_collision(
        xs, ys, zs, moving, t_array=t_array
    )
    inside = distances_to(xs, ys, zs, moving, t_array) <= moving.radius
    _parked_hit, _parked_index, parked_miss, _parked_closest = check_collision(
        xs, ys, zs, parked, t_array=t_array
    )

    assert hit
    assert miss_distance == pytest.approx(0.0, abs=COINCIDENT_TOLERANCE_M)
    assert hit_index == apex_index
    assert closest_index == apex_index
    assert inside.sum() == 1, "the coincidence is one instant long, not a stretch of trajectory"
    assert parked_miss > PARKED_MISS_FLOOR_M


def test_a_target_the_shot_cannot_reach_is_reported_as_a_miss():
    """A target beyond the shot's reach gets no hit, and a distance that says by how much.

    The target sits 500 m past where the 45 degree shot lands, on the ground, so
    the trajectory's final sample is the closest it ever comes and the miss
    distance is the overshoot to the last decimal.

    The pair of assertions is the point. A distance with no verdict, or a verdict
    with no distance, passes either one alone, and a caller cannot tell from a
    bare number whether the shot was close or never in the neighbourhood.
    """
    xs, ys, zs, t_array, ground_range, _flight_time, _peak = vacuum()

    beyond_reach = Target(ground_range + OVERSHOOT_M, radius=10.0)
    hit, hit_index, miss_distance, _closest_index = check_collision(
        xs, ys, zs, beyond_reach, t_array=t_array
    )

    assert not hit
    assert hit_index is None
    assert miss_distance == pytest.approx(OVERSHOOT_M, abs=1e-6)


def test_hit_index_is_the_earliest_hit_not_the_closest_approach():
    """A target inside the radius twice is hit on the first pass, not the nearest one.

    A 75 degree shot crosses 255 m twice, going up at t = 3.1 s and coming down
    at t = 16.5 s, so a target flying level at 255 m sees the projectile cross
    its altitude twice. Both passes are inside the 10 m radius and they are not
    equally close: 7.2 m on the way up against 4.6 m on the way down.

    That ordering is the whole test. The renderer draws its hit marker at
    `hit_index`, so a metric reporting the closest approach as the hit would
    still draw a hit here, thirteen seconds and two hundred metres downrange of
    where the target was actually reached.

    Both windows are located from the sample distances rather than taken on
    trust, so a change in the integrator that moved or merged a crossing would
    fail the two-window assertion here instead of quietly making the rest of the
    test vacuous.
    """
    xs, ys, zs, t_array, _range, _flight_time, _peak = vacuum(angle_deg=75.0, t_final=400.0)

    crossing_target = Target(10.0, y=255.0, radius=10.0, vx=25.0)
    distances = distances_to(xs, ys, zs, crossing_target, t_array)
    inside = np.flatnonzero(distances <= crossing_target.radius)
    windows = np.split(inside, np.flatnonzero(np.diff(inside) > 1) + 1)

    hit, hit_index, _miss_distance, closest_index = check_collision(
        xs, ys, zs, crossing_target, t_array=t_array
    )

    assert len(windows) == 2, "the shot is meant to cross the target's altitude twice"
    ascending, descending = windows
    assert within(ASCENDING_CROSSING_S, t_array[ascending[0]])
    assert within(DESCENDING_CROSSING_S, t_array[descending[0]])
    assert distances[ascending].min() == pytest.approx(
        ASCENDING_MISS_M, abs=CROSSING_TOLERANCE_M
    )
    assert distances[descending].min() == pytest.approx(
        DESCENDING_MISS_M, abs=CROSSING_TOLERANCE_M
    )

    assert hit
    assert within(ASCENDING_CROSSING_S, t_array[hit_index]), "the hit is the ascending pass"
    assert within(DESCENDING_CROSSING_S, t_array[closest_index]), "the nearest pass is the descending one"
    assert t_array[hit_index] < t_array[closest_index]


def test_a_moving_targets_position_over_time_is_its_launch_point_plus_velocity_times_time():
    """The formula every miss distance rests on, checked in all three axes.

    `check_collision` scores every trajectory sample against
    `positions_over_time`, so this is the function the whole metric is built on.
    The velocity is deliberately awkward, 2, -0.5, and 0.25 m/s, because a target
    that only moved downrange would let a sign error or an axis swap in the
    lateral components go unnoticed for the life of the project.

    The two accessors are asserted against each other as well as against the
    formula, since a renderer asking for one instant and the collision check
    asking for an array have to agree, and nothing else in the suite would notice
    if they did not.
    """
    target = Target(12.0, y=-3.0, z=7.0, radius=1.0, vx=2.0, vy=-0.5, vz=0.25)
    times = np.array([0.0, 0.5, 3.25, 17.0])

    positions = np.array([target.position_at(t) for t in times])

    assert target.positions_over_time(times) == pytest.approx(positions)
    assert positions[:, 0] == pytest.approx(12.0 + 2.0 * times)
    assert positions[:, 1] == pytest.approx(-3.0 - 0.5 * times)
    assert positions[:, 2] == pytest.approx(7.0 + 0.25 * times)
    assert target.position_at(0.0) == pytest.approx((12.0, -3.0, 7.0))


def test_closest_approach_lands_on_the_true_minimum_not_the_nearest_grid_point():
    """Two trajectories crossing in flight have a closest approach the resampling grid cannot move.

    The metric puts both trajectories on one shared time base by resampling onto
    a fixed 2000-point grid, and a minimum taken over a grid returns the smallest
    value the grid happens to contain rather than the smallest value between the
    samples. For a sharp crossing that error is real and it scales with the grid
    spacing, so the number the interceptor search converges on would be an
    artifact of a constant in `targets.py` if nothing held it in place.

    The case is built so the right answer is available by hand. Both projectiles
    fly straight at constant velocity, so their separation is linear in time and
    the closest approach is the foot of the perpendicular from the launch offset
    to the line of relative motion: 304.105 m at t = 3.867 s. The crossing is
    deliberately not near a grid point. Over the same 20 s, a 20-point grid
    answers 307.585 m and a 200-point grid 304.172 m, so the 1 cm tolerance below
    is 300 times tighter than a grid coarse enough to matter and passes only
    because the real one is 2000.

    The time assertion is the same claim on the other axis. The answer is within
    half a grid step of the true minimum, 5.0 ms at 2000 points over 20 s, which
    is as close as a grid minimum can get and is the property being pinned: the
    reported instant is the crossing, not a neighbour of it.

    Callers hand the trajectories in at five densities from 401 points to
    200001, and every one returns the same distance to within a micrometre. The
    internal grid, not the caller's sampling, is what sets the answer, and this
    is the assertion that says so.
    """
    first_velocity = np.array([120.0, 0.0, 0.0])
    second_velocity = np.array([0.0, 60.0, 0.0])
    second_launch = np.array([600.0, 40.0, 0.0])

    relative_velocity = second_velocity - first_velocity
    closest_time = -(second_launch @ relative_velocity) / (relative_velocity @ relative_velocity)
    closest_distance = np.linalg.norm(second_launch + relative_velocity * closest_time)

    def straight_line(launch, velocity, times):
        return launch[None, :] + np.outer(times, velocity)

    distances = []
    times_of_closest_approach = []
    for sample_count in (401, 2001, 10001, 40001, 200001):
        times = np.linspace(0.0, 20.0, sample_count)
        first = straight_line(np.zeros(3), first_velocity, times)
        second = straight_line(second_launch, second_velocity, times)
        distance, when, _index = closest_approach_between_trajectories(first, second, times, times)
        distances.append(distance)
        times_of_closest_approach.append(when)

    assert distances == pytest.approx([closest_distance] * len(distances), abs=1e-2)
    assert times_of_closest_approach == pytest.approx(
        [closest_time] * len(times_of_closest_approach), abs=0.5 * 20.0 / 1999
    )
