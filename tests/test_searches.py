"""The launch solution search and the interceptor search, checked the way a user would try them.

The search in sim.autosolve names a speed, elevation, and azimuth for a target
it is handed, and the interceptor search in utils names an angle that chases a
projectile already in flight. Neither is worth much unless it hits what it can
reach and admits it when it cannot, so every case here flies the answer through
the physics module instead of trusting the search's own verdict.

The air is the sheared, northern-hemisphere atmosphere the single-model test
uses, passed explicitly for the reason given there: reading the defaults in
config would be a test of air density and a wind table, and neither claim below
holds in still air. The wall clock is generous throughout, so what the search
answers depends on the problem and not on how fast the machine is.
"""

import time

import numpy as np
import pytest

from physics import BallisticPhysics
from sim.autosolve import solve_launch
from targets import Target, check_collision, closest_approach_between_trajectories
from utils import solve_interceptor_angle

GRAVITY = 9.81
MASS = 10.0
RHO = 0.4
DRAG_COEFFICIENT = 0.47
AREA = 0.01
LATITUDE = 0.3
ALT_LEVELS = [0, 500, 1000, 2000, 3000]
WIND_X_VALS = [4, 6, 8, 10, 12]
WIND_Z_VALS = [0, 4, 8, 12, 16]
WIND_VERTICAL_VALS = [0, 0, 0, 0, 0]
MIN_AUTO_ELEVATION_DEG = 5.0
TARGET_RADIUS_M = 40.0

# The search spends about 2.6 s here running its whole grid, so 600 s is not a
# tuned number. It is set well clear of the runtime so the answer cannot depend
# on the machine, which is what lets the hit assertions below mean anything.
FULL_GRID_WALL_S = 600.0

# At 20 km the target is out of reach of anything this gun can throw in this
# air. The best shot the search finds lands near 14 km and about 6 km short, so
# the floor below leaves a factor of six between a genuine miss and the
# boundary, rather than sitting on top of the observed value.
UNREACHABLE_TARGET_X_M = 20000.0
LARGE_MISS_FLOOR_M = 1000.0

# The interceptor case. The primary leaves at 300 m/s and 35 degrees and the
# slice starts 5 s in, by which point it is near 1150 m downrange and 690 m
# up. From there the two no longer share a launcher, so the miss distance means
# something again. An interceptor at 500 m/s closes to a couple of metres; the
# same geometry at 320 m/s stalls near 70 m, so the ceiling sits between a
# search that converged and one that could not get there.
PRIMARY_SPEED_MPS = 300.0
PRIMARY_ANGLE_DEG = 35.0
INTERCEPTOR_SPEED_MPS = 500.0
INTERCEPT_SLICE_S = 5.0
INTERCEPT_MISS_CEILING_M = 25.0


def launch_snapshot(target_x_launch, target_velocity_x):
    """One search input in the air above, with the clock moved out of the way."""
    return {
        "target_x_launch": target_x_launch,
        "target_velocity_x": target_velocity_x,
        "target_radius": TARGET_RADIUS_M,
        "gravity": GRAVITY,
        "mass": MASS,
        "rho": RHO,
        "drag_coefficient": DRAG_COEFFICIENT,
        "area": AREA,
        "latitude": LATITUDE,
        "alt_levels": ALT_LEVELS,
        "wind_x_vals": WIND_X_VALS,
        "wind_z_vals": WIND_Z_VALS,
        "wind_vertical_vals": WIND_VERTICAL_VALS,
        "min_auto_elevation": MIN_AUTO_ELEVATION_DEG,
        "max_wall_s": FULL_GRID_WALL_S,
    }


def snapshot_physics(snapshot):
    """The one acceleration model, built from the same values the search reads."""
    return BallisticPhysics(
        mass=snapshot["mass"],
        gravity=snapshot["gravity"],
        rho=snapshot["rho"],
        drag_coefficient=snapshot["drag_coefficient"],
        area=snapshot["area"],
        latitude=snapshot["latitude"],
        alt_levels=snapshot["alt_levels"],
        wind_x_vals=snapshot["wind_x_vals"],
        wind_z_vals=snapshot["wind_z_vals"],
        wind_vertical_vals=snapshot["wind_vertical_vals"],
    )


def fly_solution(snapshot, solution):
    """What the physics module makes of the launch solution the search named.

    The search scores its candidates on a fast fixed step, so the distance it
    reports is a ranking aid rather than a promise. This propagates the named
    speed, elevation, and azimuth through the integrator and scores the result
    through the target interface, which is the check the renderer performs
    before it draws the shot.
    """
    xs, ys, zs, t_array, *_ = snapshot_physics(snapshot).trajectory_3d(
        solution["speed"],
        solution["elevation"],
        solution["azimuth"],
        return_time=True,
        t_final=200.0,
    )
    target = Target(
        snapshot["target_x_launch"],
        radius=snapshot["target_radius"],
        vx=snapshot["target_velocity_x"],
    )
    return check_collision(xs, ys, zs, target, t_array=t_array)


def late_primary_trajectory(physics):
    """The primary in flight, from 5 s on, with its clock restarted at zero.

    Both projectiles leave one launcher at the same instant, so scored from
    launch the smallest separation is zero at t 0 whatever angle the search
    picks. Starting the shared clock once the primary is 690 m up and 1150 m
    downrange puts the two in different places at the new zero, which is what
    makes a miss distance worth minimising again.
    """
    xs, ys, zs, t_array, *_ = physics.trajectory_3d(
        PRIMARY_SPEED_MPS, PRIMARY_ANGLE_DEG, return_time=True, max_step=0.05
    )
    start = int(np.searchsorted(t_array, INTERCEPT_SLICE_S))
    return np.column_stack((xs, ys, zs))[start:], t_array[start:] - t_array[start]


def replay_intercept(physics, primary, primary_time, solution):
    """Re-score the interceptor angle the search named, integrated more tightly.

    The search's own distance is the objective it minimised, so it reports the
    best of the 190 grid angles it tried rather than a verified intercept. This
    flies the single angle it settled on again at a tighter step and measures
    the miss distance again, which is the number to hold the answer to.
    """
    xs, ys, zs, t_array, *_ = physics.trajectory_3d(
        INTERCEPTOR_SPEED_MPS,
        solution["angle"],
        return_time=True,
        max_step=0.01,
    )
    interceptor = np.column_stack((xs, ys, zs))
    distance, _when, _index = closest_approach_between_trajectories(
        primary, interceptor, primary_time, t_array
    )
    return distance


@pytest.fixture(scope="module")
def stationary_snapshot():
    return launch_snapshot(800.0, 0.0)


@pytest.fixture(scope="module")
def moving_snapshot():
    return launch_snapshot(1200.0, -10.0)


@pytest.fixture(scope="module")
def unreachable_snapshot():
    return launch_snapshot(UNREACHABLE_TARGET_X_M, 0.0)


@pytest.fixture(scope="module")
def stationary_solution(stationary_snapshot):
    return solve_launch(stationary_snapshot)


@pytest.fixture(scope="module")
def moving_solution(moving_snapshot):
    return solve_launch(moving_snapshot)


@pytest.fixture(scope="module")
def unreachable_solution(unreachable_snapshot):
    return solve_launch(unreachable_snapshot)


def test_the_search_hits_a_stationary_target_it_can_reach(stationary_snapshot, stationary_solution):
    """A target parked at 800 m falls to a 6.5 degree launch the search finds.

    The elevation clears the 5 degree clamp by a degree and a half, so what is
    pinned here is the physics rather than a constraint. The search reports a
    hit inside the 40 m radius, and the integrator agrees when it flies the same
    three numbers, landing about 31 m out.
    """
    assert stationary_solution["ok"]
    assert stationary_solution["hit"]
    assert stationary_solution["distance"] <= TARGET_RADIUS_M

    hit, _hit_index, miss_distance, _closest_index = fly_solution(
        stationary_snapshot, stationary_solution
    )
    assert hit, f"the search's answer misses by {miss_distance:.1f} m under the integrator"


def test_the_search_hits_a_moving_target_it_can_reach(moving_snapshot, moving_solution):
    """A target walking towards the gun at 10 m/s from 1200 m is still reachable.

    The closing speed shortens the flight the search has to cover, and the
    answer comes back near 29 degrees, well off the elevation clamp. Both the
    search's own verdict and the replay report a hit, the second by about a
    metre, so this is the case that would fail if the search were scoring the
    target against the position it launched from.
    """
    assert moving_solution["ok"]
    assert moving_solution["hit"]
    assert moving_solution["distance"] <= TARGET_RADIUS_M

    hit, _hit_index, miss_distance, _closest_index = fly_solution(moving_snapshot, moving_solution)
    assert hit, f"the search's answer misses by {miss_distance:.1f} m under the integrator"


def test_the_search_admits_a_miss_for_a_target_it_cannot_reach(
    unreachable_snapshot, unreachable_solution
):
    """A target at 20 km gets no hit, a distance of kilometres, and no exception.

    Both assertions are needed. A bare distance leaves the caller to guess
    whether the shot was close, and a bare flag leaves out by how much, so
    neither survives on its own. The replay stays thousands of metres short as
    well, which is what shows the miss comes from the target being out of reach
    rather than from the search giving up early.
    """
    assert unreachable_solution["ok"]
    assert not unreachable_solution["hit"]
    assert unreachable_solution["distance"] > LARGE_MISS_FLOOR_M

    hit, _hit_index, miss_distance, _closest_index = fly_solution(
        unreachable_snapshot, unreachable_solution
    )
    assert not hit
    assert miss_distance > LARGE_MISS_FLOOR_M


def test_the_interceptor_search_converges_on_a_primary_it_can_reach():
    """A 500 m/s interceptor closes to a couple of metres on a primary in flight.

    Two assertions do the work here. The replayed miss distance at the angle
    the search settled on says the answer is real rather than the best of 190
    grid samples, and the reported time of the miss distance says the two met
    in flight. Scored from the shared launcher instead, the search returns a
    zero at t 0 for every angle, which would sail under a distance ceiling on
    its own, so the second assertion is what catches that degenerate answer.
    """
    physics = snapshot_physics(launch_snapshot(800.0, 0.0))
    primary, primary_time = late_primary_trajectory(physics)

    result = solve_interceptor_angle(
        primary, primary_time, INTERCEPTOR_SPEED_MPS, trajectory_func=physics.trajectory_3d
    )

    replayed = replay_intercept(physics, primary, primary_time, result)

    assert replayed < INTERCEPT_MISS_CEILING_M, (
        f"the interceptor the search chose passes {replayed:.1f} m from the primary"
    )
    assert result["shared_time"] > 1.0


@pytest.mark.xfail(
    strict=True,
    reason="the wall-clock search budget lets the same input return a different launch "
    "solution once the clock runs out sooner, as it does on a slower machine",
)
def test_the_search_returns_the_same_launch_solution_on_repeated_runs(monkeypatch):
    """The same snapshot solved twice names the same speed, elevation, and azimuth.

    The second run expires the search's budget halfway through instead of at
    the first check. The call the clock is read on is counted rather than the
    seconds it reports, so the cut lands at the same point in the grid on any
    machine: expiring on the first check would only show the search giving up,
    which is a different defect and a different fix.
    """
    snapshot = launch_snapshot(800.0, 0.0)
    real_clock = time.perf_counter

    reads = {"count": 0}

    def counted_clock():
        reads["count"] += 1
        return real_clock()

    monkeypatch.setattr(time, "perf_counter", counted_clock)
    first = solve_launch(snapshot)
    assert first["ok"]
    expiry_read = 1 + reads["count"] // 2

    started = real_clock()
    reads["count"] = 0

    def expiring_clock():
        reads["count"] += 1
        return started if reads["count"] < expiry_read else started + 10000.0

    monkeypatch.setattr(time, "perf_counter", expiring_clock)
    second = solve_launch(snapshot)

    assert second["ok"], "the search gave up under a short budget on an input it solves given time"
    assert second["speed"] == pytest.approx(first["speed"])
    assert second["elevation"] == pytest.approx(first["elevation"])
    assert second["azimuth"] == pytest.approx(first["azimuth"])