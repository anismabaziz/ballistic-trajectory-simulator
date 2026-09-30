"""The launch solution search and the interceptor search, checked the way a user would try them.

The search in sim.autosolve names a speed, elevation, and azimuth for a target
it is handed, and the interceptor search in utils names an angle that chases a
projectile already in flight. Neither is worth much unless it hits what it can
reach and admits it when it cannot, so every case here flies the answer through
the physics module instead of trusting the search's own verdict.

The air is the sheared, northern-hemisphere atmosphere the single-model test
uses, passed explicitly for the reason given there: reading the defaults in
config would be a test of air density and a wind table, and neither claim below
holds in still air. The search budget is generous throughout, so what the search
answers depends on the problem and not on how far it was allowed to look.
"""

import numpy as np
import pytest

from ballistics.physics import BallisticPhysics
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

# The search runs a coarse sweep of 54 flight times by 7 azimuths and then
# refines around the winner over 7 by 7 by 7, so 721 candidates is the most it
# can ever spend. The budget below is set above that ceiling rather than tuned to
# an observed count, which is what lets the hit assertions above mean anything:
# they are claims about the whole grid, and only a budget the grid fits inside
# makes them so.
FULL_GRID_CANDIDATE_BUDGET = 1000

# A budget below the grid's ceiling, so the search is cut short partway through
# the coarse sweep and never reaches the refinement. 200 is a quarter of the
# ceiling, which is enough to have found a plausible winner to refine around and
# not enough to refine it.
TRUNCATED_CANDIDATE_BUDGET = 200

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


def launch_snapshot(target_x_launch, target_velocity_x, candidate_budget=FULL_GRID_CANDIDATE_BUDGET):
    """One search input in the air above, with the candidate budget set wide open."""
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
        "candidate_budget": candidate_budget,
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


def test_the_search_returns_the_same_launch_solution_on_repeated_runs():
    """One snapshot solved three times names the same speed, elevation, and azimuth.

    Nothing about the search reads a clock, so a repeat is the same function call
    on the same inputs and the only way this can fail is for something to reach
    outside the snapshot. Repeated runs are cheap here because the budget counts
    candidate flights rather than seconds, so the cost of a run is set by the
    problem and not by how loaded the machine happens to be.
    """
    snapshot = launch_snapshot(800.0, 0.0)

    answers = [solve_launch(snapshot) for _ in range(3)]

    for answer in answers:
        assert answer["ok"]
        assert answer["speed"] == pytest.approx(answers[0]["speed"])
        assert answer["elevation"] == pytest.approx(answers[0]["elevation"])
        assert answer["azimuth"] == pytest.approx(answers[0]["azimuth"])
        assert answer["candidates_used"] == answers[0]["candidates_used"]


def test_the_search_reports_the_candidates_it_spent():
    """A full-grid solve says what it spent and says the grid was not a compromise.

    The count is the claim that the answer is the best of everything the search
    looked at. Reporting a spent count without the exhaustion flag would leave a
    caller to guess whether it got the whole grid or a prefix of it, and a caller
    forced into that guess will get it wrong in the direction of flattering the
    search.
    """
    solution = solve_launch(launch_snapshot(800.0, 0.0))

    assert 0 < solution["candidates_used"] <= FULL_GRID_CANDIDATE_BUDGET
    assert solution["candidate_budget"] == FULL_GRID_CANDIDATE_BUDGET
    assert not solution["exhausted"]


def test_the_search_spends_no_more_than_the_budget_it_was_given():
    """A budget below the grid's ceiling stops the search and admits it.

    Three things have to hold together here. The search stops on the budget
    rather than running to the end of the grid, it reports that it stopped early
    rather than presenting a truncated answer as the best it found, and it still
    answers, because a user pressing Auto Solve is owed a launch either way.
    """
    solution = solve_launch(
        launch_snapshot(800.0, 0.0, candidate_budget=TRUNCATED_CANDIDATE_BUDGET)
    )

    assert solution["ok"]
    assert solution["candidates_used"] == TRUNCATED_CANDIDATE_BUDGET
    assert solution["candidate_budget"] == TRUNCATED_CANDIDATE_BUDGET
    assert solution["exhausted"]


def test_a_truncated_search_answers_the_same_way_every_time():
    """The point of counting candidates instead of seconds: a short budget is repeatable.

    This is the case that could not be written before. A wall clock cut the grid
    off at a point that moved with the machine, so two runs of the same input
    with the same short budget returned two different launch solutions and there
    was no way to state the budget other than in seconds. A budget counted in
    candidate flights names the same prefix of the grid on every run and on every
    machine, so the truncated answer is reproducible too, not just the full one.
    """
    snapshot = launch_snapshot(800.0, 0.0, candidate_budget=TRUNCATED_CANDIDATE_BUDGET)

    first = solve_launch(snapshot)
    second = solve_launch(snapshot)

    assert second["speed"] == pytest.approx(first["speed"])
    assert second["elevation"] == pytest.approx(first["elevation"])
    assert second["azimuth"] == pytest.approx(first["azimuth"])


def test_a_budget_past_the_grid_changes_nothing():
    """Once the budget covers the whole grid, more of it buys nothing.

    This is what makes the reported budget meaningful. A candidate budget the
    caller has to tune like a timeout is a wall clock wearing a different hat,
    and the whole claim is that there is a budget above which the answer stops
    moving because the search has nothing left to look at. That threshold is why
    `FULL_GRID_CANDIDATE_BUDGET` can be picked by arithmetic instead of measured.
    """
    full = solve_launch(launch_snapshot(800.0, 0.0))
    oversized = solve_launch(launch_snapshot(800.0, 0.0, candidate_budget=2 * FULL_GRID_CANDIDATE_BUDGET))

    assert oversized["exhausted"] is False
    assert oversized["candidates_used"] == full["candidates_used"]
    assert oversized["speed"] == pytest.approx(full["speed"])
    assert oversized["elevation"] == pytest.approx(full["elevation"])
    assert oversized["azimuth"] == pytest.approx(full["azimuth"])


def test_a_budget_that_exactly_fits_the_grid_is_not_a_truncated_search():
    """Hitting the budget on the last candidate is not the same as running out of it.

    The budget is read off the full solve rather than written down, so this holds
    whatever the grid costs. The distinction matters because it is the boundary
    the flag has to get right: a search allowed exactly the candidates it wanted
    did see the whole grid, and reporting it as truncated would tell the renderer
    to caveat an answer that needs no caveat. The returned solution is identical
    either way, which is why the assertion is on the flag and not the numbers.
    """
    exact_count = solve_launch(launch_snapshot(800.0, 0.0))["candidates_used"]

    exact = solve_launch(launch_snapshot(800.0, 0.0, candidate_budget=exact_count))

    assert exact["candidates_used"] == exact_count
    assert exact["exhausted"] is False