"""The launch solution search and the physics module must share one acceleration model.

The search used to carry a hand-copied second implementation of the physics,
complete with its own constant for the Earth's rotation rate. Two copies drift,
and the drift stays invisible until a launch solution stops reproducing.

The claim under test is a behavioural one: the launch solution the search names
still hits when the physics module integrates it. That is what sharing a model
buys, and it is the thing a user of the renderer actually sees. The physics
itself is verified against closed forms in `test_physics.py`; nothing here
re-asserts it.
"""

from pathlib import Path

import pytest

from ballistics.physics import BallisticPhysics
from ballistics.targets import Target, check_collision
from sim.autosolve import DEFAULT_CANDIDATE_BUDGET, solve_launch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The launch solution the search returned for the snapshot below while it still
# carried its own acceleration model, recorded from the pre-change code.
#
# The search is a discrete grid search, so its answer is the best cell of a fixed
# grid rather than a smooth optimum. A difference in the last bits of an
# acceleration can move the winner to a neighbouring cell and shift the reported
# speed by tens of metres per second, so the tolerances below are set for that
# rather than fitted to the observed agreement, which is exact on this input.
#
# The snapshot is chosen so the recorded elevation is not sitting on the
# `min_auto_elevation` clamp. A solution pinned to a constraint tells you the
# constraint is still there, not that the physics underneath it is unchanged.
SPEED_TOLERANCE_MPS = 0.5
ELEVATION_TOLERANCE_DEG = 0.5
AZIMUTH_TOLERANCE_DEG = 0.5
MISS_DISTANCE_TOLERANCE_M = 0.5

RECORDED_SPEED_MPS = 236.24829514313345
RECORDED_ELEVATION_DEG = 6.50457126277492
RECORDED_AZIMUTH_DEG = 0.0
RECORDED_MISS_DISTANCE_M = 39.174967007593196


def snapshot():
    """A sheared-wind, northern-latitude atmosphere with a stationary target.

    The candidate budget covers the whole grid, so the search spends what the
    problem needs and the answer does not depend on how far it was allowed to
    look. A budget below the grid's ceiling would truncate the search and change
    the launch solution this file records.
    """
    return {
        "target_x_launch": 800.0,
        "target_velocity_x": 0.0,
        "target_radius": 40.0,
        "gravity": 9.81,
        "mass": 10.0,
        "rho": 0.4,
        "drag_coefficient": 0.47,
        "area": 0.01,
        "latitude": 0.3,
        "alt_levels": [0, 500, 1000, 2000, 3000],
        "wind_x_vals": [4, 6, 8, 10, 12],
        "wind_z_vals": [0, 4, 8, 12, 16],
        "wind_vertical_vals": [0, 0, 0, 0, 0],
        "min_auto_elevation": 5.0,
        "candidate_budget": DEFAULT_CANDIDATE_BUDGET,
    }


def snapshot_physics(atmosphere):
    """The shared physics model built from a search snapshot."""
    return BallisticPhysics(
        mass=atmosphere["mass"],
        gravity=atmosphere["gravity"],
        rho=atmosphere["rho"],
        drag_coefficient=atmosphere["drag_coefficient"],
        area=atmosphere["area"],
        latitude=atmosphere["latitude"],
        alt_levels=atmosphere["alt_levels"],
        wind_x_vals=atmosphere["wind_x_vals"],
        wind_z_vals=atmosphere["wind_z_vals"],
        wind_vertical_vals=atmosphere["wind_vertical_vals"],
    )


@pytest.fixture(scope="module")
def solved_launch():
    return solve_launch(snapshot())


def test_the_search_returns_the_launch_solution_it_returned_before(solved_launch):
    assert solved_launch["speed"] == pytest.approx(RECORDED_SPEED_MPS, abs=SPEED_TOLERANCE_MPS)
    assert solved_launch["elevation"] == pytest.approx(
        RECORDED_ELEVATION_DEG, abs=ELEVATION_TOLERANCE_DEG
    )
    assert solved_launch["azimuth"] == pytest.approx(RECORDED_AZIMUTH_DEG, abs=AZIMUTH_TOLERANCE_DEG)


def test_the_search_reports_the_miss_distance_it_reported_before(solved_launch):
    assert solved_launch["distance"] == pytest.approx(
        RECORDED_MISS_DISTANCE_M, abs=MISS_DISTANCE_TOLERANCE_M
    )


def test_the_launch_solution_hits_the_target_under_the_shared_integrator(solved_launch):
    """The point of sharing one acceleration model, stated as a physical claim.

    The search steps its own candidates at a fixed step for speed, so its own
    verdict is not the one a user gets. This propagates the launch solution it
    names through the physics module's integrator and asks the target interface
    whether it hit, which is the check the renderer performs when it draws the
    shot the search recommended.

    Before the models were unified this passed for a different reason: the search
    and the physics module were two transcriptions of the same equations, and
    agreeing with each other proved only that the transcription had not yet
    drifted. Now there is nothing left to agree with but the one model.
    """
    atmosphere = snapshot()
    xs, ys, zs, t_array, *_ = snapshot_physics(atmosphere).trajectory_3d(
        solved_launch["speed"],
        solved_launch["elevation"],
        solved_launch["azimuth"],
        return_time=True,
        t_final=200.0,
    )
    target = Target(atmosphere["target_x_launch"], radius=atmosphere["target_radius"])

    hit, _hit_index, miss_distance, _closest_index = check_collision(xs, ys, zs, target, t_array=t_array)

    assert hit, f"the search's own answer misses by {miss_distance:.1f} m under the shared integrator"
