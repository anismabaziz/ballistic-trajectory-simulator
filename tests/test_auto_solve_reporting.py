"""What the renderer tells the user about the launch solution it was handed.

The search reports how many candidates it spent and whether it got through all of
them. That information is worth nothing if the renderer drops it, because a
truncated answer and the best of the whole grid look identical on screen once the
text says "best". These tests pin the status line for both cases, and pin that the
caller reads the search's budget rather than ignoring it.

The renderer is a pygame window and needs a display to open, which is out of
reach for a test. What is being claimed here is the status text and the call that
applies it, neither of which touches pygame, so the renderer is stood in for by an
object holding just the attributes the code path reads.
"""

from types import SimpleNamespace

from sim.simulation import PygameBallisticSimulation, describe_auto_solution

HIT_GREEN = (120, 240, 150)
MISS_AMBER = (255, 210, 130)


def solution(**overrides):
    """A solved launch the way the search reports one, with the budget filled in."""
    result = {
        "ok": True,
        "distance": 12.0,
        "speed": 236.2,
        "elevation": 6.5,
        "azimuth": 0.0,
        "hit": True,
        "candidates_used": 1000,
        "candidate_budget": 1000,
        "exhausted": False,
    }
    result.update(overrides)
    return result


def test_a_search_that_finished_its_grid_says_so():
    """The best of everything the search looked at is the claim being made here."""
    text, color = describe_auto_solution(solution())

    assert "best of the search" in text
    assert "1000" in text
    assert color == HIT_GREEN


def test_a_search_that_ran_out_of_budget_says_so():
    """A truncated answer is named as one, so "best" cannot be read as "best available".

    The two cases have to read differently. A status line that said the same thing
    either way is the exact silence that made the old wall-clock budget
    misleading, so this test fails on wording alone.
    """
    text, color = describe_auto_solution(
        solution(hit=False, distance=420.0, candidates_used=260, candidate_budget=260, exhausted=True)
    )

    assert "budget" in text
    assert "260" in text
    assert color == MISS_AMBER


def test_the_hit_and_miss_colours_still_say_which_it_was():
    """The budget wording does not swallow the hit verdict it sits next to."""
    _hit_text, hit_color = describe_auto_solution(solution())
    _miss_text, miss_color = describe_auto_solution(solution(hit=False, distance=420.0))

    assert hit_color != miss_color


def test_the_renderer_applies_the_status_the_search_earned():
    """The path that hands the search's answer to the window reads the budget.

    This is the assertion that the reporting is not a dead function. It calls the
    real method on a stand-in for the renderer and compares the status text it
    sets against the one the pure reporter produces, so dropping the budget from
    the applied status fails here even if the reporter still has it.
    """
    result = solution(candidates_used=260, candidate_budget=260, exhausted=True, hit=False, distance=420.0)
    expected, _color = describe_auto_solution(result)

    launched = []
    renderer = SimpleNamespace(
        launch_speed=300.0,
        launch_elevation_deg=35.0,
        launch_azimuth_deg=0.0,
        min_auto_solve_elevation_deg=12.0,
        auto_status_text="",
        auto_status_color=(160, 190, 175),
        _launch_projectile=lambda: launched.append(True),
    )

    PygameBallisticSimulation._apply_auto_solver_result(renderer, result)

    assert renderer.auto_status_text == expected
    assert "budget" in renderer.auto_status_text
    assert launched, "the applied result did not launch a projectile"


def test_a_failed_search_is_reported_as_a_failure():
    """A search that found nothing at all says so and launches nothing.

    A failed search spent no candidates and reported no launch solution, so there
    is no budget to put on the line here. What it must not do is fall through to
    the wording that reads launch numbers off the answer, which would take the
    renderer down over a search that simply found nothing.
    """
    launched = []
    renderer = SimpleNamespace(
        auto_status_text="",
        auto_status_color=(160, 190, 175),
        _launch_projectile=lambda: launched.append(True),
    )

    PygameBallisticSimulation._apply_auto_solver_result(renderer, {"ok": False})

    assert renderer.auto_status_text == "Auto solve failed"
    assert not launched