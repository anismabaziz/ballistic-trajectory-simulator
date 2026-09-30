"""The command line is one command that solves and one command that renders.

The interface used to present four peer run modes, three of which the project
never claimed to be about. The spine is a solver with a demo attached, so the
interface now says the same thing: `solve` reports a launch solution, `render`
opens the real-time window.

Two things are held here rather than left to the reader. Every command the
README documents is fed to the parser, because a documented flag the parser does
not accept is a broken instruction that only a reviewer running it would catch.
And the capabilities the removed modes carried are either reachable from one of
the two commands or still callable where they live, since dropping a scenario
without saying where it went is how a project quietly loses a feature.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# `main` is the entry point rather than an installed module, so the parser is
# read from the repository root the same way a user runs it.
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from ballistics.physics import BallisticPhysics  # noqa: E402
from main import add_render_arguments, add_solve_arguments, build_cli_parser  # noqa: E402
from utils import plot_intercept_trajectories, plot_salvo_dispersion_3d  # noqa: E402

SOLVE_COMMAND = "solve"
RENDER_COMMAND = "render"


def run_cli(*args, timeout=300):
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def reported_number(stdout, label):
    """The number a report line carries, so the assertion is about the value's shape."""
    match = re.search(rf"^{label}: (-?[\d.]+)", stdout, re.MULTILINE)
    assert match is not None, f"no {label} line in:\n{stdout}"
    return float(match.group(1))


@pytest.fixture(scope="module")
def solve_result():
    return run_cli(SOLVE_COMMAND, "--headless")


def test_the_solve_command_completes_without_a_display(solve_result):
    assert solve_result.returncode == 0, solve_result.stderr


def test_the_solve_command_reports_a_launch_angle(solve_result):
    assert re.search(r"^Launch angle: \d+\.\d+ degrees$", solve_result.stdout, re.MULTILINE)


def test_the_solve_command_says_whether_the_shot_hit(solve_result):
    assert re.search(r"^Hit: (True|False)$", solve_result.stdout, re.MULTILINE)


def test_the_solve_command_reports_the_miss_distance(solve_result):
    assert re.search(r"^Miss distance: \d+\.\d+ m$", solve_result.stdout, re.MULTILINE)


def test_the_solve_command_hits_the_target_it_defaults_to(solve_result):
    """A default solve reports a hit rather than a miss it has to explain."""
    assert re.search(r"^Hit: True$", solve_result.stdout, re.MULTILINE)


def test_the_solve_command_hits_a_moving_target():
    result = run_cli(SOLVE_COMMAND, "--headless", "--target-x", "2000", "--target-velocity-x", "40")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^Hit: True$", result.stdout, re.MULTILINE)


def test_the_solve_command_admits_a_miss_for_a_target_it_cannot_reach():
    result = run_cli(SOLVE_COMMAND, "--headless", "--target-x", "60000")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^Hit: False$", result.stdout, re.MULTILINE)
    assert reported_number(result.stdout, "Miss distance") > 1000.0


def test_the_solve_command_solves_in_a_curved_model_when_asked():
    """Earth curvature reaches the physics that scores the shot, not only the plot.

    The flag used to bend the drawn flight while the answer came from a flat
    model, so the reported miss distance and the picture disagreed. Curvature
    leaves the range alone and drops the trajectory as it goes, so the number
    that moves is the miss distance against a target sitting at ground level.
    """
    flat = run_cli(SOLVE_COMMAND, "--headless")
    curved = run_cli(SOLVE_COMMAND, "--headless", "--enable-earth-curvature")

    assert curved.returncode == 0, curved.stderr
    assert re.search(r"^Hit: True$", curved.stdout, re.MULTILINE)
    assert reported_number(curved.stdout, "Miss distance") != reported_number(flat.stdout, "Miss distance")


def test_the_render_command_rejects_the_headless_flag():
    """The flag belongs to `solve`, so argparse refuses it before anything opens.

    That the window itself starts with no display is a separate claim, covered by
    the renderer test that runs the real loop on pygame's dummy video driver.
    """
    result = run_cli(RENDER_COMMAND, "--headless")
    assert result.returncode != 0
    assert RENDER_COMMAND in result.stderr


def test_a_removed_run_mode_is_no_longer_an_entry_point():
    result = run_cli("--mode", "three-d-simulation", "--headless")
    assert result.returncode != 0


@pytest.mark.parametrize(
    "command",
    [
        "target-intercept",
        "real-time-animation",
        "three-d-simulation",
        "interactive-simulator",
    ],
)
def test_the_former_run_modes_are_rejected_by_name(command):
    assert run_cli(command, "--headless").returncode != 0


def documented_commands():
    """Every command the README tells a reader to run, as argument lists."""
    readme = (PROJECT_ROOT / "README.md").read_text()
    commands = []
    for line in readme.splitlines():
        stripped = line.strip()
        if "python main.py" in stripped:
            commands.append(stripped.split("python main.py", 1)[1].split())
    assert commands, "the README documents no command at all"
    return commands


def test_every_command_the_readme_documents_parses():
    parser = build_cli_parser()
    for arguments in documented_commands():
        parser.parse_args(arguments)


def test_the_readme_documents_both_entry_points():
    documented = {arguments[0] for arguments in documented_commands()}
    assert documented == {SOLVE_COMMAND, RENDER_COMMAND}


def accepted_flags():
    """Every option string the two commands accept, read off their parsers."""
    return {
        option
        for add_arguments in (add_solve_arguments, add_render_arguments)
        for option in _option_strings(add_arguments)
    }


def _option_strings(add_arguments):
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return {option for action in parser._actions for option in action.option_strings}


def test_the_readme_documents_no_flag_the_parser_rejects():
    """The arguments the README lists have to exist on one of the two commands.

    Read off the parsers rather than hardcoded, so a flag renamed in the code
    fails here instead of quietly leaving the documentation pointing at a name
    that no longer parses.
    """
    documented_flags = set(re.findall(r"`(--[a-z-]+)`", (PROJECT_ROOT / "README.md").read_text()))

    assert documented_flags <= accepted_flags(), (
        f"documented but not accepted: {sorted(documented_flags - accepted_flags())}"
    )


# The GIF writer draws one frame per integration sample, so what this test costs
# is the length of the flight rather than anything about the export path. The
# default 300 m/s shot flies for 26 s and integrates to about 530 samples, which
# takes the best part of a minute to encode and costs more than the rest of the
# suite together. 20 m/s flies the same code path over about 50
# samples instead, so the export is still proven and the test costs a few
# seconds.
#
# The frame floor below is what keeps that trade honest. Without it a writer that
# emitted a single frame would pass this test as readily as one that emits fifty,
# and shortening the shot would quietly have gutted the case. Twenty frames is
# well under what this shot produces and well over a stub, so the assertion
# catches both a broken writer and a test that has stopped exercising one.
GIF_EXPORT_LAUNCH_SPEED_MPS = 20.0
GIF_EXPORT_MIN_FRAMES = 20


def test_the_solve_command_exports_the_flight_as_an_animation(tmp_path):
    from PIL import Image

    gif_path = tmp_path / "trajectory.gif"
    result = run_cli(
        SOLVE_COMMAND,
        "--headless",
        "--launch-speed",
        str(GIF_EXPORT_LAUNCH_SPEED_MPS),
        "--target-x",
        "30",
        "--target-radius",
        "2",
        "--output-gif-path",
        str(gif_path),
    )

    assert result.returncode == 0, result.stderr
    assert gif_path.stat().st_size > 0

    with Image.open(gif_path) as animation:
        frame_count = getattr(animation, "n_frames", 1)
    assert frame_count >= GIF_EXPORT_MIN_FRAMES, (
        f"the exported animation holds {frame_count} frame(s), so the export path "
        "is not really being exercised"
    )


# The salvo dispersion and the intercept scenario were run modes, and a run mode
# is a place a capability can quietly disappear from. They are figure generators
# rather than entry points now, and this is what says they still run: both
# functions are called against the real physics with no display attached, so
# neither can rot behind an argument nobody passes any more.
SALVO_PROJECTILE_COUNT = 5
SALVO_AZIMUTH_SPAN_DEG = 20.0


def test_the_salvo_dispersion_figure_still_renders_without_a_display():
    physics = BallisticPhysics()
    azimuths = np.linspace(-SALVO_AZIMUTH_SPAN_DEG / 2, SALVO_AZIMUTH_SPAN_DEG / 2, SALVO_PROJECTILE_COUNT)

    plot_salvo_dispersion_3d(
        physics,
        300.0,
        35.0,
        azimuths,
        apply_earth_curvature=True,
        show_plot=False,
    )


def test_the_intercept_figure_still_renders_without_a_display():
    physics = BallisticPhysics()
    xs, ys, zs, *_ = physics.trajectory_3d(300.0, 35.0, return_time=True)
    interceptor_xs, interceptor_ys, interceptor_zs, *_ = physics.trajectory_3d(320.0, 50.0, return_time=True)

    plot_intercept_trajectories(
        np.column_stack((xs, ys)),
        np.column_stack((interceptor_xs, interceptor_ys)),
        show_plot=False,
    )