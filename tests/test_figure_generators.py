"""The scenarios the project kept for their pictures, and the one command that redraws them.

The salvo dispersion and the intercept were run modes once, and a run mode is a
place a capability quietly rots in: nobody runs it, so nobody notices it broke.
They are figure generators now, which is a job with a finish line. Each one
writes a PNG into `figures/`, states the launch conditions it flew, and runs
with no display attached.

The drift guarded against here is between three things that are supposed to agree:
the committed figures, the scripts that draw them, and the README that shows
them. A PNG in `figures/` that nothing regenerates is a screenshot with extra
steps: a reader cannot tell whether it still holds, and nobody can check. A
figure in the README that no generator writes is an instruction that cannot be
followed. Both fail here, as does a generator whose output nobody committed.
"""

import re
import subprocess
import sys
from pathlib import Path

import matplotlib
import pytest
from PIL import Image

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / "figures"

REGENERATE_COMMAND = "make_figures.py"

# The generators this ticket added, each with the file it owns. One place, so the
# two tests that walk them cannot disagree about what exists.
SCENARIO_FIGURES = [
    ("salvo_dispersion.py", "salvo_dispersion.png"),
    ("intercept.py", "intercept.png"),
]


def run_script(name, *arguments, timeout=900):
    return subprocess.run(
        [sys.executable, f"scripts/{name}", *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def figure_size(path):
    with Image.open(path) as image:
        return image.size


def reported_numbers(stdout):
    """The `Label: number` lines a run reports, keyed by label.

    Read off the report rather than off the generator's own constants, so the
    assertion is about what a reader sees when they run the script. The unit the
    report prints is ignored; a line with no leading number is not a reported
    measurement and is left out.
    """
    reported = {}
    for line in stdout.splitlines():
        label, separator, value = line.partition(": ")
        if not separator:
            continue
        match = re.match(r"-?[\d.]+", value)
        if match:
            reported[label] = float(match.group())
    return reported


def readme_figures():
    """The figure files the README shows, as paths relative to the repository.

    Remote images are left out: the badge is served by a workflow, not drawn by
    anything in this repository, so only the local paths are figures the
    regeneration command is answerable for.
    """
    targets = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", (PROJECT_ROOT / "README.md").read_text())
    return [target for target in targets if not target.startswith(("http://", "https://"))]


@pytest.fixture(scope="module")
def scenario_runs(tmp_path_factory):
    """One run of each scenario generator, shared by every test that reads a report.

    Each run flies a salvo or searches 190 interceptor angles, and the assertions
    below all read the same reports, so running each generator once rather than
    once per test is the difference between a few seconds and most of a minute.
    """
    runs = {}
    for script, output_name in SCENARIO_FIGURES:
        output = tmp_path_factory.mktemp(script) / output_name
        runs[script] = (output, run_script(script, "--output", str(output)))
    return runs


@pytest.mark.parametrize("script", [name for name, _ in SCENARIO_FIGURES])
def test_the_generator_writes_its_figure_with_no_display_attached(scenario_runs, script):
    """Each script runs from the repository root and writes the figure it owns."""
    output, result = scenario_runs[script]

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert output.stat().st_size > 0
    assert min(figure_size(output)) >= 400, "a figure this size cannot be read on a screen"


@pytest.mark.parametrize("script", [name for name, _ in SCENARIO_FIGURES])
def test_the_generator_states_the_launch_conditions_it_flew(scenario_runs, script):
    """A reader holding the PNG has to be able to tell what it is looking at.

    The conditions are printed rather than only drawn, so they can be asserted
    here and so a run in a terminal reports the same numbers the figure carries.
    """
    _output, result = scenario_runs[script]

    assert result.returncode == 0, result.stderr
    assert "Launch conditions" in result.stdout
    for value in ("m/s", "deg", "kg"):
        assert value in result.stdout, f"the report is missing a {value} it should state"


def test_the_salvo_figure_spreads_its_rounds_across_the_azimuths_it_claims(scenario_runs):
    """The dispersion is the point of the figure, so the spread has to be real.

    A salvo flown at one azimuth would satisfy every other condition here and
    draw a single line, so the report is checked for a cross-range spread wider
    than the ground the rounds land on.
    """
    _output, result = scenario_runs["salvo_dispersion.py"]

    assert result.returncode == 0, result.stderr
    reported = reported_numbers(result.stdout)
    assert reported["Impact cross-range spread"] > 100.0
    assert reported["Rounds"] >= 5


def test_the_intercept_figure_meets_a_primary_already_in_flight(scenario_runs):
    """The interceptor has to arrive somewhere the primary already is.

    Both projectiles leave one launcher, so scored from launch the smallest
    separation is zero at t=0 for every angle and the figure would be two arcs
    leaving the same point. The report states the delay before the interceptor
    launches, and the meeting has to happen after it, at a range the primary has
    already covered.
    """
    _output, result = scenario_runs["intercept.py"]

    assert result.returncode == 0, result.stderr
    reported = reported_numbers(result.stdout)
    assert reported["Interceptor launch delay"] > 1.0
    assert reported["Miss distance"] < 25.0
    assert reported["Miss time"] > 1.0
    assert reported["Primary range at intercept"] > 500.0


def reported_figures():
    """The figures the single regeneration command says it writes, as absolute paths.

    The command reads each path off the generator that draws the figure, so this
    list cannot go stale against the scripts it names.
    """
    result = run_script(REGENERATE_COMMAND, "--list", timeout=300)
    assert result.returncode == 0, result.stderr
    listed = [Path(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    assert listed, "the regeneration command lists no figures at all"
    return {(PROJECT_ROOT / path).resolve() for path in listed}


def test_every_committed_figure_is_regenerable_by_the_one_command():
    """The claim the figures rest on: nothing in `figures/` is a dead screenshot."""
    committed = sorted(path.resolve() for path in FIGURES.iterdir() if path.suffix == ".png")
    assert committed, "the repository holds no figures, so there is nothing to keep regenerable"

    regenerable = reported_figures()
    orphans = [path.name for path in committed if path not in regenerable]

    assert not orphans, f"no generator writes {orphans}"


def test_every_figure_the_command_writes_is_committed():
    """A generator whose output nobody committed is a figure the README cannot show."""
    missing = [path.name for path in sorted(reported_figures()) if not path.is_file()]

    assert not missing, f"the regeneration command writes {missing}, which is not in the repository"


def test_every_figure_the_readme_shows_is_one_the_command_regenerates():
    """A figure in the README nobody can redraw is an instruction that cannot be followed."""
    regenerable = reported_figures()
    unreproducible = [name for name in readme_figures() if (PROJECT_ROOT / name).resolve() not in regenerable]

    assert not unreproducible, f"the README shows {unreproducible}, which no generator writes"


def test_the_readme_shows_every_figure_the_command_regenerates():
    """A figure the regeneration command redraws but the README never shows is dead weight.

    The other direction of the same claim: the command is documented as covering
    everything the README displays, so it cannot quietly grow a figure the
    documentation does not mention.
    """
    shown = {(PROJECT_ROOT / name).resolve() for name in readme_figures()}
    assert shown, "the README shows no figures at all"

    unshown = sorted(path.name for path in reported_figures() if path not in shown)

    assert not unshown, f"the regeneration command draws {unshown}, which the README does not show"
