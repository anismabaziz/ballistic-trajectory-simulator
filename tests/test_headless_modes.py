import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args, timeout=300):
    return subprocess.run(
        [sys.executable, "main.py", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def intercept_result():
    return run_cli("--mode", "target-intercept", "--headless")


@pytest.fixture(scope="module")
def interceptor_output(intercept_result):
    return intercept_result.stdout.split("=== Interceptor result ===")[1]


def test_default_scenario_completes_headless(intercept_result):
    assert intercept_result.returncode == 0, intercept_result.stderr


def test_default_scenario_reports_its_results(intercept_result):
    assert "Stationary target result" in intercept_result.stdout
    assert "Moving target result" in intercept_result.stdout
    assert "Interceptor result" in intercept_result.stdout


def test_intercept_scenario_reports_the_launch_solution(interceptor_output):
    assert "Interceptor launch angle:" in interceptor_output


def test_intercept_scenario_reports_the_miss_distance(interceptor_output):
    assert re.search(r"^Miss distance: \d+\.\d+ m$", interceptor_output, re.MULTILINE)


def test_intercept_scenario_does_not_claim_a_hit_it_cannot_determine(interceptor_output):
    """Both projectiles share a launcher, so a hit verdict would be meaningless."""
    assert re.search(r"^Hit: not determined", interceptor_output, re.MULTILINE)


def test_three_dimensional_mode_completes_headless():
    result = run_cli(
        "--mode",
        "three-d-simulation",
        "--headless",
        "--enable-earth-curvature",
    )
    assert result.returncode == 0, result.stderr


def test_salvo_dispersion_completes_headless():
    result = run_cli(
        "--mode",
        "three-d-simulation",
        "--headless",
        "--enable-salvo",
        "--enable-earth-curvature",
    )
    assert result.returncode == 0, result.stderr


# The GIF writer draws one frame per integration sample, so what this test costs
# is the length of the flight rather than anything about the export path. The
# renderer's default 300 m/s shot flies for 26 s and integrates to about 530
# samples, which takes the best part of a minute to encode and costs more than
# the rest of the suite together. 20 m/s flies the same code path over about 50
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


def test_gif_export_writes_the_animation_headless(tmp_path):
    gif_path = tmp_path / "trajectory.gif"
    result = run_cli(
        "--mode",
        "real-time-animation",
        "--headless",
        "--launch-speed",
        str(GIF_EXPORT_LAUNCH_SPEED_MPS),
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


def test_interactive_mode_rejects_headless():
    result = run_cli("--mode", "interactive-simulator", "--headless")
    assert result.returncode != 0
    assert "interactive-simulator" in result.stderr
