import re
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_gif_export_writes_the_animation_headless(tmp_path):
    gif_path = tmp_path / "trajectory.gif"
    result = run_cli(
        "--mode",
        "real-time-animation",
        "--headless",
        "--output-gif-path",
        str(gif_path),
    )
    assert result.returncode == 0, result.stderr
    assert gif_path.stat().st_size > 0
