"""A screenshot of the real-time renderer, taken without a display.

The window is the project's demo rather than its claim, but a reader should see
it working without running it. This script opens the real renderer on pygame's
dummy video driver, flies a projectile partway to a moving target, and saves
what the window drew.

Run it with:

    uv run python scripts/renderer_screenshot.py
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from ballistics.physics import BallisticPhysics
from figure_common import run

# The flight the screenshot shows. These are the renderer's own defaults, so
# the picture is the window as it ships: 300 m/s at 35 degrees against a
# target walking toward the launcher.
LAUNCH_SPEED_MPS = 300.0
LAUNCH_ELEVATION_DEG = 35.0
LAUNCH_AZIMUTH_DEG = 0.0
TARGET_X_M = 2800.0
TARGET_RADIUS_M = 20.0
TARGET_VELOCITY_X_MPS = 40.0

# How far into the flight the screenshot is taken. Eight simulated seconds at
# 60 fps puts the projectile mid-arc with a trail behind it, rather than a
# launch frame with nothing in the air or a landed frame with nothing to see.
SCREENSHOT_AT_S = 8.0
FRAME_STEP_S = 1.0 / 60.0
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

OUTPUT_PATH = "figures/renderer.png"


def describe_launch():
    """The conditions the screenshot was produced under, printed on the run."""
    return (
        f"v0 = {LAUNCH_SPEED_MPS:.0f} m/s   elevation = {LAUNCH_ELEVATION_DEG:.0f} deg   "
        f"azimuth = {LAUNCH_AZIMUTH_DEG:.0f} deg   target x0 = {TARGET_X_M:.0f} m   "
        f"target vx = {TARGET_VELOCITY_X_MPS:.0f} m/s   radius = {TARGET_RADIUS_M:.0f} m   "
        f"screenshot at t = {SCREENSHOT_AT_S:.1f} s"
    )


def main(output_path=OUTPUT_PATH):
    import pygame

    from sim.simulation import PygameBallisticSimulation

    pygame.init()
    try:
        renderer = PygameBallisticSimulation(
            simulator=BallisticPhysics(),
            launch_speed=LAUNCH_SPEED_MPS,
            launch_elevation_deg=LAUNCH_ELEVATION_DEG,
            launch_azimuth_deg=LAUNCH_AZIMUTH_DEG,
            target_x=TARGET_X_M,
            target_radius=TARGET_RADIUS_M,
            target_velocity_x=TARGET_VELOCITY_X_MPS,
            window_width=WINDOW_WIDTH,
            window_height=WINDOW_HEIGHT,
        )
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        renderer._font_main = pygame.font.SysFont("consolas", 20)
        renderer._font_small = pygame.font.SysFont("consolas", 16)
        renderer._font_title = pygame.font.SysFont("consolas", 22)
        renderer._initialize_sliders(pygame)
        renderer._begin_camera_goals()

        renderer._launch_projectile()
        steps = int(SCREENSHOT_AT_S / FRAME_STEP_S)
        for _ in range(steps):
            renderer._apply_slider_values()
            renderer._update_physics(FRAME_STEP_S)
        renderer._draw_scene(screen, pygame)
        pygame.image.save(screen, output_path)
    finally:
        pygame.quit()

    print("=== Renderer screenshot ===")
    print("Launch conditions")
    print(describe_launch())
    print(f"wrote {output_path}")


if __name__ == "__main__":
    run(main, OUTPUT_PATH, "Capture the real-time renderer without a display")
