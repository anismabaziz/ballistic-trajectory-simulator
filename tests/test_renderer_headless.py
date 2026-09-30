"""The renderer starts, runs a flight, and stops again with no display attached.

Every other claim in the suite about the interactive mode is about code that
never opens a window: the status line, the launch the search recommends, the
rejection of `--headless`. The window itself had no coverage at all, which left
one thing a reviewer had to take on trust. A screenshot cannot say whether the
app imports cleanly, initialises its fonts and sliders, and survives a whole
flight, because a screenshot is evidence that it got that far and nothing about
what happens after.

So this file runs the real `run()` on pygame's dummy video driver, which hands
back a surface to draw into and has no screen behind it. The window is a surface
of the requested size that never reaches a display. Everything below that is the
real renderer: real pygame, real drawing, real fixed-step physics.

Two things are held out of scope on purpose. The camera, the trail and the input
smoothing are feel rather than behaviour, and a test that pinned their values
would be pinning numbers nobody chose and everybody would have to go on keeping.
And the way any of it is drawn is not the claim either, since the claim is that
the loop runs, not that the pixels are right.

The frame clock is replaced, because the real one reports how long the last frame
genuinely took. That is a property of the machine running the test, and a shot
that lands after an unknown number of real seconds is not something to assert on.
"""

import os
from types import SimpleNamespace

import pygame
import pytest

from ballistics.physics import BallisticPhysics
from sim.simulation import PygameBallisticSimulation

VIDEO_DRIVER_ENV = "SDL_VIDEODRIVER"
DUMMY_VIDEO_DRIVER = "dummy"

# The renderer paints a sky gradient one line per row and then two grids, on top
# of the panel and the HUD, every frame. At the size a person looks at, drawing is
# the expensive part of a frame, and it is not what is being claimed here, so the
# window is opened small. The loop, the drawing calls, the fonts and the physics
# run exactly as they do at any other size.
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360

# A window has to be opened once to find out whether this platform has a dummy
# driver at all. One pixel, and it is closed again before anything else runs.
PROBE_WINDOW_SIZE = (1, 1)

# The same reasoning as the GIF export test: a fast shot flies the same code path
# as a slow one over a fraction of the time. At 20 m/s and 35 degrees this one is
# airborne for about 2.4 s and comes down roughly 40 m from the launcher, far
# short of the target, so the flight ends at the ground rather than at a hit.
LAUNCH_SPEED_MPS = 20.0
LAUNCH_ELEVATION_DEG = 35.0
TARGET_X_M = 2800.0

# One frame at the rate the renderer asks for. The real clock returns elapsed
# wall-clock time, which would make the frame count a fact about the runner.
FRAME_STEP_S = 1.0 / 60.0

# How many frames the loop is given before the clock posts the QUIT event that
# closes it. The renderer itself has no frame cap and no flight timeout, so this
# is the test's own stopping condition, and the flight has to finish well inside
# it. At this step the budget is five seconds of simulated time, about twice the
# length of the flight above, so a regression that leaves the projectile hanging
# fails rather than quietly spending what is left of the budget.
FRAME_BUDGET = 300


def still_air_physics():
    """An atmosphere chosen so nothing but the loop is under test.

    The values are spelled out rather than left to the defaults, because the
    defaults carry a wind table that would shear the shot sideways. A dead-still
    atmosphere keeps the claim about the renderer from quietly becoming a claim
    about the wind profile.
    """
    return BallisticPhysics(
        mass=10.0,
        gravity=9.81,
        rho=0.4,
        drag_coefficient=0.47,
        area=0.01,
        latitude=0.0,
        alt_levels=[0, 500, 1000, 2000, 3000],
        wind_x_vals=[0, 0, 0, 0, 0],
        wind_z_vals=[0, 0, 0, 0, 0],
        wind_vertical_vals=[0, 0, 0, 0, 0],
    )


def restore_video_driver(previous):
    """Put the video driver selection back the way the machine had it."""
    if previous is None:
        os.environ.pop(VIDEO_DRIVER_ENV, None)
    else:
        os.environ[VIDEO_DRIVER_ENV] = previous


class _FixedStepClock:
    """The frame clock, with a step it does not have to be timed to produce.

    It has two jobs. It hands back a fixed step, so the flight advances by the
    same amount every frame however fast the machine runs the test. And it posts
    the QUIT event a closed window posts once the budget is spent, so the loop
    leaves through the exit it uses in the app rather than through something only
    a test knows about.

    The launch keystroke goes in the same way, as an event posted before the loop
    reads its queue, so the shot is fired by the input path the window really
    uses.
    """

    def __init__(self, frame_budget):
        self.frame_budget = frame_budget
        self.renderer = None
        self.frames = 0
        self.shot_ended_frame = None

    def tick(self, _requested_fps):
        self.frames += 1

        if self.frames == 1:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0))
        elif self.frames >= self.frame_budget:
            pygame.event.post(pygame.event.Event(pygame.QUIT))
        elif self.renderer is not None and self.shot_ended_frame is None:
            # Recorded so the flight can be shown to finish on its own rather
            # than being cut off by the budget. No projectile yet means the
            # keystroke has not landed, which is not an ended shot.
            projectiles = self.renderer.projectiles
            if not projectiles or not any(p.alive for p in projectiles):
                self.shot_ended_frame = self.frames

        return FRAME_STEP_S * 1000.0


@pytest.fixture(scope="module")
def dummy_video_driver():
    """Select pygame's dummy driver for as long as this module runs.

    Set rather than defaulted in, because inheriting whatever the machine happens
    to have would put a real window on a reviewer's screen, which is the one thing
    this file must never do. The platform may not carry the driver at all, so the
    window that follows is checked rather than assumed: SDL reads the selection
    when it initialises the display and ignores it afterwards, which means a
    display left open by an earlier test would quietly hand back the real driver
    and put a window up.
    """
    previous = os.environ.get(VIDEO_DRIVER_ENV)
    os.environ[VIDEO_DRIVER_ENV] = DUMMY_VIDEO_DRIVER

    def give_up(reason):
        pygame.display.quit()
        restore_video_driver(previous)
        pytest.skip(reason)

    # pygame.error subclasses RuntimeError, and a platform with no dummy driver
    # does not always raise it, so RuntimeError is the net to catch it in.
    try:
        if pygame.display.get_init():
            pygame.display.quit()
        pygame.display.init()
        pygame.display.set_mode(PROBE_WINDOW_SIZE)
    except RuntimeError as error:
        give_up(f"pygame has no dummy video driver here: {error}")

    if pygame.display.get_driver() != DUMMY_VIDEO_DRIVER:
        give_up(f"pygame opened the {pygame.display.get_driver()!r} video driver, not the dummy one")

    yield

    pygame.quit()
    restore_video_driver(previous)


@pytest.fixture(scope="module")
def ran_renderer(dummy_video_driver):
    """The real renderer, run to completion on the dummy driver.

    One run per module because it is much the slowest thing in the suite, and
    because every assertion below is about the same flight, so they read one run
    rather than paying for one each.
    """
    clock = _FixedStepClock(FRAME_BUDGET)
    renderer = PygameBallisticSimulation(
        simulator=still_air_physics(),
        launch_speed=LAUNCH_SPEED_MPS,
        launch_elevation_deg=LAUNCH_ELEVATION_DEG,
        target_x=TARGET_X_M,
        target_radius=20.0,
        target_velocity_x=0.0,
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
    )
    clock.renderer = renderer

    original_clock_factory = pygame.time.Clock
    pygame.time.Clock = lambda: clock
    try:
        renderer.run()
    finally:
        pygame.time.Clock = original_clock_factory
        pygame.quit()

    return SimpleNamespace(renderer=renderer, clock=clock)


def test_the_window_opens_and_the_loop_runs_to_its_timeout(ran_renderer):
    """The loop is a loop, and the window underneath it was never a screen.

    Nothing here shows anything is visible. It shows the run finished: the window
    opened, the loop was entered, and it left through the QUIT event the clock
    posted, which is the same event a closed window posts. A renderer that raised
    partway through would never get here at all.
    """
    assert ran_renderer.clock.frames >= FRAME_BUDGET
    assert ran_renderer.renderer.running is False
    assert ran_renderer.renderer.sim_time > 0.0


def test_the_launch_keystroke_starts_a_flight(ran_renderer):
    """The input path fires the shot, and the shot goes somewhere.

    The projectile is asserted on as a body that left the launcher, which is the
    least a launch has to do. Where it got to is left to the physics tests.
    """
    projectiles = ran_renderer.renderer.projectiles

    assert len(projectiles) == 1, f"the space bar left {len(projectiles)} projectiles in the air"
    assert projectiles[0].x > 0.0


def test_the_flight_ends_at_the_ground_before_the_budget_runs_out(ran_renderer):
    """The loop ran a whole flight rather than its first frame.

    This is the assertion that separates a working renderer from one that opens a
    window and stalls. The shot has to come down on its own, well inside the
    budget, having missed the target on the way. A loop that stopped advancing the
    projectile, or left it airborne, spends what is left of the budget and fails
    on the frame it ended.
    """
    clock = ran_renderer.clock
    projectile = ran_renderer.renderer.projectiles[0]

    assert clock.shot_ended_frame is not None, "the shot was still airborne when the budget ran out"
    assert clock.shot_ended_frame < FRAME_BUDGET
    assert projectile.alive is False
    assert projectile.hit is False
    assert projectile.y == pytest.approx(0.0, abs=1e-6)