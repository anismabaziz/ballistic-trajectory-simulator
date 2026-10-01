---
status: accepted
---

# Lead the project with the physics, and report the integrator divergence as the finding

The project has four run modes and no stated thesis. We are reframing it as a 3D
ballistics solver (drag, altitude-dependent wind shear, Coriolis) plus a shooting
method that searches speed, elevation, and azimuth to hit a moving target, with
the real-time sandbox kept as the demo rather than the claim.

The headline result is deliberately a problem we found in our own code: the
solver runs an adaptive RK45 integration while the interactive sim and the
autosolver run fixed-step semi-implicit Euler, and the two disagree by enough for
the solver to recommend a launch angle the renderer then misses. We will quantify
that disagreement rather than quietly paper over it.

We chose this over three alternatives: a pure numerics-library pitch (no hook for
a reader who does not care about ODEs), an interactive-sandbox pitch (leans on a
software-mode pygame renderer with no shading, which is the weakest part of the
codebase), and a fire-control-system pitch (reads as weapons-adjacent and closes
more doors than it opens).

**Considered options:** interactive-sandbox-first, fire-control framing, and pure
library framing, all rejected above.

**Consequences:** feature work is frozen. Tests, docs, and consistency between the
solver and the renderer take the remaining time, and the README must state plainly
which physical approximations are wrong at long range, namely the post-hoc Earth
curvature correction and the Coriolis term that omits the vertical component.
