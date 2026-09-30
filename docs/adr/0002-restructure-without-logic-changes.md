---
status: accepted
---

# Restructure by moving and deleting only, no logic changes

We will split the flat modules into a `src/ballistics/` package (physics, targets,
solvers) with the pygame app left importable where it is, and delete the dead
code: `_linspace`, `_simulate_candidate`, `_simulate_candidate_for_target`,
`_axis_input`, and `_smooth_velocity` in `sim/simulation.py`, plus the
module-level `trajectory_3d` wrapper in `physics.py`, which nothing calls and
which cannot forward `earth_radius` to its constructor.

The reason to constrain this is that a weekend is not enough to both restructure
and verify, and a rename-and-move sweep that quietly changes behaviour is worse
than leaving the layout alone. Bug fixes are exempt: the autosolver's snapshot
takes references to the wind arrays while the main loop mutates them in place, so
the solver's problem mutates under it mid-solve, and that gets fixed.

**Considered options:** a full library/app separation including moving the renderer
into the package, and a no-restructure option limited to adding `pyproject.toml`
and `tests/`. The first risks consuming the time budget with no visible output;
the second forfeits the library-plus-demo-app shape that the project spine depends
on.

**Consequences:** any refactor beyond a file move, a deletion, or a rename defers
to a later session. `closest_distance` in `targets.py` is renamed to match the
*miss distance* term in `CONTEXT.md`; that is a rename, not a behaviour change.

## Amendment: the launch solution search loses its own acceleration model

One exception is admitted beyond the deletions above. `sim/autosolve.py` carried a
hand-copied second implementation of the physics, complete with a private copy of
the Earth's rotation rate, and the search now calls `BallisticPhysics.compute_acceleration`
instead. The copy is deleted rather than moved.

The reason is that the convergence study cannot be trusted otherwise. It compares
two integrators, adaptive RK45 against fixed-step semi-implicit Euler, and a plot
of one against the other only measures the integrators if the acceleration
underneath them is the same function. Two acceleration models behind two
integrators produces a figure whose error curve is the sum of two disagreements
and cannot be attributed to either.

The comparison that justifies it: for a fixed wind-sheared snapshot, the search
returns the same launch solution before and after, and the projectile it names
passes within the target radius when integrated by the shared physics module. The
two implementations were already equivalent, which is the point. They were two
transcriptions of one model, and keeping both only guaranteed that one of them
would eventually stop matching. `tests/test_single_acceleration_model.py` pins
the launch solution and the hit it produces, so a re-introduced copy is a failing
test rather than a slow drift.

Routing the search through the shared model also changes how it receives the wind
table. The copy the search used to make for itself is gone with the rest of the
duplicated code, and `BallisticPhysics` keeps whatever arrays it is handed, which
does not copy a float array. The search therefore copies them again before
building the model, so the guarantee that the problem being solved is fixed at
launch survives the refactor instead of being quietly lost in it.

One behavioural difference is accepted rather than prevented. The search's copy
guarded the zero-relative-airspeed branch with an epsilon, the shared model
compares against zero exactly; the spec's separate decision to use an epsilon
there is still open, and it is the physics module that will settle it. On the
input above the two produce the same answer, because the projectile is never at
rest in the air at a step boundary.

**Considered options:** leaving the copy in place and having the convergence study
subtract it, and unifying the two into a shared private module neither integrator
owns. The first is not measurable in practice. The second re-creates the same
duplication one level down.

## Amendment: the renderer's step rule becomes RK4

A second exception is admitted, narrower than the first. The renderer's frame
loop no longer steps with semi-implicit Euler inline; it calls
`BallisticPhysics.integrate_fixed_step` with the RK4 rule. The acceleration model
is unchanged, the step is still one per frame, and the work is a loop of five
lines that moves rather than a rewrite.

The reason is that this is the decision the divergence study returned, and a
recorded decision that the code does not implement is a record of an intention
rather than of a decision. `docs/adr/0003-render-with-fixed-step-rk4.md` carries
the measurement and the reasoning.

**Considered options:** leaving the renderer on Euler and recording the RK4 swap
as follow-up work. That was the narrower reading of this boundary and it was
rejected because it leaves the study's finding published and unacted on, which
is the outcome this whole branch exists to avoid.

## Amendment: the launch solution search loses its wall clock

A second exception is admitted, recorded in
`docs/adr/0004-budget-the-launch-search-in-candidate-flights.md`. `sim/autosolve.py`
spent a fixed two seconds and returned whatever it had found when the clock ran
out, which made the launch solution depend on the machine and left the search
impossible to pin in a test. It is now bounded by a budget counted in candidate
flights, and the answer reports what it spent.

## Amendment: the package sits at the repository root, not under `src/`

The package is `ballistics/` beside `sim/`, not `src/ballistics/`. Physics and
the constants it reads default from move into it; the renderer stays at the root
where it already was.

The reason is that the renderer is staying at the root either way, and a `src/`
layout means shipping code from two roots. Setuptools describes that with a
`package_dir` mapping instead of the single `packages.find` this project already
uses, and the packaging test resolves those glob patterns against the repository
root to work out what the install ships. One layout for the library and the demo
app reads better than two layouts for one project, and it keeps the renderer's
own moves out of scope.

`config.py` moves with physics rather than staying behind as a loose root
module. It holds physical constants and the default wind table, which is physics
input, and the convergence script reads it only to label its figure. Leaving it
behind would give the package a dependency on the repository root.

**Considered options:** the `src/` layout as originally written here, and
keeping `config.py` at the root. The first costs a two-root `package_dir` and a
second root for the packaging test to resolve. The second leaves the physics
reading its defaults from a module that is not part of the package it lives in.

