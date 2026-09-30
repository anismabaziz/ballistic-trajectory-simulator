---
status: accepted
---

# Budget the launch solution search in candidate flights, not in seconds

`solve_launch` searched a grid of 54 flight times by 7 azimuths and then refined
around the winner over 7 by 7 by 7, and it stopped when a two-second wall clock
ran out. It returned whatever it had found at that moment and set a `timed_out`
flag, which nothing read. Two consequences, and the second is the one that made
the search untestable.

The first is that the answer depended on the machine. The same snapshot solved on
a loaded laptop and on an idle one returned different launch solutions, because
the cut landed at a different point in the grid. There was no way to write a test
that solved an input twice and got the same answer, which is why the determinism
test in `tests/test_searches.py` was marked as an expected failure.

The second is that the cost of the search stopped being a property of the search.
A two-second budget is not a statement about the problem, it is a statement about
the machine, and nothing in the codebase could say how large the grid was. The
ceiling is 721 candidate flights, knowable from the loop bounds, and the grid can
now be given a budget derived from that arithmetic instead of from a measurement.

## The decision

The budget counts candidate flights. Each call to `_simulate_candidate` claims one
from a `_CandidateBudget`, and the loops stop when the claim fails. The default,
1000, sits above the 721 ceiling, so a solve given the default looked at its whole
grid. Every answer carries what it spent and whether it ran out:

- `candidates_used`, how many flights were flown
- `candidate_budget`, what it was allowed
- `exhausted`, whether the budget ran out before the grid did

`exhausted` is the field that does the work, and it records a refused claim rather
than a counter that reached the limit. The two differ at the boundary. A budget
that happens to equal the grid size lets the search fly every candidate it wanted,
so its answer really is the best of the whole grid, and a counter would report it
as truncated. False means the answer is the best of everything the search looked
at, which is a claim about the problem. True means it is the best of a prefix,
which is a claim about the budget, and the two are not the same claim and cannot
share one sentence.

The renderer reads all three. `describe_auto_solution` puts the wording on the
status line, so a truncated answer reads "budget reached, best of 260/260
candidates" and a complete one reads "best of the search, 574 candidates". The
Auto Solve button asks for the default budget, so in practice the button shows the
complete form; the truncated form is what a caller that lowers the budget gets, and
it now says so instead of looking identical.

**Considered options:** keeping the wall clock and reading the `timed_out` flag at
the call site, and making the grid size the parameter rather than the budget. The
first fixes the reporting and leaves the nondeterminism, which is the part that
made the search impossible to pin in a test. The second is the same idea stated
less usefully, since a caller asking for a coarse grid gets the answer we already
have at a lower cost and cannot tell the two apart.

## What it costs

The renderer's default scene, a 20 m target at 2800 m walking towards the gun at
40 m/s, spends 574 candidates, which took 3.1 s on this machine against the two
seconds it used to be cut off at. That is slower than before, and the wait is the
price of the answer being the same on every machine.

Truncating that budget to claw the second back was tried and rejected. At 260
candidates the coarse sweep is cut off before it reaches the flight times that
solve this scene, and the button misses by 1131 m where the full grid gets within
22 m. The search runs on its own thread and the window keeps drawing throughout,
so the wait is the button's and not the frame's.

**Consequences:** the cost of a solve is now set by the problem rather than by the
machine, so the runtime of anything that calls `solve_launch` is bounded by
arithmetic instead of by whoever is running it. The status line is longer than it
was and wraps in the control panel; if that becomes a problem the fix is a second
line rather than dropping the budget from the first. The README makes no claim
about how fast the search is, and there was nothing in it to remove.

The search does not hit this scene at any budget, which is what it did before too.
The default elevation clamp and the grid resolution put the best cell 22 m from a
20 m target. Changing that is a question about the search's grid and its
constraints, not about how long it is allowed to run, and this decision leaves it
where it found it.