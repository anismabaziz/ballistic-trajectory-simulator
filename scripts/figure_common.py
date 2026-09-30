"""What every figure generator needs to say and do around its own drawing.

The three generators differ in what they draw and share everything else: the
atmosphere they flew through, a single output-path argument, and a banner of
launch conditions under the figure. Those live here so a figure cannot state one
launch and get regenerated with another.

This is a script-directory module rather than one in the package because it is
about figures. Nothing in `ballistics` should know what a figure caption reads
like.
"""

import argparse

from ballistics import config


def describe_atmosphere(extra_lines=()):
    """The physical conditions a figure was produced under.

    The wind is a table rather than a number, so it is reported as the surface
    value and the top-of-table value. Printing the whole list as though it were a
    single speed would misstate the conditions the figure was produced under,
    which is the one job this text has.
    """
    return "\n".join(
        [
            f"m = {config.MASS:.0f} kg   Cd = {config.CD}   A = {config.AREA} m2   "
            f"rho = {config.RHO} kg/m3   g = {config.G} m/s2",
            f"wind x = {config.WIND_X[0]:.0f} m/s at {config.ALT_LEVELS[0]:.0f} m, "
            f"{config.WIND_X[-1]:.0f} m/s at {config.ALT_LEVELS[-1]:.0f} m   "
            f"wind z = {config.WIND_Y[0]:.0f} to {config.WIND_Y[-1]:.0f} m/s   "
            f"{len(config.ALT_LEVELS)} levels, linear in altitude",
            *extra_lines,
        ]
    )


def run(main, default_output, description):
    """The tail every generator shares: read the output path, draw there."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output", default=default_output, help="where to write the figure")
    main(parser.parse_args().output)
