"""Redraw every figure the README shows, in one command.

Each figure in the repository is produced by a script in this directory, and each
of those scripts is named in `GENERATORS` below. The output path is read off the
generator rather than written out again here, so moving a figure cannot leave
this command writing to a stale location: the list and the script that draws the
figure are the same fact.

Every generator states the launch conditions it flew, so a run says what produced
the picture rather than only writing it.

Run it with:

    uv run python scripts/make_figures.py

To see where the figures go without drawing anything:

    uv run python scripts/make_figures.py --list
"""

import argparse
import importlib
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent

# The generators, in the order their figures are drawn. Each module names its own
# `OUTPUT_PATH`, which is where the figure goes; the suite checks the two against
# each other so neither can drift from the other.
GENERATORS = [
    "integrator_convergence",
    "salvo_dispersion",
    "intercept",
]


def load(module_name):
    """Import a generator from this directory, which is not a package."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(module_name)


def output_paths():
    """Where each figure goes, read off the generator that draws it."""
    return [load(module_name).OUTPUT_PATH for module_name in GENERATORS]


def main(list_only=False):
    if list_only:
        for output_path in output_paths():
            print(output_path)
        return

    for module_name in GENERATORS:
        print(f"=== {module_name} ===")
        generator = load(module_name)
        generator.main(generator.OUTPUT_PATH)


def build_parser():
    parser = argparse.ArgumentParser(description="Regenerate every figure the README shows")
    parser.add_argument(
        "--list",
        action="store_true",
        help="print where each figure goes and draw nothing",
    )
    return parser


if __name__ == "__main__":
    main(list_only=build_parser().parse_args().list)
