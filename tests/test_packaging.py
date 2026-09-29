import ast
import json
import os
import subprocess
import sys
import textwrap
import tomllib
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISTRIBUTION = "ballistic-trajectory-simulator"

# The plotting and renderer packages a consumer of the ballistics library should
# not have to import. Listed here so the expectation is in one place.
PLOTTING_AND_RENDERER = ("matplotlib", "pygame", "sim")


def installed_distribution():
    """The installed distribution, ignoring the egg-info a build leaves in the tree.

    Running from the repository root puts that egg-info on sys.path ahead of
    site-packages. It describes the same distribution but records no install, so
    metadata questions about the install have to skip past it. `.dist-info` is
    the shape an installer writes; the private `_path` attribute is the only way to tell
    the two apart, since `metadata.distributions()` gives no public accessor.
    """
    candidates = [
        d
        for d in metadata.distributions()
        if d.metadata["Name"].lower().replace("_", "-") == DISTRIBUTION
        and Path(str(d._path)).name.endswith(".dist-info")
    ]
    return candidates[0] if candidates else None


def declared_dependencies():
    distribution = installed_distribution()
    if distribution is None:
        return set()
    return {Requirement(r).name.lower().replace("_", "-") for r in distribution.requires or []}


def declared_modules():
    """Top-level module and package names the install makes importable.

    Read from the installed metadata rather than hardcoded, so a move into the
    ballistics package does not silently stop this file from covering the code
    that moved.
    """
    distribution = installed_distribution()
    if distribution is None:
        return set()
    return set((distribution.read_text("top_level.txt") or "").split())


def project_modules():
    """Module and package names the packaging config claims to ship, read from pyproject.

    `packages.find.include` holds glob patterns rather than names, so they are
    resolved against the tree to get the packages setuptools will actually find.
    """
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    setuptools_config = config["tool"]["setuptools"]

    found = set()
    for pattern in setuptools_config["packages"]["find"]["include"]:
        found.update(
            path.name for path in PROJECT_ROOT.glob(pattern.strip("*")) if (path / "__init__.py").exists()
        )

    return set(setuptools_config["py-modules"]) | found


def third_party_imports():
    """Top-level modules the shipped code imports that come from a wheel, not the stdlib."""
    own = declared_modules() | set(PLOTTING_AND_RENDERER)
    imported = set()

    for name in own:
        path = PROJECT_ROOT / f"{name}.py"
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])

    return sorted(imported - own - sys.stdlib_module_names)


def run_in_unrelated_directory(code, tmp_path):
    """Run a snippet from a directory with no claim on the project.

    PYTHONPATH is cleared so an exported repository root cannot stand in for the
    install and make the test pass for the wrong reason.
    """
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def test_the_project_is_installed_in_editable_mode():
    distribution = installed_distribution()
    assert distribution is not None, f"run `uv sync` in {PROJECT_ROOT}"

    direct_url = distribution.read_text("direct_url.json")
    assert direct_url is not None, "expected an editable install, found a copied one"
    assert json.loads(direct_url)["dir_info"]["editable"] is True
    assert json.loads(direct_url)["url"].removeprefix("file://") == str(PROJECT_ROOT)


def test_the_install_ships_exactly_the_modules_the_packaging_config_declares():
    assert declared_modules() == project_modules()


def test_the_ballistics_library_imports_from_an_unrelated_directory(tmp_path):
    result = run_in_unrelated_directory(
        """
        from physics import BallisticPhysics
        from targets import Target, check_collision
        from utils import find_launch_angle, solve_interceptor_angle, solve_moving_target_angle

        print(BallisticPhysics.__module__, Target.__module__, find_launch_angle.__module__)
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["physics", "targets", "utils"]


def test_the_physics_and_target_interfaces_import_without_the_plotting_or_renderer_code(tmp_path):
    """Covers physics and targets only, and the docstring says why.

    The three searches are not in this test because they share `utils` with the
    four matplotlib plotting functions, so importing a search still costs a
    matplotlib import. Separating them widens this to the searches, and moving
    the ballistics code into a package widens it further.
    """
    result = run_in_unrelated_directory(
        f"""
        import sys

        from physics import BallisticPhysics
        from targets import Target, check_collision

        unwanted = {PLOTTING_AND_RENDERER!r}
        print(",".join(sorted(name for name in unwanted if name in sys.modules)))
        """,
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_every_third_party_import_is_shipped_by_a_declared_dependency():
    """Every wheel-supplied import must be accounted for by a declared dependency.

    Checks the import against the modules the declared distributions actually
    provide, rather than looking the import up and skipping when no installed
    distribution claims it. An import from an undeclared dependency has no
    provider here, so it fails, which is the case worth catching.
    """
    declared = declared_dependencies()
    assert declared, "nothing is declared, so nothing can be accounted for"
    provided = set()
    for name in declared:
        for module in metadata.packages_distributions().get(name.replace("-", "_"), []):
            provided.add(module.lower().replace("_", "-"))
        provided.add(name)

    unaccounted = [module for module in third_party_imports() if module.lower() not in provided]
    assert unaccounted == [], f"imports no declared dependency ships: {unaccounted}"


def test_the_gif_writer_is_declared_directly_rather_than_inherited():
    """`anim.save(..., writer="pillow")` names the writer, so no import line declares it.

    pillow is also a matplotlib dependency, so it resolves either way. Declaring
    it is what makes a future matplotlib that drops it a loud failure here rather
    than a broken GIF export at the end of an animation run.
    """
    assert "pillow" in declared_dependencies()
