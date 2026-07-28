"""The README is checked, not trusted.

Documentation drifted three times while this project was built: a renamed command
stayed documented under its old name, a command that had silently broken kept its
entry, and the test count went stale twice. None of it was caught, because nothing
read the README.

These tests are cheap and mechanical. They do not check prose — only the claims
that are derivable from code: which commands exist, which settings exist, what the
defaults are, what the pipeline stages are, and whether every path it points at is
real.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from styleloom_cli.main import app
from styleloom_core import Settings, build_plan
from styleloom_core.session import ARTIFACT_FILES

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"

# Rows where the README groups related variables into one line rather than
# repeating the prefix. Listed explicitly so the grouping is a deliberate choice
# and not a hole in the check.
ABBREVIATED_IN_README = {
    "STYLELOOM_HEIGHT",
    "STYLELOOM_FPS",
    "STYLELOOM_HOOK_SOFTMAX_TEMP",
}


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def real_commands() -> set[str]:
    found: set[str] = set()

    def walk(typer_app, prefix: str = "") -> None:
        for command in typer_app.registered_commands:
            name = command.name or command.callback.__name__.replace("_", "-")
            found.add(f"{prefix}{name}")
        for group in typer_app.registered_groups:
            walk(group.typer_instance, prefix=f"{group.name} ")

    walk(app)
    return found


def test_every_command_is_documented(readme):
    missing = sorted(c for c in real_commands() if f"styleloom {c}" not in readme)
    assert not missing, f"undocumented commands: {missing}"


def test_no_documented_command_is_imaginary(readme):
    """The failure that shipped: `style history` was documented while the code
    still exposed `style hooks`."""
    real = real_commands()
    groups = {c.split()[0] for c in real if " " in c}

    imaginary = []
    for match in re.finditer(r"`styleloom ([a-z]+)(?: ([a-z]+))?", readme):
        head, sub = match.group(1), match.group(2)
        if head in groups:
            if sub and f"{head} {sub}" not in real:
                imaginary.append(f"{head} {sub}")
        elif head not in real:
            imaginary.append(head)

    assert not imaginary, f"documented but not implemented: {sorted(set(imaginary))}"


def test_every_setting_is_documented(readme):
    documented = set(re.findall(r"STYLELOOM_[A-Z0-9_]+", readme)) | ABBREVIATED_IN_README
    real = {f"STYLELOOM_{name.upper()}" for name in Settings.model_fields}
    missing = sorted(real - documented)
    assert not missing, f"undocumented settings: {missing}"


def test_no_documented_setting_is_imaginary(readme):
    documented = set(re.findall(r"STYLELOOM_[A-Z0-9_]+", readme))
    real = {f"STYLELOOM_{name.upper()}" for name in Settings.model_fields}
    imaginary = sorted(documented - real)
    assert not imaginary, f"settings that do not exist: {imaginary}"


@pytest.mark.parametrize(
    "field",
    ["llm_model", "fal_t2i_model", "fal_i2v_model", "render_mode", "llm_provider"],
)
def test_quoted_defaults_match_the_code(readme, field):
    """Stale defaults are worse than absent ones: `claude-sonnet-4-6` sat in the
    README after the model string had moved on."""
    assert str(getattr(Settings(), field)) in readme


def test_pipeline_stages_match_the_plan(readme):
    for step in build_plan().steps:
        assert step in readme, f"stage {step!r} missing from the README"


def test_artifact_filenames_match_the_session(readme):
    for filename in ARTIFACT_FILES.values():
        assert filename in readme, f"artifact {filename!r} missing from the README"


def test_linked_docs_exist(readme):
    for relative in re.findall(r"\((docs/[A-Za-z_]+\.md)\)", readme):
        assert (REPO_ROOT / relative).exists(), f"broken link: {relative}"


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "configs/archetypes.yaml",
        "configs/casting.yaml",
        "configs/fal_models.yaml",
        "agent-core",
        "cli",
    ],
)
def test_referenced_paths_exist(readme, path):
    assert path in readme, f"{path} should be mentioned in the README"
    assert (REPO_ROOT / path).exists(), f"README points at missing path: {path}"


def test_the_test_count_is_current(readme):
    """Went stale twice (106, then 137). If this fails, update the number."""
    claimed = re.search(r"(\d+) tests, no network", readme)
    assert claimed, "the README should state how many tests there are"

    collected = 0
    for test_file in REPO_ROOT.glob("*/tests/test_*.py"):
        source = test_file.read_text(encoding="utf-8")
        collected += len(re.findall(r"^def test_", source, flags=re.MULTILINE))

    # Parametrised cases expand at collection time, so the function count is a
    # lower bound rather than the exact total.
    assert collected <= int(claimed.group(1)), (
        f"README claims {claimed.group(1)} tests but there are at least {collected} "
        "test functions -- the number is stale"
    )


def test_env_example_matches_the_code_defaults():
    """`.env.example` is copied to `.env` verbatim, so a stale line here does not
    merely document the wrong default -- it overrides the right one.

    Both drifts this catches were live: `STYLELOOM_LLM_MODEL` still said
    `claude-sonnet-4-6`, and `STYLELOOM_FAL_I2V_MODEL` pointed at a Seedance
    endpoint, which silently gave up `elements` and with it the creator
    consistency the default exists to provide. The README was checked against the
    code and stayed correct through the same period. This file was not checked.
    """
    defaults = Settings()
    drift: list[str] = []
    for line in (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(STYLELOOM_[A-Z0-9_]+)=(.*)$", line.strip())
        if not match:
            continue
        var, raw = match.groups()
        field = var.removeprefix("STYLELOOM_").lower()
        assert field in type(defaults).model_fields, f"{var} is not a real setting"
        # Compared after coercion, not as strings: the file writes `600` for a
        # float field, which is the same value and not drift.
        actual = getattr(defaults, field)
        if getattr(Settings(**{field: raw}), field) != actual:
            drift.append(f"{var}={raw!r}, code default is {str(actual)!r}")
    assert not drift, "stale .env.example values: " + "; ".join(drift)
