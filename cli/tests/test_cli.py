"""CLI smoke tests.

Thin on purpose: the pipeline is tested in `agent-core/tests`. What matters here
is that the harness wires up, that a fresh clone can go from reference video to
three videos without any configuration, and that errors exit cleanly instead of
printing a traceback.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from styleloom_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()

W, H, FPS = 240, 426, 24
COLOURS = ["red", "blue", "green", "orange", "purple", "teal"]


@pytest.fixture(autouse=True)
def _no_real_credentials(monkeypatch):
    """Keep the suite hermetic.

    Provider selection defaults to `auto`, so a real key in the developer's shell
    would otherwise silently point these tests at a paid API.
    """
    for var in (
        "ANTHROPIC_API_KEY",
        "STYLELOOM_ANTHROPIC_API_KEY",
        "FAL_KEY",
        "STYLELOOM_FAL_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="module")
def reference_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("cli_ref")
    segments = []
    for i, colour in enumerate(COLOURS):
        seg = d / f"seg{i}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                "-i", f"color=c={colour}:s={W}x{H}:d=1.2:r={FPS}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(seg),
            ],
            check=True,
        )
        segments.append(seg)
    listing = d / "l.txt"
    listing.write_text("\n".join(f"file '{s}'" for s in segments), encoding="utf-8")
    out = d / "reference.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(out)],
        check=True,
    )
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STYLELOOM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("STYLELOOM_WIDTH", str(W))
    monkeypatch.setenv("STYLELOOM_HEIGHT", str(H))
    monkeypatch.setenv("STYLELOOM_FPS", str(FPS))
    monkeypatch.setenv("STYLELOOM_LLM_PROVIDER", "mock")
    monkeypatch.setenv("STYLELOOM_VIDEO_PROVIDER", "mock")
    return tmp_path


# --- inspection commands, no work done ------------------------------------- #


def test_help_lists_the_two_verbs():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run" in result.output
    assert "batch" in result.output


def test_plan_shows_the_pipeline():
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    for stage in ("ingest", "outline", "hook", "storyboard", "render", "assemble", "qc"):
        assert stage in result.output
    assert result.output.index("outline") < result.output.index("hook")


def test_doctor_reports_the_environment(env):
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ffmpeg" in result.output
    assert "video provider" in result.output


def test_version_names_both_distributions():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "styleloom-cli" in result.output
    assert "styleloom-core" in result.output


# --- errors exit, they do not crash ---------------------------------------- #


def test_run_without_input_is_a_clean_error(env):
    result = runner.invoke(app, ["run", "nostyle"])
    assert result.exit_code != 0
    assert "--text" in result.output
    assert "Traceback" not in result.output


def test_run_on_a_missing_style_is_a_clean_error(env):
    result = runner.invoke(app, ["run", "nope", "--text", "무언가"])
    assert result.exit_code != 0
    assert "style not found" in result.output
    assert "Traceback" not in result.output


def test_extract_refuses_to_clobber_without_force(env, reference_video):
    args = ["style", "extract", "demo", str(reference_video)]
    assert runner.invoke(app, args).exit_code == 0
    second = runner.invoke(app, args)
    assert second.exit_code != 0
    assert "already exists" in second.output
    assert runner.invoke(app, [*args, "--force"]).exit_code == 0


# --- the actual deliverable ------------------------------------------------- #


def test_fresh_clone_goes_from_reference_to_three_videos(env, reference_video):
    """The submission, as a single test: extract a style, then push three
    different inputs through the unchanged system."""
    extract = runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    assert extract.exit_code == 0, extract.output

    batch = runner.invoke(
        app,
        [
            "batch", "demo",
            "-t", "회사에서 아무도 안 알려주는 엑셀 단축키",
            "-t", "자취 3년차가 후회하는 가전 구매 순서",
            "-t", "러닝 첫 달에 무릎이 아픈 진짜 이유",
        ],
    )
    assert batch.exit_code == 0, batch.output
    assert "3/3 done" in batch.output

    videos = sorted((env / "data" / "runs").glob("*/final.mp4"))
    assert len(videos) == 3
    assert all(v.stat().st_size > 0 for v in videos)


def test_runs_ls_and_hook_inspect_a_finished_run(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    assert runner.invoke(app, ["run", "demo", "-t", "엑셀 단축키"]).exit_code == 0

    listing = runner.invoke(app, ["runs", "ls"])
    assert listing.exit_code == 0
    run_id = listing.output.split()[0]

    detail = runner.invoke(app, ["runs", "hook", run_id])
    assert detail.exit_code == 0
    assert "SystemRandom" in detail.output
    assert "softmax" in detail.output


def test_hook_preview_reports_its_own_variety(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    result = runner.invoke(app, ["hook", "preview", "demo", "-t", "엑셀 단축키", "-n", "8"])

    assert result.exit_code == 0
    assert "distinct archetypes" in result.output
    assert "distinct texts" in result.output


def test_hook_preview_leaves_no_run_behind(env, reference_video):
    """A preview must not pollute run history or archetype history, or it would
    change the next real run's sampling."""
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    runner.invoke(app, ["hook", "preview", "demo", "-t", "엑셀 단축키", "-n", "4"])

    assert not list((env / "data" / "runs").glob("*/run.json"))
    assert not (env / "data" / "styles" / "demo" / "hook_history.jsonl").exists()


# --- submission bundle ------------------------------------------------------- #


def _batch_of_three(reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    return runner.invoke(
        app,
        ["batch", "demo", "-t", "엑셀 단축키", "-t", "가전 구매 순서", "-t", "러닝 무릎 통증"],
    )


def test_export_pairs_every_video_with_its_original_prompt(env, reference_video):
    """The deliverable is each mp4 alongside the prompt that produced it. Run
    folders hold both but spread across timestamped directories."""
    assert _batch_of_three(reference_video).exit_code == 0
    out = env / "bundle"
    result = runner.invoke(app, ["export", str(out)])
    assert result.exit_code == 0, result.output

    videos = sorted(out.glob("*.mp4"))
    assert len(videos) == 3
    assert all(v.stat().st_size > 0 for v in videos)

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == 3
    for entry in manifest:
        assert entry["input_prompt"]
        assert entry["hook_text"]
        assert entry["creator"] and entry["setting"]
        assert (out / entry["video"]).exists()

    prompts = (out / "prompts.txt").read_text(encoding="utf-8")
    for entry in manifest:
        assert entry["input_prompt"] in prompts


def test_export_is_ordered_by_generation_not_recency(env, reference_video):
    assert _batch_of_three(reference_video).exit_code == 0
    out = env / "bundle"
    runner.invoke(app, ["export", str(out)])

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    run_ids = [e["run_id"] for e in manifest]
    assert run_ids == sorted(run_ids), "01 should be the first video generated"


def test_export_carries_the_per_run_artifacts_but_not_the_shot_scratch(env, reference_video):
    assert _batch_of_three(reference_video).exit_code == 0
    out = env / "bundle"
    runner.invoke(app, ["export", str(out)])

    run_dirs = list((out / "runs").iterdir())
    assert len(run_dirs) == 3
    for d in run_dirs:
        assert (d / "hook.json").exists()
        assert (d / "casting.json").exists()
        # Intermediate shot files are large and regenerable; the bundle stays small.
        assert not (d / "shots").exists()


def test_export_reports_the_variety_across_the_bundle(env, reference_video):
    assert _batch_of_three(reference_video).exit_code == 0
    result = runner.invoke(app, ["export", str(env / "bundle")])
    assert "hook archetypes" in result.output
    assert "creators" in result.output


def test_export_without_any_runs_is_a_clean_error(env):
    result = runner.invoke(app, ["export", str(env / "bundle")])
    assert result.exit_code != 0
    assert "no completed runs" in result.output
    assert "Traceback" not in result.output


# --- cost estimator ---------------------------------------------------------- #


def test_models_prices_the_endpoints_without_a_style(env):
    """Has to say something useful on a fresh clone, before any style exists."""
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "kling" in result.output
    assert "multi_prompt" in result.output
    assert "wasted" in result.output


def test_models_uses_the_styles_own_shot_count(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    result = runner.invoke(app, ["models", "--style", "demo"])
    assert result.exit_code == 0
    assert "style 'demo'" in result.output
    assert "shots at" in result.output


def test_models_marks_the_current_default(env):
    result = runner.invoke(app, ["models"])
    assert "current default" in result.output


def test_models_declines_to_invent_a_missing_rate(env):
    """A guessed price is worse than no price when someone is budgeting."""
    result = runner.invoke(app, ["models"])
    assert "rate not recorded" in result.output


def test_models_on_a_missing_style_is_a_clean_error(env):
    result = runner.invoke(app, ["models", "--style", "nope"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


# --- style inspection -------------------------------------------------------- #
#
# These exist because `style hooks` shipped broken: after the history log grew a
# `kind` field, the command passed `limit` into the `kind` parameter and read
# fields that no longer existed. It did not crash -- it reported "no history" on a
# style that had six entries. Nothing tested it, so nothing caught it.


def test_style_history_lists_every_kind(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    assert runner.invoke(app, ["run", "demo", "-t", "엑셀 단축키"]).exit_code == 0

    result = runner.invoke(app, ["style", "history", "demo"])
    assert result.exit_code == 0
    for kind in ("hook", "creator", "setting"):
        assert kind in result.output, f"{kind} missing from history output"
    assert "no history" not in result.output


def test_style_history_filters_by_kind(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    runner.invoke(app, ["run", "demo", "-t", "엑셀 단축키"])

    result = runner.invoke(app, ["style", "history", "demo", "--kind", "hook"])
    assert result.exit_code == 0
    assert "creator" not in result.output
    assert "setting" not in result.output


def test_style_history_on_an_unused_style_says_so(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    result = runner.invoke(app, ["style", "history", "demo"])
    assert result.exit_code == 0
    assert "no history" in result.output


def test_style_ls_shows_recent_hooks(env, reference_video):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    runner.invoke(app, ["run", "demo", "-t", "엑셀 단축키"])
    result = runner.invoke(app, ["style", "ls"])
    assert result.exit_code == 0
    assert "demo" in result.output


def test_style_show_round_trips_through_set(env, reference_video, tmp_path):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    shown = runner.invoke(app, ["style", "show", "demo"])
    assert shown.exit_code == 0

    edited = tmp_path / "edited.json"
    edited.write_text(shown.output, encoding="utf-8")
    assert runner.invoke(app, ["style", "set", "demo", str(edited)]).exit_code == 0


def test_style_set_rejects_a_mismatched_id(env, reference_video, tmp_path):
    runner.invoke(app, ["style", "extract", "demo", str(reference_video)])
    shown = runner.invoke(app, ["style", "show", "demo"])
    other = tmp_path / "other.json"
    other.write_text(shown.output, encoding="utf-8")

    result = runner.invoke(app, ["style", "set", "someone_else", str(other)])
    assert result.exit_code != 0
    assert "mismatch" in result.output


def test_every_registered_command_has_working_help():
    """A blanket guard: a command whose signature no longer matches its
    implementation fails here rather than in someone's terminal."""
    groups = ["style", "runs", "hook"]
    top = ["run", "batch", "export", "models", "plan", "doctor", "version"]

    for name in top:
        assert runner.invoke(app, [name, "--help"]).exit_code == 0, name
    for group in groups:
        listing = runner.invoke(app, [group, "--help"])
        assert listing.exit_code == 0, group
