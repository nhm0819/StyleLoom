# StyleLoom

Decompose a reference short-form video into a reusable **style schema**, then
generate new videos in that style from arbitrary inputs.

The deliverable is not one good video. It is a system where *different inputs
through the same unchanged pipeline* produce different videos that still read as
the same channel.

---

## Install

### Prerequisites

Two things are installed outside pip, because they are binaries rather than Python
packages. Neither is optional: ffmpeg does every render, probe, caption burn and
concat, and without a font containing Hangul the caption filter is skipped and you
get a silent, uncaptioned video.

```bash
# Debian, Ubuntu
sudo apt-get install -y ffmpeg fonts-noto-cjk

# macOS
brew install ffmpeg   # 맑은 고딕 has no macOS equivalent; see the note below
```

```powershell
# Windows
winget install -e --id Gyan.FFmpeg
```

Windows needs no font install — Malgun Gothic (`malgunbd.ttf`) ships with the OS and
is in the lookup list. What it does need is a **new terminal**: winget edits `PATH`,
and a process already running does not see the change. If you launch tests from an
editor, restart the editor itself — reloading the window is not enough, because the
child process inherits the old environment. winget also has an intermittent bug
where it does not register `PATH` at all, in which case add the `...\bin` directory
by hand.

On macOS no bundled font covers the lookup list, so install one and add its path to
`FONT_CANDIDATES` in `agent-core/styleloom_core/media.py` — Apple SD Gothic Neo
lives outside the searched locations and is not currently listed.

Verify before anything else:

```bash
ffmpeg -version   # in a new terminal
```

### The package

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -e agent-core -e cli
styleloom doctor
```

`python -m pip` rather than bare `pip` on purpose. Both distributions are editable
installs, and an editable install binds to whichever interpreter's `site-packages`
pip happens to belong to. If that is not the interpreter you later run, the install
succeeds and `import styleloom_core` still fails. `python -m pip` makes the two the
same by construction. See [Import errors](#import-errors) if you hit it anyway.

`doctor` reports both prerequisites by resolved path, so it is the fastest way to
tell an environment problem from a code problem. It also prints which providers
resolved, and why:

```
ok    ffmpeg      /usr/bin/ffmpeg
ok    CJK font    /usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc
```

No API keys are required. The defaults fall back to an offline LLM and an
ffmpeg-only renderer, so the whole pipeline runs end to end on a fresh clone. See
[API keys](#api-keys) to plug in real models.

---

## Docker

```bash
docker build -t styleloom .
docker run --rm styleloom doctor
```

`ENTRYPOINT` is the CLI, so arguments go straight through and read as the command
they wrap:

```bash
docker run --rm \
  -v "$PWD/.env:/app/.env:ro" \
  -v "$PWD/data:/app/data" \
  styleloom run my_style --text "이 제품 소개해줘"
```

**`.env` is a bind mount, not `--env-file`.** `Settings` declares
`env_file=".env"` and reads it relative to the process cwd, which the image fixes
at `/app` — so mounting the file lets pydantic-settings parse it, comments and all,
exactly as it does outside a container. `--env-file` also works but hands the file
to Docker's parser instead, which has different quoting rules. Real environment
variables take precedence over the file either way, so `-e ANTHROPIC_API_KEY` still
overrides:

```bash
docker run --rm -e ANTHROPIC_API_KEY -e KLING_ACCESS_KEY -e KLING_SECRET_KEY \
  -v "$PWD/data:/app/data" \
  styleloom run my_style --text "..."
```

**Mount `data/` or the output is lost.** Runs, styles and the choice history all
live under `data/`, inside the container, and `--rm` deletes them with it. The
mount is also how `style extract` finds reference videos — put them in
`data/uploads/` on the host and pass the filename, which resolves to the mounted
copy inside the container:

```bash
cp reference1.mp4 data/uploads/
docker run --rm -v "$PWD/data:/app/data" styleloom \
  style extract my_style reference1.mp4
```

Anything outside the mount is invisible to the container, so a host path that the
same command resolves fine outside Docker will not be found here.

Three things the image settles on purpose:

- **`WORKDIR /app` is load-bearing.** `data_dir` defaults to the relative
  `Path("data")` and `env_file` to `.env`, both resolved against cwd. Any other
  working directory writes runs elsewhere and stops reading `.env`, neither with an
  error.
- **The install is editable.** `config.py` derives `REPO_ROOT` from
  `Path(__file__).resolve().parents[2]`, which `resolve_config()` uses to fall back
  to the bundled `configs/*.yaml`. Under a regular site-packages install that
  expression lands on `/usr/local/lib/python3.12` and the fallback breaks;
  editable keeps it at `/app`.
- **It runs as UID 1000, not root**, so bind-mounted output is not root-owned on
  the host. If your UID differs, pass `--user "$(id -u):$(id -g)"`.

Two apt packages and no more: `ffmpeg`, which every render and probe shells out to,
and `fonts-noto-cjk`, without which `drawtext` renders Korean captions as empty
boxes and reports no error. `opencv-python-headless` needs nothing beyond what
`slim` already ships — which is why the dependency is pinned to the headless wheel
rather than `opencv-python`, whose libGL, libX11 and Qt5 requirements would all
have to be installed here.

---

## Walkthrough

### 1. Extract a style from reference videos

Drop the references in `data/uploads/` and pass the filenames:

```bash
cp ~/Downloads/ref01.mp4 ~/Downloads/ref02.mp4 data/uploads/
styleloom style extract my_style ref01.mp4 ref02.mp4
```

Each argument is looked up in `<data-dir>/uploads/` first, and used as a path when
it is not there — so an absolute path or one relative to the working directory
keeps working, and the two forms can be mixed in one command:

```bash
styleloom style extract my_style ref01.mp4 /Volumes/footage/ref02.mp4
```

Writes `data/styles/my_style/style.json`. Pacing, colour and the hook window are
measured from pixels; only the qualitative names — grade, camera vocabulary, tone —
come from a model.

If the extractor mislabels something, edit the file and put it back. The schema is
the contract, not the extractor:

```bash
styleloom style show my_style > edited.json
styleloom style set my_style edited.json
```

### 2. Generate three videos from three different inputs

```bash
styleloom batch my_style \
  -t "회사에서 아무도 안 알려주는 엑셀 단축키" \
  -t "자취 3년차가 후회하는 가전 구매 순서" \
  -t "러닝 첫 달에 무릎이 아픈 진짜 이유"
```

Each run prints its stages, the hook it generated, its QC score and the path to
`final.mp4`. The batch then reports how much variety it achieved.

One input at a time, with an image or a video instead of text. `--file`, `--bgm`
and `--persona` take the same lookup as a reference video — `<data-dir>/uploads/`
first, then the path as given:

```bash
cp product.jpg clip.mp4 data/uploads/
styleloom run my_style --file product.jpg --text "이 제품 소개해줘"
styleloom run my_style --file clip.mp4
styleloom run my_style --file /abs/path/clip.mp4        # still fine
```

All three are checked before the run directory is created, so a typo in `--bgm`
fails immediately instead of after the renders have been paid for. `inputs.json`
records the resolved path, not the argument.

### 3. Prove the hook is not frozen

```bash
styleloom hook preview my_style -t "엑셀 단축키" -n 8
```

Same input, eight generations, no rendering. Reports `distinct archetypes` and
`distinct texts` directly, so the non-determinism requirement is checkable in one
command rather than taken on faith. A preview writes nothing to disk, so it cannot
skew the next real run.

To see how a finished run decided:

```bash
styleloom runs ls
styleloom runs hook <run_id>       # pool, sampled archetype, candidates, scores
styleloom style history my_style   # what this style used recently, per element
```

### 4. Check what a run will cost

```bash
styleloom models --style my_style
```

Prices every endpoint in `configs/fal_models.yaml` against that style's own shot
count, and shows what `multi_shot` mode would save. Arithmetic only — no API calls.

### 5. Collect the submission bundle

```bash
styleloom export bundle/
```

Pairs each video with the exact prompt that produced it, in generation order:

```
bundle/
├── 01_run_....mp4  02_run_....mp4  03_run_....mp4
├── prompts.txt          input + hook + cast + qc, plain text
├── manifest.json        the same, machine-readable
└── runs/<run_id>/       per-stage artifacts for each video
```

---

## Command reference

| Command | What it does |
|---|---|
| `styleloom style extract <id> <refs...>` | Reference videos → `style.json`. Refs resolve against `<data-dir>/uploads/` first, then as given. `--force` to overwrite |
| `styleloom style ls` | Saved styles, with each one's recent hook archetypes |
| `styleloom style show <id>` | Print `style.json` |
| `styleloom style set <id> <file>` | Replace it with a hand-edited copy (validated) |
| `styleloom style history <id>` | Recent hook / creator / setting choices. `--kind` to filter |
| `styleloom run <id>` | One input → one video. `--text`, `--file`, `--bgm`, `--persona`, `--lang`. File options resolve against `<data-dir>/uploads/` first, then as given |
| `styleloom batch <id>` | Several inputs through one system. Repeat `-t` / `-f`, or `--inputs-file` |
| `styleloom runs ls` | Recent runs, status, hook, QC score. `--style` to filter |
| `styleloom runs show <run_id>` | The run record |
| `styleloom runs hook <run_id>` | The full hook decision trail for that run |
| `styleloom hook preview <id>` | N hook generations from one input, no rendering |
| `styleloom models` | Endpoint specs and per-video cost. `--style` for real numbers |
| `styleloom export <dir>` | Submission bundle. `--run` to pick runs, `-n` for how many |
| `styleloom plan` | The pipeline and each stage's input/output artifacts |
| `styleloom doctor` | ffmpeg, fonts, providers, config paths, render mode |
| `styleloom version` | Both distribution versions |

`--data-dir`, `--llm`, `--video` and `--render-mode` override settings for a single
invocation.

---

## Pipeline

```bash
$ styleloom plan
1. ingest      (- -> brief)
2. casting     (style, brief -> casting)
3. outline     (style, brief -> outline)
4. hook        (style, brief, outline -> hook)
5. storyboard  (style, brief, casting, outline, hook -> storyboard)
6. render      (storyboard, casting -> render)
7. assemble    (style, storyboard, render -> assemble)
8. qc          (style, storyboard, render, assemble -> qc)
```

Reference videos go through a separate entry point (`style extract`) because
`style.json` is a reusable asset, not a per-run artifact.

Every stage is `JSON in → JSON out` and writes its artifact to `data/runs/<run_id>/`:

```
data/runs/<run_id>/
├── run.json          status, stage, hook, qc score
├── inputs.json       what the caller supplied
├── style_used.json   snapshot of the style this run actually ran on
├── brief.json        normalised input
├── casting.json      creator + background, and the pools they came from
├── outline.json      body beats + payoff
├── hook.json         pool, sampled archetype, every candidate, selection method
├── storyboard.json   per-shot prompts, sizes, durations, captions
├── render.json       clip segments and per-shot errors
├── assemble.json     final video path
├── qc_report.json    measured conformance to the style
└── final.mp4
```

A finished run is fully inspectable; a failed run shows exactly which stage broke
and leaves the earlier artifacts intact.

The step order is data, and it is validated before execution: a stage that reads an
artifact nothing earlier writes fails at plan time with the artifact named, rather
than at render time after money has been spent.

### What changes between the three videos

The assignment asks that outputs differ in more than styling. Three elements are
drawn per run, all by the system and none from the user's input:

| Element | Pool | Reaches the output as |
|---|---|---|
| **Hook** (required) | `configs/archetypes.yaml` | The first 3 seconds: on-screen text and opening visual |
| **Creator** (recommended) | `configs/casting.yaml` → `creators` | Presenter tokens leading every shot prompt, plus a generated reference portrait where the endpoint supports one |
| **Background** (recommended) | `configs/casting.yaml` → `settings` | Location tokens in every shot prompt |

All three use the same draw: weighted sample, penalised by what this style used in
its recent runs, entropy from `secrets.SystemRandom`. `styleloom style history`
shows the window that penalty is reading.

### Why the hook comes *after* the outline

A hook is a promise the body has to pay off. Generating it first produces a generic
attention grabber; generating it fourth lets it reference the actual payoff. The
hook tool receives `outline.payoff` explicitly and is instructed never to bait
something the video does not deliver.

### How the hook varies across reruns

Temperature alone would make variation an accident. Three independent randomisation
points instead:

| Point | Mechanism | Effect |
|---|---|---|
| Archetype | weighted draw with a recency penalty over this style's last 4 runs | a *structurally* different opening, not a paraphrase |
| Candidates | N generations, temperature 0.9, no seed | lexical variety within the archetype |
| Selection | top-k softmax sample instead of argmax | the best candidate is favoured, not guaranteed |

Every decision is written to `hook.json`: the pool, the sampled archetype, all
candidates with scores, the selection method, and the entropy source.

**The recency penalty biases, it does not forbid.** At the default of 0.35 a
recently used archetype's weight is multiplied rather than removed — measured over
4,000 draws on the bundled six-archetype pool, that takes one archetype from ~17%
to ~7%. So a batch of three usually gets three different archetypes and sometimes
two; the hook *text* differs every time. For guaranteed variety set
`STYLELOOM_HOOK_RECENCY_PENALTY=0`, which makes the exclusion hard.

### Render mode: `per_shot` or `multi_shot`

Image-to-video bills a minimum clip length — 3s on Kling v3 — while short-form
cuts run 1–2s. There are two ways to live with that, and the choice is yours:

```bash
styleloom run my_style -t "..." --render-mode multi_shot
```

| | `per_shot` (default) | `multi_shot` |
|---|---|---|
| Calls | one per cut | one per ~15s window |
| Billed | the floor, every cut | the delivered length |
| 30s / 20 cuts on Kling v3 Pro | $6.72, half discarded | **$3.36, nothing discarded** |
| Cut timing | exact — ffmpeg trims each file | **the model's** — requested, then measured |
| One failure costs | one shot | a whole window |
| Captions | one per clip | placed by timestamp inside the clip |

`per_shot` is the default because shot durations are exact by construction, and
pacing is the property the style schema exists to reproduce. `multi_shot` is roughly
half the cost and asks the endpoint to cut where told — `qc` reports
`cut_timing_drift` so you can tell whether it complied.

**Not verified against a paid endpoint.** Offline the sequence renderer is ffmpeg,
which cuts exactly where told, so drift is zero in both modes and only the plumbing
is proven. Before committing to `multi_shot`, run one window on a real endpoint and
read `cut_timing_drift` — see [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md).

### Quality control

`qc` re-probes the finished video with the *same* measurement used on the reference
and diffs the numbers. Using one function for both sides matters: a separate
verifier can be wrong on its own.

| Check | Target | Tolerance |
|---|---|---|
| `avg_shot_sec` | `style.pacing.avg_shot_sec` | ±0.6s |
| `total_duration` | `style.total_duration` | ±6.0s |
| `saturation` / `contrast` / `warmth` | `style.look` | ±0.18 / ±0.20 / ±0.12 |
| `hook_window_shots` | `style.hook_style.cut_count` | ±0.5 |
| `cut_timing_drift` | 0 — requested cuts vs cuts actually present | ±0.35s |

The score is the share of checks that pass; ≥0.7 is a pass. Tolerances sit where a
viewer stops noticing, not where the offline renderer happens to pass.

---

## Structure

```
StyleLoom/
├── agent-core/                distribution: styleloom-core
│   └── styleloom_core/          framework-independent library
│       ├── schema.py              data contracts
│       ├── config.py              Settings, env-driven, injected not global
│       ├── context.py             settings + providers + stores + event sink
│       ├── events.py              Event, EventSink protocol
│       ├── errors.py              error taxonomy
│       ├── media.py               ffmpeg + OpenCV primitives
│       ├── sampling.py            weighted draw with recency penalty
│       ├── providers/             LLM and video adapters
│       ├── memory/                StyleStore, RunStore, ChoiceHistory
│       ├── session/               per-run state and artifact bag
│       ├── tools/                 one module per stage, plus the registry
│       ├── planner/               builds and validates the execution order
│       └── runner/                executes a plan, emits events
├── cli/                       distribution: styleloom-cli
│   └── styleloom_cli/
│       ├── main.py                typer app
│       ├── console.py             renders core events to a terminal
│       ├── options.py             shared flags, core errors → clean exits
│       └── commands/              style, run, runs, hook, models, export
├── configs/                   archetypes, casting pools, endpoint specs (data)
├── docs/
└── data/                      styles/ runs/ uploads/
```

Two distributions rather than one package, because "framework independent" is only
enforceable as a dependency boundary: `styleloom-core` cannot import typer because
typer is not among its dependencies. A test reads the installed metadata to keep it
that way.

### Using the core directly

The CLI is one caller. The library is the product:

```python
from styleloom_core import RunInputs, Settings, build_context, run_batch

ctx = build_context(Settings())
records = run_batch(ctx, "my_style", [RunInputs(text=t) for t in topics])
```

`run_once` / `run_batch` / `extract_style` are the whole surface for producing
video. An API route or a queue consumer would call the same functions and swap the
event sink — which is what makes them additive rather than a rewrite.

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — data flow diagram, layering, and
  the API/worker extension path
- [docs/TOOL_RATIONALE.md](docs/TOOL_RATIONALE.md) — what each tool is and why,
  including the alternatives that were rejected and the model cost comparison
- [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md) — assumptions, where this got
  stuck, what was measured versus what remains unverified, and how to make the
  system judge its own output

---

## Retargeting to your own references

Nothing about a specific reference video is in the code:

1. Put your references in `data/uploads/`, then
   `styleloom style extract <id> ref01.mp4 ref02.mp4` → a new `style.json`.
   Paths from anywhere else work too; the uploads directory is only the default
   lookup.
2. Edit `configs/archetypes.yaml` and `configs/casting.yaml`, or point
   `STYLELOOM_ARCHETYPES_PATH` / `STYLELOOM_CASTING_PATH` at your own files. The
   bundled casting pools are written for a beauty/skincare channel; a different
   vertical wants different creators and locations.
3. Adding a video model means adding an entry to `configs/fal_models.yaml` — its
   parameter names, duration floor and output key — not editing the provider.

---

## API keys

Copy `.env.example` to `.env`, or just export the variables. **No key is hardcoded
anywhere**, and the repo runs end to end without any.

| Variable | Where to get it | What it is used for | Without it |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API keys | Brief extraction, outline, hook candidates, naming the reference's grade and tone | Offline `mock` LLM: real structure, placeholder wording |
| `KLING_ACCESS_KEY` + `KLING_SECRET_KEY` | [kling.ai/dev](https://kling.ai/dev) → console → API keys | Keyframes (Kling Image) and image-to-video (Kling Video) | ffmpeg `mock` renderer: real MP4s, gradient footage |

All three are also accepted as `STYLELOOM_ANTHROPIC_API_KEY`, `STYLELOOM_KLING_ACCESS_KEY` and `STYLELOOM_KLING_SECRET_KEY`, which
win if both forms are set.

**Providers default to `auto`**: each uses the real service when its key is present
and the offline mock when it is not. Injecting a key is the only step — there is no
second variable to remember. Every run prints its resolution before doing any work:

```
providers: llm=anthropic (auto) video=kling (auto) render=per_shot
```

Real video needs no extra install. The Kling provider signs its own JWT and posts
with `httpx`, both already dependencies — there is no SDK to add.

To pin a provider regardless of keys — offline testing on a keyed machine — set
`STYLELOOM_LLM_PROVIDER=mock` or `STYLELOOM_VIDEO_PROVIDER=mock`.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `STYLELOOM_DATA_DIR` | `data` | Styles, runs and uploads live here |
| `STYLELOOM_LLM_PROVIDER` | `auto` | `auto` \| `mock` \| `anthropic` |
| `STYLELOOM_LLM_MODEL` | `claude-sonnet-5` | |
| `STYLELOOM_VIDEO_PROVIDER` | `auto` | `auto` \| `mock` \| `kling` |
| `STYLELOOM_KLING_T2I_MODEL` | `kling-v3` | a `model_name` sent in the body, not a URL segment. Must be a key in `configs/kling_models.yaml` |
| `STYLELOOM_KLING_I2V_MODEL` | `kling-v3` | ditto. `styleloom models` compares the alternatives |
| `STYLELOOM_KLING_MODE` | `std` | `std` \| `pro`. The quality tier, which is a request field here rather than a separate endpoint |
| `STYLELOOM_KLING_BASE_URL` | `https://api-singapore.klingai.com` | International account system. `api-beijing.klingai.com` is a different account, not a region |
| `STYLELOOM_KLING_TIMEOUT_SEC` | `900` | Covers queue time as well as inference |
| `STYLELOOM_KLING_POLL_INTERVAL_SEC` | `5` | Generation is asynchronous |
| `STYLELOOM_KLING_MAX_CONCURRENCY` | `1` | An account-tier property, so it cannot be recorded per model |
| `STYLELOOM_RENDER_MODE` | `per_shot` | `per_shot` \| `multi_shot` |
| `STYLELOOM_WIDTH` / `_HEIGHT` / `_FPS` | `720` / `1280` / `30` | Vertical 9:16 by default |
| `STYLELOOM_HOOK_CANDIDATE_COUNT` | `5` | Candidates per hook generation |
| `STYLELOOM_HOOK_TEMPERATURE` | `0.9` | |
| `STYLELOOM_HOOK_TOP_K` / `_SOFTMAX_TEMP` | `3` / `0.8` | Selection stochasticity |
| `STYLELOOM_HOOK_RECENCY_WINDOW` | `4` | How many recent runs are penalised |
| `STYLELOOM_HOOK_RECENCY_PENALTY` | `0.35` | Weight multiplier. `0` = hard exclusion |
| `STYLELOOM_MAX_CONCURRENT_RENDERS` | `3` | Upper bound; endpoint limits clamp it down |
| `STYLELOOM_ARCHETYPES_PATH` | `configs/archetypes.yaml` | Hook archetype pool |
| `STYLELOOM_CASTING_PATH` | `configs/casting.yaml` | Creator and background pools |
| `STYLELOOM_KLING_MODELS_PATH` | `configs/kling_models.yaml` | Endpoint specs |

Config files fall back to the copies bundled in the repo, so a fresh clone runs from
any working directory.

### The `KLING_*_MODEL` values are `model_name`s, not URL segments

The official API has one path per capability and picks the model in the request
body, so the setting is short and the path is a field in
`configs/kling_models.yaml`:

```
POST https://api-singapore.klingai.com/v1/videos/image2video
{"model_name": "kling-v3", "mode": "std", "image": "<raw base64>", ...}
```

An unrecognised name fails at startup rather than at the first request, because
the provider checks it against the registry before doing any work.

**`kling-v3` versus `kling-v3-omni` are different paths.** Kling's own docs
present 3.0 and 3.0 Omni on one page, and fal's naming hid the split further by
putting both under `kling-video/{v3,o3}/…` with one call shape. Officially they
are separate endpoints:

| `model_name` | path | |
|---|---|---|
| `kling-v3` | `/v1/videos/image2video` | the default |
| `kling-v3-omni` | `/v1/videos/omni-video/` | reference-driven tier |

### What the move off fal cost, and what it bought

**Cost: the creator-reference path is not wired up.** On fal a persona was one
inline field, `elements: [{frontal_image_url}]`. The official API references
elements by id — `element_list: [{element_id}]` plus `<<<element_1>>>` in the
prompt — and the id comes from a separate Element Management endpoint whose
request schema is not verified here. So `supports_persona` is `false` on every
specced endpoint, and passing `--persona` raises rather than being ignored: a
silently dropped reference renders a perfectly good video in which the creator's
face changes between cuts, which is the exact failure this harness exists to
catch. Creator identity still carries through the keyframe, which is generated
from a prompt describing the cast creator. That is the weaker mechanism, not none.

**Bought: one account, one credential pair, no SDK.** Keyframes come from Kling
Image and motion from Kling Video, so there is no second vendor to hold a key
for. Images upload as raw Base64 in the request body, so there is no CDN step.
And the JWT is twelve lines of `hmac`, so the optional `[fal]` extra is gone —
`pip install -e agent-core` reaches real generation.

**Neither: price.** The official platform bills credits against a subscription
tier rather than dollars per second, and the per-second credit rates live in a
PDF-style user guide this repo could not read. So `styleloom models` reports
billed seconds and the wasted share — the part of the arithmetic that does not
depend on a price list — and prints no currency figure rather than copying a
reseller's rate:

```
  model_name     floor    elo  per-shot (today)            multi_prompt
* kling-v3          3s      -  42s billed  (75% wasted)    14s billed  (3 calls)
  kling-v3-omni     3s      -  42s billed  (75% wasted)    14s billed  (3 calls)
```

With no Kling credentials at all the pipeline runs end to end on the mock video
provider for nothing, which is still the cheapest way to test everything except
generation quality.

---

## Tests

```bash
pip install pytest
python -m pytest
```

Both suites build their fixture videos with ffmpeg, so `conftest.py` at the repo
root checks for it at session start and exits with one line if it is missing. That
check exists because the raw symptom is otherwise a `FileNotFoundError` repeated
once per affected test — fifty on Linux, fifty `[WinError 2]`s on Windows — none of
which name ffmpeg or `PATH`.

221 tests, no network, no keys — `cli/tests/test_readme.py` fails if that number
goes stale, along with any command, setting or default this README describes but
the code no longer has. They cover style extraction recovering the fixture's
pacing, hook non-determinism and the recency penalty's measured effect, casting
varying creator and background, plan validation rejecting a misordered pipeline, a
full run producing a playable MP4 with one artifact per stage, three batched inputs
diverging, both render modes producing the same shape of output, and failure
isolating to the stage that broke.

`test_kling_provider.py` asserts the payload *shape* per endpoint without a key
or a network, because that is the part that breaks silently: `multi_shot` and
`shot_type` must accompany `multi_prompt` or the storyboard is quietly one shot,
the top-level `prompt` becomes invalid once they are set, shot entries are ordered
by an `index` field rather than array position, and `duration` is a bare string.
It also checks the hand-rolled JWT against an independently computed HMAC rather
than against a stored token, so the test fails if the encoding drifts.

Regression tests exist for bugs found during development:

- the outline passing the model's raw beat durations straight through, so a 7s
  reference produced a 19s video
- duplicate hook candidates making the softmax selection a no-op
- `max_concurrent_renders` exceeding Kling's per-user limit of 1
- adding `validation_alias` to the key fields making `Settings(fal_key=...)`
  evaluate to `""` with no exception
- the camera move living only in the motion prompt, so consecutive cuts requested
  byte-identical imagery — a two-cut hook was one frame shown twice
- `style hooks` reporting "no history" on a style with six entries, after the
  history log grew a `kind` field and the command passed `limit` into it

---

## Debugging

`.vscode/launch.json` has separate targets for the two distributions.

| Config | Target |
|---|---|
| `cli: doctor` | Environment check. No I/O, no keys — run it first if the debugger wiring itself looks wrong |
| `cli: run (offline, sample style)` | The full pipeline against `mock` providers |
| `cli: hook preview (offline)` | Hook sampling only, sub-second. The breakpoint target for `tools/hook.py` |
| `cli: pytest (cli/tests)` | CLI tests |
| `core: pytest (current file)` | Whichever test file is focused |
| `core: pytest (agent-core/tests)` | Core tests |
| `core: current file (scratch driver)` | A scratch `.py` that imports `styleloom_core` directly |

`agent-core` has no entry point — it is a library, so its debug targets are tests
and a scratch driver rather than a `main`.

Two things every config sets deliberately:

- **`cwd` is pinned to the workspace root.** `data_dir` defaults to the relative
  `Path("data")` with no repo-root fallback, so launching from anywhere else reads
  and writes a different `data/` without saying so. `configs/*.yaml` are safe either
  way — `resolve_config()` falls back to `REPO_ROOT`.
- **`args` is a literal array, not a `${input:...}` prompt.** VS Code does not split
  a `promptString` into argv, so a prompted `run my_style --text hello` arrives as
  one single argument and the CLI rejects it. Copy a config and edit its `args`.

`cli: run` needs the style installed once:

```bash
styleloom style set biodance_asmr_montage samples/styles/biodance_asmr_montage/style.json
```

---

## Import errors

`ModuleNotFoundError: No module named 'styleloom_core'` after the install reported
success. Both distributions are editable, and an editable install writes an import
hook holding the **absolute path** of this checkout into one specific interpreter's
`site-packages`. Two things can therefore be true at once: the package is installed,
and the interpreter you are running cannot see it.

Run this first — it says which of the two it is:

```bash
python - <<'EOF'
import importlib.util, shutil, sys, sysconfig
print("python    :", sys.executable)
print("venv      :", "yes" if sys.prefix != sys.base_prefix else "NO")
print("site-pkgs :", sysconfig.get_paths()["purelib"])
print("shell pip :", shutil.which("pip"))
for pkg in ("styleloom_core", "styleloom_cli"):
    spec = importlib.util.find_spec(pkg)
    print(f"{pkg:14}:", spec.origin if spec else "NOT FOUND")
EOF
```

**Cause 1 — pip and python are different interpreters.** `shell pip` sits under a
prefix that is not `site-pkgs` (a conda env, a pyenv shim, `/usr/bin/pip` against an
active venv, or the Windows Store shim). `pip list | grep styleloom` shows both
packages while the import fails. Reinstall through the interpreter itself:

```bash
python -m pip install -e agent-core -e cli
```

**Cause 2 — the checkout moved after install.** The path baked into the import hook
no longer exists, so the hook resolves nothing. `pip list` still shows the packages,
pointing at the old location. Reinstall from the current path:

```bash
python -m pip uninstall -y styleloom-core styleloom-cli
python -m pip install -e agent-core -e cli
```

Renaming a parent directory, moving the repo between a Docker build stage and a
mounted volume, or installing on the host and running in a container all do this.

Verify either fix with:

```bash
python -c "import styleloom_core, styleloom_cli; print(styleloom_core.__file__)"
```

The printed path must be inside the checkout you are editing. `pip install -e .` at
the repo root is not the fix and will fail: the root `pyproject.toml` is a uv
workspace with no `[project]` table, deliberately, because the installable
distributions are `agent-core` and `cli`.

### Pylance reports the import as missing but it runs fine

A different problem with the same message. `python -c "import styleloom_core"`
succeeds, and VS Code still underlines the import with
`reportMissingImports`.

Setuptools implements PEP 660 with an import hook. The `.pth` file it drops in
`site-packages` holds an executable statement, not a path:

```
import __editable___styleloom_core_0_2_0_finder; __editable___styleloom_core_0_2_0_finder.install()
```

The package-to-directory mapping lives inside that finder module. Pylance reads
`.pth` files but does not execute them, so at runtime the mapping is applied and to
static analysis it does not exist. Nothing is misconfigured — a language server
cannot see through an import hook.

`.vscode/settings.json` in this repo restates the mapping in a form Pylance reads:

```json
{ "python.analysis.extraPaths": ["agent-core", "cli"] }
```

Reload the window after checking it out. Also confirm the selected interpreter is
the venv (**Python: Select Interpreter** → the `.venv` entry); `extraPaths` fixes
the first-party imports, the interpreter is what supplies pydantic and typer.

Two alternatives, neither needed if the above works:

- `pyrightconfig.json` with the same `extraPaths`. Portable across editors and
  picked up by the `pyright` CLI, but when that file exists Pylance ignores every
  `python.analysis.*` setting, including `typeCheckingMode` — on this repo that
  means inheriting pyright's stricter default and seeing roughly 100 pre-existing
  type errors that Pylance was not reporting before.
- `python -m pip install -e agent-core -e cli --config-settings editable_mode=compat`,
  which makes setuptools write a plain directory path into the `.pth` instead of an
  import hook. Static analysis then resolves with no editor config at all. It is a
  per-machine install flag that is easy to forget on the next clone.

---

## Known limitations

**These are real, not hedging.**

- **Mock output is structurally correct, not watchable.** The offline renderer makes
  gradient keyframes with a push-in. It exists so the harness is runnable and
  gradeable without keys. Set a real provider for output you would publish.
- **Mock QC scores below 1.0 by design.** The offline renderer makes no attempt to
  match the reference's colour grade, so the grade checks correctly fail. Making the
  mock pass its own QC would make QC meaningless. Pacing, runtime, hook window and
  cut timing *do* pass offline, because those are reproducible by construction.
- **`multi_shot` is implemented but unproven on a real endpoint.** Offline drift is
  always zero because the offline sequence renderer *is* ffmpeg, so what has been
  verified is the plumbing — segments, captions by timestamp, the drift check — not
  the endpoint's obedience. `per_shot` remains the default for that reason.
- **`per_shot` discards most of what it generates.** At 20 cuts on Kling v3 Pro it
  bills 60s to deliver 30s. A deliberate trade: stretching shots to fill the floor
  would destroy the pacing the style schema exists to reproduce. Run
  `styleloom models` for the arithmetic on your own style.
- **Shot-size distribution is not measured.** Extraction recovers pacing and grade
  from pixels; shot size falls back to the schema default and is meant to be
  corrected via `styleloom style set`. Measuring it needs subject detection, which is
  the obvious next increment — and it shows, because the bundled references are
  person-centred.
- **BPM is estimated from cut intervals**, not audio onsets. Short-form edits cut on
  the beat often enough for this to be useful, but it is a heuristic and it is wrong
  on footage that does not.
- **Caption timing is per shot, not per beat.** Captions land at shot boundaries.
  Beat-level timing would need `pacing.cut_times` and the estimated BPM wired into
  `assemble`; neither is done, so the schema does not pretend otherwise.
- **Creator consistency currently rides on the keyframe alone.** The official
  API takes a character reference as an `element_id` minted by a separate Element
  Management endpoint, whose request schema is not verified here, so
  `supports_persona` is `false` and `--persona` raises rather than being dropped
  in silence. Identity still carries through the keyframe prompt, which is the
  weaker mechanism. Offline the creator varies through prompt tokens only.
- **Runs are in-process.** `styleloom batch` blocks until done, and killing it loses
  the current run — earlier completed runs are already on disk. Fine for a
  single-operator harness.
- **Kling endpoint schemas are pinned to July 2026, and some fields are
  inferred.** They are data (`configs/kling_models.yaml`), not code, but they do go
  stale — if a request starts failing on a parameter name, check the docs before
  the code. Every value in that file is marked verified or unverified; the
  unverified ones are duration floors and enums, because `kling.ai/document-api`
  renders client-side and only fragments that reached a search index were
  readable.
- **Reference videos are used for analysis only.** Extraction reads measurements and
  keyframes from them; no reference footage is ever placed in an output.
- **`api/`, `worker/`, `infra/` and `observability/` are not implemented.** The seams
  they attach to exist and are documented; the directories are deliberately absent
  rather than stubbed, so the repo does not imply commitments it has not made.
