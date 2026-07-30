# Architecture

The harness is a CLI. The layering exists so that adding an API or a worker later
is a new transport rather than a rewrite.

---

## Data flow

Two entry points. The left one runs once per reference channel and produces a
reusable asset; the right one runs once per input.

```
 STYLE EXTRACTION  (styleloom style extract)
 ─────────────────────────────────────────────────────────────────────────
   reference .mp4 x N
        │
        ▼
   media.probe_video ......... one decode pass per file
        │                      cuts, avg shot length, saturation,
        │                      contrast, warmth, keyframes
        ▼
   tools.analyze ............. LLM names ONLY the qualitative parts
        │                      (grade, camera moves, tone) + keyframes
        ▼
   data/styles/<id>/style.json ....... hand-editable, the contract


 GENERATION  (styleloom run / batch)          artifact written
 ─────────────────────────────────────────────────────────────────────────
   text | image | video  +  style_id
        │
        ▼
   runner.prepare_session ............................ style_used.json
        │                                              inputs.json
        ▼
   1. ingest ....... normalise any input to a Brief .. brief.json
        │
        ▼
   2. casting ...... draw creator + background,       casting.json
        │            recency-penalised, then
        │            specialise both against the brief
        ▼
   3. outline ...... body beats + payoff,             outline.json
        │            rescaled to the reference length
        ▼
   4. hook ......... archetype draw -> N candidates   hook.json
        │            -> top-k softmax pick
        │            (AFTER outline: it must promise
        │             the actual payoff)
        ▼
   5. storyboard ... per-shot prompts; style +        storyboard.json
        │            creator + setting tokens
        │            injected into EVERY shot
        ▼
   6. render ....... text -> clip (t2v),              render.json
        │            concurrent, per-shot failure     shots/*.mp4
        │            isolation, trim to exact length
        ▼
   7. assemble ..... burn captions, concat, BGM       assemble.json
        │                                             final.mp4
        ▼
   8. qc ........... re-probe the OUTPUT with the     qc_report.json
        │            SAME probe used on the reference
        ▼
   data/runs/<run_id>/          ->  styleloom export  ->  bundle/
                                    (mp4s + prompts.txt + manifest.json)
```

Each numbered step is a registered tool declaring the artifacts it reads and
writes, so the order above is validated before execution rather than implied by
the code. Progress is emitted as events, not printed.

---

## Layout

```
StyleLoom/
├── agent-core/            distribution: styleloom-core
│   └── styleloom_core/
│       ├── schema.py        data contracts (pydantic)
│       ├── config.py        Settings, env-driven, injected not global
│       ├── context.py       Context: settings + providers + stores + event sink
│       ├── events.py        Event, EventSink protocol
│       ├── errors.py        error taxonomy
│       ├── media.py         ffmpeg + OpenCV primitives
│       ├── providers/       LLM and video adapters (mock | anthropic | fal)
│       ├── memory/          StyleStore, RunStore, HookHistory
│       ├── session/         RunSession: per-run state and artifact bag
│       ├── tools/           one module per stage, plus the registry
│       ├── planner/         builds and validates the execution order
│       └── runner/          executes a plan, emits events
├── cli/                   distribution: styleloom-cli
│   └── styleloom_cli/
│       ├── main.py          typer app
│       ├── console.py       ConsoleSink: renders core events to a terminal
│       ├── options.py       shared flags, core errors -> clean exits
│       └── commands/        style / run / batch / runs / hook
├── configs/               archetypes.yaml, fal_models.yaml  (data, not code)
├── docs/
└── data/                  styles/ runs/ uploads/   (the deliverable lives here)
```

`api/`, `worker/`, `infra/` and `observability/` are deliberately absent rather
than stubbed. See "Extension path" below for what each would contain and what it
would plug into. Empty directories would suggest more design commitment than has
actually been made.

---

## Two distributions, not one package

`agent-core/` and `cli/` each have their own `pyproject.toml` and install
separately. This is the one structural decision worth defending, because folders
alone would have been cheaper.

"Framework independent" is only enforceable as a *dependency* boundary.
`styleloom-core` declares pydantic, PyYAML, httpx, OpenCV and numpy — and no web
or CLI framework. Nothing in the core can import typer, because typer is not
installed as its dependency. A single package with a `core/` subfolder would have
relied on discipline instead, and discipline erodes.

There is a test for it (`test_the_core_declares_no_framework_dependency`), which
reads the installed metadata rather than trusting the file.

---

## Dependency direction

```
        schema ── errors ── config
                    │
                    ▼
        media ── providers          (adapters: ffmpeg, OpenCV, HTTP, fal)
                    │
                    ▼
        memory ── session ── context
                    │
                    ▼
                  tools             (one per stage; never import each other)
                    │
                    ▼
                 planner            (order as data, validated)
                    │
                    ▼
                  runner            (executes, emits events)
                    │
                    ▼
        cli  │  api  │  worker      (transports; peers, never import each other)
```

Two rules do the work:

**Tools do not call tools.** Previously the render stage imported the post stage
for `trim_to`, and QC imported analyze for `probe_video`. That put execution order
in the import graph, which is exactly what a declared plan is supposed to make
explicit. Shared operations moved down to `media.py`, and tools now depend only on
that.

**Nothing reaches for a global.** `Settings` used to be a module-level singleton
instantiated at import. The old test suite carried a comment explaining that
`STYLELOOM_DATA_DIR` had to be set *before importing the package*, which is the
symptom. Settings are now constructed by the caller and passed in a `Context`, so
two configurations can coexist in one process — which is a hard requirement the
moment a server handles more than one tenant.

---

## Tools, plans, and why the plan is static

Each tool declares what it reads and writes:

```python
@tool("outline", reads=("style", "brief"), writes="outline")
def outline(ctx: Context, session: RunSession) -> Outline: ...
```

A plan is an ordered tuple of tool names. Because dependencies are data, a plan is
checkable before anything expensive runs:

```
$ styleloom plan
plan: standard
  1. ingest      (- -> brief)
  2. outline     (style, brief -> outline)
  3. hook        (style, brief, outline -> hook)
  4. storyboard  (style, brief, outline, hook -> storyboard)
  5. render      (storyboard -> render)
  6. assemble    (style, storyboard, render -> assemble)
  7. qc          (style, storyboard, assemble -> qc)
```

Reordering `hook` before `outline` fails at plan time with the artifact name, not
at render time with a `KeyError`.

The plan is static rather than LLM-generated on purpose. The claim this system
makes is that the *same* pipeline turns different inputs into videos that read as
one channel. If a model reorders or drops stages between runs, output differences
stop being attributable to the input and the reproducibility claim is gone.
Non-determinism is confined to where it is a stated requirement — hook content —
instead of spread across control flow.

What the registry buys over a hardcoded call chain is the validation above, plus
the ability to add a stage without editing the runner. What it does not buy, and
does not pretend to, is dynamic tool selection.

`style` is the one artifact no tool writes: the runner loads it from the store and
seeds it, and `Plan.validate` is told about it via `PRELOADED`. It is also
snapshotted into the run folder as `style_used.json`, because `style.json` is
meant to be hand-corrected between runs and a run that cannot say which version
it used is not reproducible.

---

## Events: the seam that makes transports thin

The core never prints. The runner emits `Event` objects to an `EventSink`, which
is a `Protocol` — so a transport satisfies it structurally, without the core
importing anything from the transport side.

| Transport | Sink implementation |
|---|---|
| CLI | `ConsoleSink` — one line per stage with elapsed time |
| API | an SSE or WebSocket writer |
| Worker | a queue status / progress update |

`NullSink` is the default, so the core requires no transport at all, and
`ListSink` collects events in memory, which is how the tests assert stage
ordering.

---

## The one entry point

```python
from styleloom_core import RunInputs, Settings, build_context, run_once

ctx = build_context(Settings())
record = run_once(ctx, "my_style", RunInputs(text="러닝 첫 달에 무릎이 아픈 이유"))
```

`run_once` is the whole surface for producing a video. The CLI calls it directly;
an API route would call it from a background task; a worker would call it from a
queue consumer. It does not raise for a failed render — the returned `RunRecord`
carries `status` and `error`, because a batch has to continue after one input
fails. Configuration and missing-style errors *do* propagate, since those mean the
caller asked for something impossible.

`run_batch` is sequential, and that is a design decision rather than laziness: the
hook's recency penalty reads the style's archetype history, so run N has to see
what run N-1 chose. Running them in parallel would let all three draw the same
archetype and lose exactly the variety the batch exists to show.

---

## Extension path

Nothing below is implemented. This is what each layer would contain and where it
would attach, so the shape of the boundary is checkable now rather than
discovered later.

**`api/`** — FastAPI over the same core.

```
api/
├── pyproject.toml        depends on styleloom-core, fastapi, uvicorn
└── styleloom_api/
    ├── main.py
    ├── dependencies/     build_context() per request, or once per process
    ├── schemas/          request/response models wrapping RunInputs / RunRecord
    ├── routes/           styles, runs, hooks
    ├── middleware/
    └── auth/
```

Routes call `run_once` / `run_batch` / `extract_style` and nothing else. The event
sink becomes an SSE writer, so `GET /runs/{id}/events` streams the same stages the
CLI prints. No core change is required for any of this — which is the property the
layering was built to have.

**`worker/`** — for durability. A render is a long HTTP-bound job; if in-process
background tasks are not good enough, `queue_consumer.py` pulls a job and
`task_handlers.py` calls `run_once`. The handler is a few lines because the core
function is already the unit of work.

**`observability/`** — the `EventSink` protocol is the integration point. An
`OTelSink` would map `STAGE_STARTED` / `STAGE_FINISHED` to spans without the core
knowing tracing exists.

**`infra/`** — a Dockerfile needs `ffmpeg` and a CJK font (`fonts-noto-cjk`);
`styleloom doctor` checks for both. `data/` needs a volume, since run artifacts
are the deliverable.

The load-bearing claim is narrow and testable: no core module imports a transport,
and `run_once` is sufficient. Both hold today.
