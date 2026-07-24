# StyleLoom

Decompose a reference short-form video into a reusable **style schema**, then
generate new videos in that style from arbitrary inputs.

The deliverable is not one good video. It is a system where *different inputs
through the same unchanged pipeline* produce different videos that still read as
the same channel.

---

## Quick start

No API keys required. The default providers are offline (`mock` LLM + ffmpeg
renderer), so the whole pipeline runs end to end on a fresh clone.

```bash
pip install -e ".[dev]"
uvicorn styleloom.api.main:app --reload
# docs at http://127.0.0.1:8000/docs
```

Requires `ffmpeg` on PATH.

### 1. Extract a style from reference videos

```bash
curl -X POST http://127.0.0.1:8000/styles/extract \
  -F style_id=my_style \
  -F files=@refs/ref01.mp4 \
  -F files=@refs/ref02.mp4
```

Returns `style.json`, saved to `data/styles/my_style/`. Edit it by hand and
`PUT /styles/my_style` if the extractor mislabels something — the schema is the
contract, not the extractor.

### 2. Generate three videos from three different inputs

```bash
curl -X POST http://127.0.0.1:8000/runs/batch \
  -H 'Content-Type: application/json' \
  -d '{"style_id":"my_style","inputs":[
        "회사에서 아무도 안 알려주는 엑셀 단축키",
        "자취 3년차가 후회하는 가전 구매 순서",
        "러닝 첫 달에 무릎이 아픈 진짜 이유"]}'
```

Poll `GET /runs/{run_id}`, then fetch `GET /runs/{run_id}/video`.

Image or video input instead of text:

```bash
curl -X POST http://127.0.0.1:8000/runs \
  -F style_id=my_style -F text="이 제품 소개해줘" -F input_file=@product.jpg
```

### 3. Prove the hook is not frozen

```bash
curl -X POST http://127.0.0.1:8000/hooks/preview \
  -H 'Content-Type: application/json' \
  -d '{"style_id":"my_style","text":"엑셀 단축키","n":8}'
```

Same input, eight generations. The response reports `distinct_archetypes` and
`distinct_texts` directly, so the non-determinism requirement is checkable in
one call rather than taken on faith.

---

## Pipeline

```
reference videos ──▶ ① ANALYZE ──▶ style.json  (reusable asset)
                                        │
user input (text/image/video) ──▶ ② INGEST ──▶ brief.json
                                        │
                                        ▼
                               ③ OUTLINE ──▶ outline.json   (body beats + payoff)
                                        │
                                        ▼
                               ④ HOOK ──▶ hook.json         ★ non-deterministic
                                        │
                                        ▼
                               ⑤ STORYBOARD ──▶ storyboard.json
                                        │
                                        ▼
                               ⑥ RENDER  (t2i keyframe → i2v clip, per shot)
                                        │
                                        ▼
                               ⑦ POST    (burn captions → concat)
                                        │
                                        ▼
                               ⑧ QC ──▶ qc_report.json
```

Every stage is `JSON in → JSON out` and writes its artifact to
`data/runs/<run_id>/`. A finished run is fully inspectable; a failed run shows
exactly which stage broke.

### Why the hook comes *after* the outline

A hook is a promise the body has to pay off. Generating it first would produce a
generic attention grabber; generating it fourth means it can reference the
actual payoff. `generate_hook` receives `outline.payoff` explicitly and is
instructed never to bait something the video does not deliver.

### How the hook varies across reruns

Temperature alone would make variation an accident. Three independent
randomisation points instead:

| Point | Mechanism | Effect |
|---|---|---|
| Archetype | weighted draw from `configs/archetypes.yaml`, with a recency penalty over the last 4 runs of the same style | a *structurally* different opening, not a paraphrase |
| Candidates | N generations, temperature 0.9, no seed | lexical variety within the archetype |
| Selection | top-k softmax sample instead of argmax | the best candidate is favoured, not guaranteed |

Entropy comes from `secrets.SystemRandom` — OS entropy, unseeded, so a run can
never accidentally repeat. Every decision (pool, sampled archetype, all
candidates with scores, selection method) is written to `hook.json`.

The recency penalty is why `POST /runs/batch` with three inputs gives three
different archetypes: each run sees what the previous ones chose.

---

## Reusing this with your own references

Nothing about a specific reference video is in the code. To retarget:

1. `POST /styles/extract` with your videos → a new `style.json`.
2. Optionally edit `configs/archetypes.yaml`, or point
   `STYLELOOM_ARCHETYPES_PATH` at your own file. No code change.
3. Point `STYLELOOM_VIDEO_PROVIDER=fal` at whichever models you use. Adding an
   endpoint means adding an entry to `configs/fal_models.yaml` — its parameter
   names, duration floor and output key — not editing the provider.

---

## Configuration

Copy `.env.example` to `.env`. Everything is env-driven.

| Variable | Default | Notes |
|---|---|---|
| `STYLELOOM_LLM_PROVIDER` | `mock` | `mock` \| `anthropic` |
| `STYLELOOM_VIDEO_PROVIDER` | `mock` | `mock` \| `fal` |
| `STYLELOOM_FAL_T2I_MODEL` | `fal-ai/flux-2-flex` | must be a key in `configs/fal_models.yaml` |
| `STYLELOOM_FAL_I2V_MODEL` | `bytedance/seedance-2.0/fast/image-to-video` | ditto |
| `STYLELOOM_HOOK_CANDIDATE_COUNT` | `5` | candidates per hook generation |
| `STYLELOOM_HOOK_TOP_K` / `_SOFTMAX_TEMP` | `3` / `0.8` | selection stochasticity |

See [`docs/TOOL_RATIONALE.md`](docs/TOOL_RATIONALE.md) for why each tool was
chosen — including the ones that were not.

---

## Tests

```bash
pytest
```

16 tests, no network, no keys. They cover style extraction recovering the
fixture's pacing, hook non-determinism, hook candidate distinctness, a full run
producing a playable MP4, and three batched inputs diverging.

`test_fal_provider.py` stubs the fal SDK and asserts the payload *shape* per
endpoint, which is the part that silently breaks: Seedance takes `image_url`,
Kling takes `start_image_url`, `duration` is a string on both, and Kling infers
aspect ratio from the start image so sending one is wrong.

Four tests are regressions for bugs found during development: the outline passed
the model's raw beat durations straight through (a 7s reference produced a 19s
video), duplicate hook candidates made the softmax selection a no-op, sub-minimum
shot durations were sent to endpoints that reject them, and
`max_concurrent_renders` could exceed Kling's per-user limit of 1 and fail a run.

---

## Known limitations

**These are real, not hedging.**

- **Mock output is structurally correct, not watchable.** The offline renderer
  makes gradient keyframes with a push-in. It exists so the harness is runnable
  and gradeable without keys. Set a real provider for output you would publish.
- **Mock QC scores ~0.5–0.7 by design.** The mock renderer makes no attempt to
  match the reference's colour grade, so the grade checks correctly fail. Making
  the mock pass its own QC would make QC meaningless. With a real provider these
  checks become informative.
- **BPM is estimated from cut intervals**, not audio onsets. Short-form edits cut
  on the beat often enough for this to be useful, but it is a heuristic, and it
  is wrong on footage that does not cut to music.
- **Shot-size distribution is not measured.** `analyze` recovers pacing and
  grade from pixels; shot size currently falls back to the schema default and is
  meant to be corrected via `PUT /styles/{id}`. Measuring it needs subject
  detection, which is the obvious next increment.
- **Background jobs run in-process** via FastAPI `BackgroundTasks`. Restarting
  the server loses in-flight runs. Fine for a single-operator harness; swap
  `runner.execute` onto a queue if you need durability.
- **Persona consistency is passed through, not enforced.** A reference image
  reaches the keyframe call, but the mock ignores it and real providers vary in
  how well they honour it. Treat cross-shot identity as best-effort.
- **Real rendering costs about $11 per video, and discards 80% of what it
  generates.** Every image-to-video endpoint has a duration floor — Seedance 4s,
  Kling 3s — while short-form shots run 1–2s. StyleLoom requests the floor and
  trims, because pacing is the thing the style schema exists to reproduce, but
  you pay for the full clip. An 11-shot video bills 11 × 4s even though it uses
  9s. Kling's `multi_prompt` and Seedance's in-generation multi-shot could
  collapse several shots into one call and cut this substantially; that is the
  obvious next increment and is not implemented.
- **Persona consistency only works on Kling v3 endpoints**, via its `elements`
  parameter. On Seedance the provider emits a warning and proceeds without it
  rather than silently dropping the reference.
- **fal endpoint schemas are pinned to July 2026.** They are data
  (`configs/fal_models.yaml`), not code, but they do go stale — if a request
  starts failing on a parameter name, check the model page before the code.
