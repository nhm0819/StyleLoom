# Tool rationale

Why each choice was made, including the alternatives that were rejected.

## Summary

| Tool | Role | Why this one |
|---|---|---|
| **ffmpeg** | cut, trim, concat, burn captions | Cut timing and caption position are deterministic operations; a model does them worse than a `for` loop and gives up reproducibility. |
| **OpenCV** | measure pacing, colour, keyframes | Style properties have to be numbers to be reproducible *and* checkable in QC. One decode pass yields cuts, colour stats and keyframes together. |
| **Claude Sonnet 5** (Anthropic) | brief, outline, hook candidates, style naming | Vision-capable, so reference keyframes go in alongside the measured stats; reliable structured JSON under an explicit schema. |
| **KlingAI Open Platform** | text-to-image, image-to-video, text-to-video | Called directly rather than through an aggregator, and `multi_shot` — the storyboard that lowers the billed floor from 3s to 1s per cut — is a first-class part of its own schema rather than a wrapper's translation of it. |
| **Kling Video v3** | text-to-video, **default** | Takes `aspect_ratio` rather than pixel dimensions, so 9:16 is requested directly rather than cropped afterwards. `multi_shot` storyboards, 3s floor, 15s ceiling. `mode: std\|pro` selects the quality tier in the request, so switching tiers is a setting rather than a different endpoint. |
| **pydantic** | data contracts, settings | Every stage artifact is written to disk and read back; validation at the boundary is the contract. Env-driven settings come free. |
| **Typer** | CLI | Type hints *are* the parser, so the command signature and its validation are one declaration instead of two. |
| **pytest** | tests | 137 tests, no network, no keys. |

Detail and rejected alternatives below.

---

## Editing-based, not purely synthesis-based

The pipeline generates footage with a model but assembles, captions and times it
with ffmpeg.

Fully synthetic generation of a 30-second multi-shot video is still unreliable at
the thing that actually defines a short-form style: exact cut timing and caption
placement. Those are deterministic operations. Handing them to a probabilistic
model buys nothing and loses reproducibility — and reproducibility is the entire
claim being made.

So the model does what only a model can do (imagery, motion), and ffmpeg does what
a model does worse than a `for` loop (cutting on a schedule, burning text at a
fixed position).

---

## Two-stage render: text → anchor → clip

`tools/render.py` calls text-to-video directly. An earlier revision generated a
still keyframe and animated it, on the reasoning recorded below — kept because
the reasoning was sound and the mistake was in a premise, which is worth seeing:

> Shot-to-shot consistency of colour grade and subject identity is what makes a
> set of clips read as one channel. Image models are cheaper to iterate, easier to
> constrain with a reference image, and produce a *fixed* look that the video model
> then only has to move. Straight text-to-video re-rolls the look on every shot,
> and by shot six the grade has drifted.

The premise that failed is "a fixed look". Every cut generated **its own**
keyframe from **its own** text, so nothing was fixed across cuts — shot 6 re-rolled
the look exactly as text-to-video would, having paid an extra generation for the
privilege. The keyframe was doing real work within one cut, fixing composition
before motion was bought, and none at all between cuts.

What would have made the premise true is one keyframe reused as every cut's start
image, or a character reference pinned by `element_list`. The first is not what the
code did; the second needs an Element Management call this repo could not verify.

So the two calls per cut were buying single-shot composition control, and the
consistency argument was not being served.

**The keyframe stage is now back, in the shape that sentence describes.** One anchor
per run, generated from the cast creator and setting, reused as the *reference image*
for every request's opening frame. `tools/keyframe.py` builds it and `render` opens
each request on it.

Two mechanisms, two axes, which is why both are on:

| | within one request | between requests |
|---|---|---|
| `multi_shot` (one generation) | holds | does not |
| `first_frame` (shared anchor) | — | holds |

`multi_shot` alone makes a 14-cut montage three requests and therefore three people.
The anchor ties them together. And the cost is not the old cost: a 30s video in
`multi_shot` mode buys 1 anchor + 3 frames, not one image per cut.

Trade-off, stated plainly: three or four image calls per video that text-to-video
does not pay, and `first_frame` only fixes the *first* frame — how far the model
drifts from it over the following seconds is the model's decision. `element_list`
would pin identity harder and still needs an Element Management call this repo has
not verified. So identity is requested, not guaranteed, and QC measures colour
rather than faces. Full reversal recorded in
[KEYFRAME_SCOPE.md](KEYFRAME_SCOPE.md).

---

## OpenCV measurement instead of asking an LLM to describe the style

`media.probe_video` computes cut times, average shot length, saturation, contrast
and warmth from pixels. The LLM is called once, only to *name* things that are
genuinely linguistic (the grade, the camera vocabulary, the tone), and is told
explicitly not to invent numbers.

An LLM answering "fast cuts, punchy colour" is not reproducible. `avg_shot_sec =
1.34` is, and it can be checked against the output in QC. The division is: pixels
for anything measurable, language for anything nameable.

**PySceneDetect** would have been the conventional choice for cut detection. It was
not used because OpenCV was already a dependency for the colour statistics, and the
same single decode pass yields cuts, colour stats and keyframes together. Adding a
second library to re-decode the same file for one of those three would be worse.

**librosa** was rejected for BPM for the same reason: a heavy dependency for one
number. BPM is estimated from the median cut interval on the assumption that
short-form edits cut on the beat. This is documented as a heuristic, not presented
as measurement.

---

## Captions burned in ffmpeg, not generated by the video model

Video models render text unreliably, inconsistently, and often in the wrong
language. The caption spec in `style.json` (font, position, wrap width, stroke) has
to be byte-identical across every video for the set to look like one channel.
`drawtext` guarantees that; a model does not.

Caption text is passed via `textfile=` rather than inline, which sidesteps drawtext
escaping entirely — Korean captions routinely contain colons, quotes and commas
that would each otherwise need escaping.

Wrapping is by character count, not by word. Korean does not put spaces at
predictable places, so a word wrapper leaves ragged lines.

---

## CLI first, with the API as a peer rather than a port

The previous version of this project was API-first, with the pipeline orchestrated
inside a FastAPI `BackgroundTasks` call. That made the transport a load-bearing
part of the pipeline: there was no way to run a render without starting a server.

Now `run_once` in `styleloom_core.runner` is the single entry point and the CLI is
a thin caller. Progress is reported by emitting events rather than printing, so the
same core drives a terminal, an SSE stream, or a queue consumer.

The cost is one extra indirection (an `EventSink` instead of `print`). The benefit
is that adding the API is additive. See [ARCHITECTURE.md](ARCHITECTURE.md).

**Celery was rejected** for now. A render is one long HTTP-bound job with no
fan-out, and its artifacts have to land on disk in human-readable form regardless.
A broker plus a worker plus a result backend would add three moving parts to serve
a single-operator harness. `worker/` is where that goes if durability becomes a
requirement, and the handler is a few lines because `run_once` is already the unit
of work.

---

## Static plan instead of an LLM planner

Tools are registered with declared `reads` / `writes`, and a plan is an ordered
tuple of tool names that gets validated before execution.

An LLM planner was considered and rejected. The claim under test is that the *same*
pipeline turns different inputs into videos that read as one channel. If a model
reorders or drops stages between runs, output differences stop being attributable
to the input, and the reproducibility claim collapses. Non-determinism belongs
where it is a stated requirement — hook content — not in control flow.

What the registry buys over a hardcoded call chain is that a misordered step fails
at plan time with the missing artifact named, rather than at render time with a
`KeyError` after money has been spent. What it does not buy, and does not pretend
to, is dynamic tool selection.

---

## Filesystem store, not a database

`data/runs/<run_id>/` *is* the deliverable. Every stage artifact is a JSON file
someone can open and read. A database would hide exactly the intermediate state
that makes the system inspectable, and would then need an export step to produce
what the filesystem already provides.

One exception to "just files": archetype history is an append-only JSONL log per
style (`data/styles/<id>/hook_history.jsonl`) rather than something derived by
scanning runs. The previous implementation loaded and parsed every `run.json` on
disk to find the last four archetypes, which is O(all runs) on every single run. A
tail read of one small file is O(window), and the log is still a text file anyone
can read.

---

## Provider abstraction with an offline default

`BaseLLM` and `BaseVideoProvider` are two-method interfaces with a `mock`
implementation that requires no key and no network.

This is not a testing convenience. The frontier video models turn over every few
months — several of the current leaders did not exist a year ago. Hardcoding one
gives the harness a shelf life. Keeping model choice in settings means someone else
can point this at whatever is best when they read it.

The `mock` renderer is real ffmpeg output, not a stub: it produces a playable MP4
with the correct shot count, durations and captions. The full pipeline — including
QC — is therefore verifiable by anyone who clones the repo, which matters more than
the mock footage being pretty.

**KlingAI's own Open Platform** is the one real integration. An earlier revision
went through fal.ai, which fronts Seedance, Kling and others behind one queue API;
that bought model breadth at the cost of a translation layer between this repo and
the vendor schema. Going direct trades the breadth away for three things: one
account and one credential pair for both pipeline stages, raw Base64 image upload
so there is no CDN step, and no SDK — the JWT is twelve lines of `hmac`, so
`pip install -e agent-core` reaches real generation with no extra.

It costs something real, recorded here rather than buried: the creator-reference
path is not wired up. fal accepted `elements: [{frontal_image_url}]` inline; the
official API wants an `element_id` from a separate Element Management endpoint
whose schema is unverified in `configs/kling_models.yaml`. `supports_persona` is
therefore `false`, and `--persona` raises rather than being silently ignored,
because a dropped reference produces a good-looking video in which the creator's
face changes between cuts.

Changing model is still a `model_name` string plus a path in the spec file, which
is what keeps the comparison below actionable rather than academic.

### Which image-to-video endpoint, and why not the best one

> Superseded in part by the move off fal. The dollar figures below were fal's
> published per-second rates, and Seedance is no longer reachable from this repo
> at all. The *shape* of the argument survives the move and is the reason the
> pipeline is built the way it is, so it is kept rather than deleted.
> `styleloom models` now reports billed seconds instead of dollars, because the
> official platform bills credits against a subscription and publishes no
> per-second rate this repo could record honestly.

`styleloom models` prints this for any saved style. For a 30s video at short-form
pacing (20 shots, ~1.5s each):

| Endpoint | Floor | Elo | Per-shot, as built today | With `multi_prompt` |
|---|---|---|---|---|
| Seedance 2.0 | 4s | 1194 | **$24.27** (62% wasted) | not supported |
| Kling v3 Pro | 3s | 1075 | **$6.72** (50% wasted) | **$3.36** (2 calls) |

Seedance 2.0 is the better model and is not the default. Three reasons, in order
of weight:

1. **`multi_prompt`.** Kling 3.0 Pro accepts a list of `{prompt, duration}` shots in
   one generation, lowering the billed floor from 3s to 1s per cut. It does *not*
   remove the floor: re-reading the schema showed the nested duration is an integer
   enum (1–15), so a 0.76s cut is billed and delivered as 1s. On the ASMR montage
   that inflates an 10.7s video to 14s and destroys the pacing being copied — which
   is why `per_shot` plus an ffmpeg trim remains the default and `multi_prompt` is
   only the cheaper path for styles whose cuts already run a second or longer.
   Nothing else here offers it at all.
2. **`elements`.** It is the only endpoint that takes a character reference, so the
   creator cast per run stays the same person across cuts. Without it the
   "creator" element of the assignment is prompt tokens and hope.
3. **Rate.** $0.112/s against $0.3034/s, before the floor is even accounted for.

The Elo gap is 119 points and real. For a single hero video where cost and creator
consistency do not matter, `STYLELOOM_FAL_I2V_MODEL=bytedance/seedance-2.0/image-to-video`
is one environment variable. What this system sells is repeatable output with a
consistent presenter, and on those axes Kling wins.

**Veo 3.1 is available and is not a contender for the default.**
`fal-ai/veo3.1/fast/image-to-video` replaces the `veo2` endpoint, which fal marks
deprecated. It loses on all three axes above: no `multi_prompt`, no reference-image
parameter, and a 4s floor. Its `duration` is also a sparse enum — 4s, 6s, 8s, with
no 5s or 7s — so the payload builder rounds up into a legal member rather than
clamping. Selectable with one environment variable; not a default.

**Not added, and deliberately so.** Three endpoints look worth having and are absent
from `configs/fal_models.yaml` because their parameter names were not read off the
model page: `google/nano-banana-2` (holds one person's identity across generations
without fine-tuning — the natural upgrade for the casting portrait),
`alibaba/happy-horse` (native Korean lip-sync at $0.14/s, and it *outscores* Kling
at Elo 1092), and `fal-ai/kling-video/o3/pro` (start+end frame, also multi_prompt).
Guessing a parameter name is the exact failure this file exists to prevent — a wrong
name is ignored silently. Each is one verified YAML entry away.

It does *not* front them behind one schema, which was the surprise. Verified
against the official model pages (July 2026):

| | Seedance 2.0 | Kling v3 Pro |
|---|---|---|
| start image | `image_url` | `start_image_url` |
| duration | string, 4–15s | string, 3–15s |
| aspect ratio | `aspect_ratio` param | inferred from image; param ignored |
| reference character | not on this endpoint | `elements` array |
| concurrency | not documented | 1 per user |

A single hardcoded payload would fail on at least one of these for every model
except the one it was written against — and would fail silently, by sending a
parameter name the endpoint ignores. Hence `configs/fal_models.yaml`: the
differences are data, and adding a model is a data change. This is the one place
the project deliberately buys configurability, because the alternative is not
simpler code, it is code that quietly sends the wrong thing.

Payload construction is separated from the network call
(`build_animate_payload`), so the shapes above are unit-tested without a key.

The official `fal_client` SDK is used rather than hand-rolled HTTP: it owns the
queue protocol and CDN upload, and fal's own documentation discourages base64 data
URIs above a few KB, while keyframes are hundreds of KB.

---

## The duration floor, and what it costs

Every image-to-video endpoint has a minimum clip length — 4s on Seedance, 3s on
Kling. Short-form shots run 1–2s. These are irreconcilable, so `render` requests
the floor and trims the result with ffmpeg.

This is the expensive decision in the project, and it was made deliberately:

- An 11-shot video bills 11 × 4s ≈ **$10.64** on Seedance fast, plus keyframes, for
  roughly **9 seconds of usable footage**. About 80% is discarded.
- The alternative — stretching shots to the 4s floor — would destroy pacing, which
  is the single most characteristic property of a short-form style and the thing
  `style.json` exists to reproduce. A system that cannot hold a 1.2s cut cannot
  replicate the reference at all.

So the money is spent on the requirement rather than saved by abandoning it.

The real fix is batching: Kling's `multi_prompt` takes a list of shots with
individual durations in one generation, and Seedance cuts between shots natively
within a single 15s output. Either would collapse many calls into one and cut cost
several-fold. Not implemented — it moves the render/assembly boundary
significantly, and doing it badly would trade a known cost for unpredictable
pacing.

Trimming re-encodes rather than stream-copying, so the cut lands on the requested
frame instead of the nearest keyframe. Frame accuracy here is load-bearing for the
same reason as above.

---

## Anthropic for the LLM

Claude Sonnet 5. Vision-capable, which `tools/analyze.py` uses to send reference keyframes alongside
the measured statistics, and reliable at returning parseable JSON under an explicit
schema instruction. `extract_json` still handles fenced output and stray prose
defensively, because "return only JSON" is a request, not a guarantee.

---

## Typer for the CLI

Type hints become the argument parser, so the command signature and the validation
are the same declaration. The alternative was argparse, which would have meant
writing the parsing twice — once as types for the reader and once as
`add_argument` calls for the machine.

Typer lives in `styleloom-cli`'s dependencies and nowhere near the core. That is
checked by a test, not by convention.
