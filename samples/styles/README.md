# Sample styles

Two `style.json` files extracted from Biodance short-form references.

They live here rather than in `data/styles/` because `.gitignore` excludes
`data/styles/*` — that directory is runtime state, not a versioned asset.
Install either one before running:

```bash
styleloom style set biodance_asmr_montage    samples/styles/biodance_asmr_montage/style.json
styleloom style set biodance_ugc_talkinghead samples/styles/biodance_ugc_talkinghead/style.json
styleloom style ls
```

`style set` validates against `StyleSchema` before writing, so a bad edit fails
at install time rather than mid-run.

## Provenance

Every number came from `media.probe_video()` on the reference file — one OpenCV
decode pass, not a model's impression. The qualitative fields (`grade`,
`cut_style`, `moves`, `voice_tone`, `keywords`) were named from extracted frames,
which is the division of labour `tools/analyze.py` already enforces: measure the
numbers, name only what is linguistic.

To regenerate from scratch with a real vision model:

```bash
export ANTHROPIC_API_KEY=...
styleloom style extract biodance_asmr_montage    사전과제_reference1.mp4 --force
styleloom style extract biodance_ugc_talkinghead 사전과제_reference2.mp4 --force
```

Expect the measured fields to come back identical and the named fields to differ
in wording.

## What the two references actually are

| | `biodance_asmr_montage` | `biodance_ugc_talkinghead` |
|---|---|---|
| source | `사전과제_reference1.mp4` | `사전과제_reference2.mp4` |
| duration | 10.7s | 12.13s |
| shots | 14 | 3 |
| avg shot | 0.764s | 4.044s |
| cuts in first 3s | 5 | 0 → clamped to 1 |
| saturation / contrast / warmth | 0.348 / 0.442 / 0.579 | 0.380 / 0.384 / 0.524 |
| bpm from median cut interval | 128.8 | 114.3 |

The two are near-opposites, which is the useful part: they exercise different
paths through the same pipeline rather than producing the same video twice.

Supporting signals used to classify them (not stored in the schema, recorded
here so the labels are checkable):

- **ref1** — a face was detected in 5% of sampled frames, and when detected it
  filled 4.9% of the frame. Mean optical-flow magnitude 3.13. That is a
  product/hands montage, not a presenter.
- **ref2** — a face was detected in 57% of sampled frames filling 28% of the
  frame, continuously until 8.1s. Mean optical-flow magnitude 1.94. That is a
  sustained close-up talking head. The cut at 8.13s has a histogram distance of
  0.994 — effectively a full-frame replacement — and face detection stops there,
  so it reads as a product end card.
- **caption position** — estimated from the vertical distribution of bright,
  high-gradient pixels. ref1 peaks in the 70–80% band (`center_lower` is the
  nearest `Y_BY_POS` value); ref2 peaks in the 40–60% band (`center`).

## Hand-edited fields

`extract_style()` leaves several fields at schema defaults because it never
measures them. Those were corrected by hand, which is what the extract command
tells you to do: the contract is the schema, not the extractor.

Both styles:

- `camera.shot_size_dist` — the extractor never measures subject scale. Set from
  the face-area ratios above.
- `caption.pos`, `caption.max_chars_per_line` — set from the caption-band
  estimate and the shot length (0.76s reads far less text than 4.04s).
- `hook_style.shot_size` — the extractor defaults to `CU` regardless of the
  reference. The archetype pool the hook actually samples from is not a style
  field at all; it lives in `configs/archetypes.yaml`.

`beat_pattern` was also changed on both, to make the storyboard's derived shot
count match the measured one:

| | body beats | storyboard emits | measured | delta |
|---|---|---|---|---|
| `biodance_asmr_montage` | 3 (`apply`, `reveal`, `cta`) | 5 + 9 = 14 | 14 | 0 |
| `biodance_ugc_talkinghead` | 2 (`context`, `payoff`) | 1 + 2 = 3 | 3 | 0 |

With the default 5-beat pattern, ref1 came out at 17 shots against a measured
14, and ref2 at 5 against a measured 3.

## Note on render cost for `biodance_asmr_montage`

14 shots averaging 0.76s, against a video provider whose `min_clip_sec` is 3–4s,
means roughly 56–68s of generated footage is paid for to keep 10.7s. About a 16%
yield in `per_shot` mode. Prefer `--render-mode multi_shot` for this style and
check `cut_timing_drift` in the QC report, since multi-shot moves cut positions
from file boundaries into the model's output where they are requested rather than
guaranteed.

`biodance_ugc_talkinghead` has the opposite profile: 3 shots of ~4s each sit at
or above `min_clip_sec`, so `per_shot` wastes almost nothing.
