# Tracks

Ordered learning paths built from existing challenges. Not deployed by any workflow; consumed by infra/frontend behind a flag.

Source of truth: `AlphaGPU/leetgpu-challenges`, branch `kernel-interview-track`.
Status: draft, unmerged; needs Kunal's review. No pricing is defined here (placeholder only, nothing in Stripe).

## 1. Track file: `tracks/kernel-interview.json` (schema_version 1)

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | int | Bump on breaking changes. Currently `1`. |
| `id` | string | Track id, `"kernel-interview"`. Use it as the key for the progress API. |
| `title`, `description` | string | Display copy (the Marketer may override it on the landing page). |
| `status` | `"draft"` \| `"live"` | Always `"draft"` until Kunal approves. |
| `enabled` | bool | Default `false`. Infra and Frontend should AND this with their own feature flag. |
| `entitlement` | string | `"interview_pass"`: the entitlement Infra adds behind a flag. |
| `pricing.placeholder` | bool | `true`. No real price lives in this repo. |
| `modules[]` | array | Ordered modules. |
| `modules[].id` | string | Stable slug, e.g. `"memory-coalescing"`. |
| `modules[].order` | int | Display and unlock order, 1-based. |
| `modules[].title`, `.objective`, `.interview_tip` | string | Card copy. `interview_tip` is one line. |
| `modules[].challenges[]` | array | Ordered challenge refs. |
| `…challenges[].id` | int | Challenge id (= directory number = backend `id`). |
| `…challenges[].slug` | string | `formatTitleForUrl(title)`, i.e. lowercase, non-alphanumerics → `-`. |
| `…challenges[].title` | string | Snapshot of `Challenge.name`. The backend title wins if they differ. |
| `…challenges[].target_speedup` | float (slow_kernel only) | Copied from `challenge.py`; matches Frontend's expected field. |
| `…challenges[].bottleneck` | enum str (slow_kernel only) | Same codes as the profile endpoint's `bottleneck.code`. |
| `…challenges[].role` | `"core"` \| `"slow_kernel"` | `slow_kernel` = a "why is this slow?" round, so show the profiler panel and speedup target. |

Aligned with `/workspace/interview-track/frontend-expectations.md`: `GET /api/v1/tracks/kernel-interview` returns 404 when off, or `{enabled, track: <this JSON, with per-challenge solved/best_speedup merged in>, progress:{solved:[slugs]}, entitlement:{interview_pass}}`. The per-submission profile (`/api/v1/interview/submissions/{id}/profile`) should return `bottleneck.code` from the challenge's `bottleneck` attribute (when the metrics agree), plus `targetSpeedup` from `target_speedup`. Module completion = all `core` challenges solved, plus `slow_kernel` challenges passed (speedup ≥ target).

## 2. Slow-kernel challenge metadata (extra `Challenge` class attributes in `challenge.py`)

The current challenge format has NO field for track membership, speedup targets, or hidden/unlisted status. `validate_challenges.py` and `update_challenges.py` ignore extra attributes, so these are proposed new fields:

| Attribute | Type | Example | Meaning |
|---|---|---|---|
| `track` | str | `"kernel-interview"` | Track membership (proposed new field). |
| `problem_type` | str | `"slow_kernel"` | Marks a "why is this slow?" problem. |
| `bottleneck` | enum str | `uncoalesced_loads`, `bank_conflicts`, `warp_divergence`, `low_occupancy`, `missing_tiling`, `atomic_contention` | Ground-truth bottleneck (for the profiler "likely bottleneck" hint and grading the explanation). |
| `target_speedup` | float | `4.0` | Required speedup over the baseline. |
| `baseline_language` | str | `"cuda"` | The baseline is the shipped `starter/starter.cu` kernel. |
| `hints` | list[str] | 3 items | Progressive hints, revealed in order. |
| `access_tier` | `"premium"` | | Placeholder until an `interview_pass` tier exists. |

`update_challenges.py` does not send these to the backend yet. It needs to add e.g. `track`, `problemType`, `bottleneck`, `targetSpeedup`, `baselineLanguage`, `hints`, and a `listed`/`hidden` flag to the payload.

## 3. What Infra must enforce (the format can't express it)

1. **Speedup grading:** on submit, time the user's solution AND the baseline `starter.cu` on the same GPU, the performance test, and the same run (or a cached baseline per GPU type). Then compute `speedup = baseline_ms / user_ms`. Pass = all functional tests correct AND `speedup ≥ target_speedup`. Use the median of several runs to absorb timing variance. Correctness alone is NOT a pass.
2. **Non-CUDA languages:** the other 5 starters are empty (repo convention). Either restrict these problems to CUDA in track mode, or grade other languages against the same CUDA baseline time.
3. **Hidden/unlisted:** nothing exists today. Merging to `main` auto-deploys every changed `challenges/**` directory into the public list (`deploy-challenges.yml`). Before merge, add a `listed=False` (or track-only) field to the payload and the public list query, or move these out of `challenges/`.
4. **Calibration:** targets are calibrated from Infra's measured T4 run (Oct 16 2026): ~0.8× the demo solution's measured speedup; 139 capped at 6.0× (measured 18.27×) as a product call.
5. **Profiler feedback:** compare the ncu metrics to `bottleneck` (e.g. gld_efficiency, shared bank conflicts, branch efficiency, achieved occupancy, dram throughput vs. FLOPs, atomic throughput).

## 4. The 6 slow-kernel challenges (all `medium`, `premium`)

| id | slug | bottleneck | target | perf test |
|---|---|---|---|---|
| 134 | speed-up-column-scale | uncoalesced_loads | 4.7× | M=N=8192 |
| 135 | speed-up-tiled-transpose | bank_conflicts | 2.0× | 8192×8192 |
| 136 | speed-up-row-sum | warp_divergence | 3.8× | M=16384, N=4096 |
| 137 | speed-up-segment-reverse | low_occupancy | 5.6× | N=67,108,864, S=8192 |
| 138 | speed-up-matrix-multiply | missing_tiling | 1.3× | M=N=K=4096 |
| 139 | speed-up-histogram | atomic_contention | 6.0× | N=50M, 32 bins |
