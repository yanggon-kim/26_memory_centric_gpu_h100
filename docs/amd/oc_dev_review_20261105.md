# OC6 development review and freeze — MI300X job 502805, MI350X job 502806

Plan: `docs/amd/overlap_calibration_execution_plan_20261105.md` §6; pilot review: `docs/amd/oc_pilot_review_20261105.md`.
Release cbd16bae; 44/44 steps rc 0 on each cohort (8 min each). Fit: `scripts/amd/oc/fit_oc6.py fit` on `$WORK` copies
(devel CPU node). Frozen models (committed before any held-out acquisition):

| Cohort | file | sha256 |
|---|---|---|
| MI300X | `docs/amd/oc/oc6_frozen_mi300x.json` | fd6486c901e5856f696698b6675d14191546c54bdda0fa96dd99f565307a3fb2 |
| MI350X | `docs/amd/oc/oc6_frozen_mi350x.json` | 9c9381c58993daf52cbf24e82293293ad1a9428fcb48aeda57209f2853d53ff1 |

Development cells: Qwen32B and Qwen1.5B × 5 shapes × 4 (mask, grid) pairs = 40 per cohort; 3 acquisitions × 10
trials; acquisition spread ≤ 2 % except the smallest Qwen1.5B shapes on MI350X (5.0–8.4 µs, launch-floor regime).

## Issued MFMA (amendment D1)

Per-dispatch SQ_INSTS_MFMA (`oc6_counts`, one call per shape and grid) is independent of grid and of mask (checked at
n_p) and equals **4·m·⌈k/1024⌉ exactly** on all 30 points per cohort (wvSplitK template: 64 threads × 8 elements ×
unroll 2 = 1024 k per step, 4 issued MFMA per row and step). The pre-registered form a·m·k + b·g, and an extended
linear form, leave up to 70 % residual (Qwen1.5B k = 1536 pads to 2048). The exact form is frozen; the held-out job
re-measures counts for new k (3584, 4096, 14336, 13824, 18944) as a check.

## Pre-registered model M1: O + C + M − η·min(C, M) — fails in development

| | MI300X | MI350X |
|---|---|---|
| η (log-LS, grid [−3, 3]) | 2.26 (95 % profile 1.52–2.97) | 3.00 (at the grid edge; profile 2.57–3.00) |
| cells within 10 % (in-sample) | 19/40 | 25/40 |
| median signed error | +6.6 % | +5.2 % |

η > 1 has no overlap meaning: for the large shapes O_ref + M alone already reaches or exceeds T (e.g. MI300X Qwen32B
gate_up at (128, 128): M = 137.4 µs, T = 135.9 µs), so η absorbs a misplaced additive launch floor. The (O-scale, η)
trade-off in the frozen file shows η falling as O is scaled down — the two are not separately identifiable.

## Identifiability of η from timings

C/M on the development cells is 0.04–0.07 and C ∝ M (both ∝ m·k). In-sample, models with η = 0 (max(O, κ(C+M))),
η = 1 (max(O, κM)) and fitted η are indistinguishable (MI300X 24/40 each; MI350X 14/40 each). **η is not identifiable
from kernel timings for this family on these GPUs**; the residual is dominated by the launch floor and by the kernel's
streaming efficiency relative to the matched read reference (B_ref at grid 128 is below what wvSplitK reaches).

## Time-domain overlap (OC4 candidate B, per traced CU)

| | MI300X (full / n_p) | MI350X (full / n_p) |
|---|---|---|
| A_C/T over 10 shapes | 0.034–0.064 / 0.051–0.170 | 0.050–0.171 / 0.055–0.204 |
| A_M/T | 0.59–0.997 / 0.65–0.996 | 0.60–0.995 / 0.54–0.996 |
| **X/A_C** (MFMA time overlapped by an outstanding load) | **0.937–0.999** / 0.958–1.000 | **0.938–0.999** / 0.946–0.999 |

On every development shape and mask, 94–100 % of MFMA execution on the traced CU happened while a VMEM load was
outstanding: per CU, compute is (almost) fully hidden. The lowest values are the small o_proj/qkv shapes where
A_M/T ≈ 0.6 (launch/reduction phases without loads). Scope: one CU per SE, not device HBM activity.
Caveat kept: the split traces at 4 and 8 workgroups per CU on CUs 0 and 1 again received only MFMA workgroups, so the
"X = 0 by construction" check (iii) remains degenerate.

## Amended model M6 (frozen alongside M1)

M6: T = sqrt(O² + (κ·M)²), one κ per cohort, no η (O as a floor that merges smoothly with the streaming time).
In-sample: MI300X κ = 1.0965, 28/40 within 10 %, median −0.6 %; MI350X κ = 0.985, 23/40, median −2.7 %. Worst cells
are the small Qwen1.5B shapes (down, o_proj) in the floor-to-stream transition. M6 is a development amendment, not
the pre-registered model; both are judged on the held-out cells with the unchanged acceptance rule (≥ 90 % of cells
within 10 %, median signed error within ±5 %). Neither model is expected to pass from the in-sample numbers; the
held-out job is run as pre-registered and the result reported either way.

Held-out job: `plan_oc.py --kind heldout` (Qwen2.5-7B, Mistral-7B × 5 shapes, Qwen2.5-14B gate_up/down at the four
development pairs plus MI300X (224, 224) / MI350X (160, 160), fresh references, 3 × 10; Qwen2.5-7B graph decode trace as
the secondary transfer check). Estimated 0.23 + 0.32 units.
