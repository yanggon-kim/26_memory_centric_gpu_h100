# OC-v0 results — compute/memory overlap calibration on MI300X and MI350X (decode projection family)

Assignment: `docs/amd/overlap_calibration_followup_20261105.md`. Plan: `docs/amd/overlap_calibration_execution_plan_20261105.md`.
Reviews: `docs/amd/oc_pilot_review_20261105.md`, `docs/amd/oc_dev_review_20261105.md`. Frozen models:
`docs/amd/oc/oc6_frozen_{mi300x,mi350x}.json` (commit dd6cb0ef, before any held-out acquisition). No clock, power,
driver or firmware change; frozen G8 predictions untouched. MI325X: no OC jobs (no single-GPU MI325X partition).

| Job | MI300X (mi3001x) | MI350X (mi3501x) |
|---|---|---|
| pilot | 502623, 45/45 | 502625, 45/45 |
| development | 502805, 44/44 | 502806, 44/44 |
| held-out | 502898, 68/68 | 502899, 68/68 |

## Headline

1. **Observability (OC0/OC4).** Dispatch-aggregate "activity" counters (SQ_ACTIVE_INST_*) are instruction counts, not
   time, on gfx942 and gfx950 → candidate A unavailable. Thread trace (ATT) gives per-instruction timelines on one CU
   and passes the activity checks → **per-CU time-domain compute/memory activity is bounded and measured**; device-level
   HBM activity remains unavailable.
2. **Overlap, time domain (per CU).** On every development shape (Qwen32B, Qwen1.5B × 5) at full and reduced masks on
   both GPUs, **94–100 % of MFMA execution overlaps an outstanding VMEM load** (X/A_C = 0.937–1.000). Compute occupies
   3–20 % of the CU window; memory is outstanding 54–99.7 %.
3. **Overlap, from timings (OC6).** η is **not identifiable** from kernel timings for this family: C/M = 0.04–0.07 and
   C ∝ M. The pre-registered model O + C + M − η·min(C, M) fails in development (η > 1) and on held-out cells.
4. **Issued work.** wvSplitK issues exactly 4·m·⌈k/1024⌉ MFMA per call, independent of grid and CU mask (78 shape × grid
   points per cohort, 0 mismatches including 48 held-out points never used in the fit).
5. **Held-out acceptance (≥ 90 % within 10 %, median within ±5 %): not met by either frozen model on either cohort.**

## Held-out validation (60 cells per cohort: Qwen2.5-7B, Mistral-7B × 5 shapes, Qwen2.5-14B gate_up/down × 5 pairs)

| Cohort | Model | within 10 % | median signed | accepted |
|---|---|---:|---:|---|
| MI300X | M1 pre-registered (η = 2.26) | 40/60 | −1.3 % | no |
| MI300X | M6 amended, sqrt(O² + (κM)²), κ = 1.0965 | 53/60 (88 %) | −1.6 % | no (54 needed) |
| MI350X | M1 pre-registered (η = 3.0, grid edge) | 18/60 | +12.7 % | no |
| MI350X | M6 amended, κ = 0.985 | 39/60 | +5.1 % | no |

Per-model aggregates (sum of shapes per pair), M1: MI300X −8.7 % … +1.0 %; MI350X −2.9 % … +21.0 %. Worst cells: see
`results_published/amd_20261105_*_oc_v0_heldout_job*/validation.json`.

Failure mechanism (MI350X): at reduced grids (160, 192 workgroups) the projection kernel runs faster than the matched
streaming-read reference allows (e.g. Qwen2.5-14B gate_up at (160, 160): T = 48.1 µs vs M = W/B_ref = 60.1 µs). The
1024-thread read at g workgroups is not an upper bound on the bandwidth wvSplitK reaches on g CUs, so the memory term is
biased high at low grid; at the stock grid the bias reverses for the largest shapes (Mistral-7B lm_head −22 %). On
MI300X the same terms stay within ~±13 %. Errors are set by the memory reference and the launch floor, not by overlap.

## Inputs reported per cohort (`model_inputs` in the frozen files)

- R_mfma4 (mfma4 wave-instr/s), B_ref (matched read), O_ref (tiny wvSplitK) per (mask, grid), 3 acquisitions each.
- Issued MFMA per (shape, grid), measured and fitted.
- ATT per shape: A_C/T, A_M/T, X/T, X/A_C, loads retired without stall (0 on all routes: the m(t) estimate is tight).

## Limits and caveats

- ATT scope: one CU per SE, all SIMDs; replay/serialization as the tool records them.
- Split control (iii): degenerate in all traces (traced CUs received only MFMA workgroups).
- SQ_INSTS_VALU validated only against the ISA-amended expectation (packed FMA).
- Secondary transfer check (Qwen2.5-7B graph decode trace) acquired (`oc6_transfer`), not yet evaluated.
- OC5/OC7 synthesis follows the G1–G8 main jobs.

AMD acknowledgement for any publication: "This work was supported in part by Advanced Micro Devices, Inc. under the
AMD University Program's AI & HPC Cluster".
