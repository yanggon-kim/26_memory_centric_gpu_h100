# H200 correctness recovery v4

The v3 allocation 14215309 failed at the B8 exact HF/engine reference and at a CPU analysis import. The import was fixed in e77f9125; CPU analysis, consolidation and figures have now actually run. Original batch failures and outputs remain unchanged. The audit is `h200/20261020_job14215309/analysis/correctness_audit_v4/`.

B8's first disagreement is generated token 6, request 0: HF SDPA/StaticCache selects 4; vLLM eager and graph select 15. All three saved producers are internally deterministic and share the prompt. The cause of the valid SDPA route's disagreement is unresolved.

## Confirmed diagnostic bug

The legacy eager/StaticCache diagnostic passed a Boolean 4D keep mask into Qwen2 eager attention. Transformers 4.57.6 returns supplied 4D masks unchanged; Qwen2 eager attention adds that mask to scores. Unused cache slots therefore were not properly excluded. Its reported agreement with vLLM is invalid correctness evidence. SDPA accepts the Boolean keep mask; this bug does not explain or repair the production SDPA disagreement.

`make_decode` now uses an additive mask for eager attention, with zero on allowed entries and the model dtype's minimum on excluded entries. Its default SDPA Boolean path is unchanged. The CPU regression verifies that high scores in unused slots receive zero attention mass. No prior timing cohort is relabelled.

## Bounded GPU sequence

`batch/h200_failure_recovery_v4.sbatch` uses `configs/h200_batch_failure_recovery_v4.json` and the name `yg_h200-fix-v4`. It binds a fresh single H200 allocation and frozen sources, preserves the prior evidence and checkpoints/pushes each acquisition/stage.

1. Inventory/runtime and existing C128 functional smoke.
2. B8/C4096 and B1/C4096 correctness diagnostics. A separately scored vLLM eager execution is checked against preserved output identity. For each condition, six predetermined HF routes run three warmups/three diagnostic trials: SDPA/static Boolean, SDPA/static built-in mask, SDPA/dynamic, eager/static additive, eager/dynamic, and SDPA/static built-in mask with math dispatch. These are diagnostics, not latency comparisons or route selection.
3. At the earliest differing token across all requests, force every request's vLLM prefix before evaluating logits. Save raw BF16-derived logits, top log probabilities, masks/cache/dispatch labels and exact outputs. This avoids comparing logits after divergent prefixes. Alternate-route agreement cannot promote the original gate.
4. Fresh unchanged HF SDPA/StaticCache versus ordinary vLLM eager/graph exact-output references. Mismatches are saved as rejected scientific gates. No clock pilot or B8 sweep runs when this gate fails.
5. Bounded conditioned eager/graph clock pilots, then conditional J6/J8/J10 pilots and three independent acquisitions/ten trials. Original rejection policy, <=5% clock envelope and isolation gates remain unchanged. Memory conditioning remains a separately labelled program, without clock/power settings changes.
6. CPU consolidation and figures. If the strict reference and same-allocation graph pilot passed, run the existing predeclared decoder calibration v4 in this allocation. Otherwise save calibration as blocked before acquisition. A separate calibration allocation with a missing prerequisite is unnecessary.

The held standalone calibration job 14227806 is superseded only after the replacement is successfully submitted. No other researcher's jobs are changed. The four training/two held-out conditions, parameter freeze/push before validation, <=10% prediction error gate and all physical-overlap limitations in `h200_decoder_calibration_v4.md` remain binding. A persistent correctness failure stops expansion; this implementation cannot claim a numerical cause or successful GPU repair before execution.

## Remaining independent limitations

The original unconditioned clock rejects, rejected native held-out model, unsupported graph SM restriction, incomplete Hopper trace semantics/calibration, and inaccessible private archive remain preserved. No identical rejected sweep or private-archive retry is scheduled. Service-reference prediction, physical time decomposition and physical architecture benefit remain distinct claims.
