# H200 failure recovery, version3

CPU audit: `h200/failure_audit_20261020_v3/report.md` and `audit.json`. It covers63 rejected J6/J8/J10 captures, both unsuccessful earlier conditioning pilots and the rejected J4 native fit. All original measurements/gates remain intact.

## What can be repaired or investigated

- J6 context teardown and J7 excess profiler ranges were repaired in version2; no repeat captures are scheduled.
- B8 clocks often start around1830MHz after prefill and finish at1980MHz. Earlier25–60ms register/FMA conditioning did not fix this. Version3 tests a different program: one fixed second of private-buffer SM read/write traffic after the verified prefill barrier and before decode timing/sensor start. This changes cache history and adds first-token latency; it does not repair the original unconditioned latency, reduce measurement exclusions, or alter clocks/power settings.
- B1/C4096 exact-output mismatch needs score/cache diagnosis. Request top20 vLLM logprobs, verify whether that diagnostic reproduces the original vLLM output, then collect three warmups/three HF trials for static/dynamic caches under SDPA and eager attention. Save full first-difference HF logits and common-prefix comparisons. Different backend agreement cannot replace the original exact gate. This case receives no timing expansion.
- J4's shared overlap coefficient fails the10% held-out gate, with one held-out point also outside the coupled feasible interval. No implementation bug was established. Refitting to an exposed holdout is not validation; retain the rejection.
- Graph/context restriction and Hopper async/wait replay have no established safe adapter fix here. No identical retries or full trace captures are scheduled.
- The private archive destination requires authorized access; no repeated upload attempt without that change.

## Submitted sequence

Use `batch/h200_failure_recovery_v3.sbatch` / `configs/h200_batch_failure_recovery_v3.json`. Fresh inventory, single allocated UUID, pinned BF16 model/TP1/seed20261013 and smoke precede:

1. C4096 correctness/score/cache diagnostics.
2. Fresh B8 HF/eager/graph exact references.
3. One conditioned eager clock pilot and one graph clock pilot.
4. If the eager pilot passes: bounded idle/memory100/arithmetic100 fixed-partition pilots, then only a passing route expands to J6 B8 three acquisitions/ten trials per treatment. Separately, none/empty/complete event pilots gate J8 B8 three-by-ten collection.
5. If the graph pilot passes: a three-trial long-window energy pilot gates J10 bridge B8 three-by-ten acquisition.
6. CPU treatment analysis, consolidation, figures and final Git checkpoint. No unresolved archive upload is retried.

No pilot is retried to search for a favorable clock sample. Existing <=5% complete-window and pooled-family clock gates, process/source isolation, exact outputs and raw-energy cadence/agreement checks remain binding. All new results belong to the separately labelled `memory_settle_v3` program and allocation; no pooling with original unconditioned measurements. Higher-model clock failures are not swept before this representative route is validated.

Commits/pushes occur after each pilot/acquisition and job disposition. Failed diagnostics and partial runs are kept. Actual GPU behavior is unverified until allocation and pilot execution.

API reference for the score diagnostic: [official vLLM sampler documentation](https://docs.vllm.ai/en/stable/api/vllm/v1/sample/sampler/). The pinned runtime remains vLLM0.22.1; unsupported diagnostic API behavior produces a retained failure.
