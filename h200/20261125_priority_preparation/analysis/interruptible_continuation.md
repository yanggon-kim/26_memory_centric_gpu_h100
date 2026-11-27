# Sequential two-H200 continuation

The owner explicitly authorized repeated two-H200 Slurm submissions after preemption until this campaign is complete. Eight hours is an upper runtime limit; Embers can be preempted after the cluster's one-hour exemption. Confirm terminal Slurm state, preserve accounting and actual GPU-hours, close out partial evidence and verify a remote checkpoint before each replacement. Review the runbook queue and reuse completed results within their qualified scope; a completed core curve must advance the next job to unfinished eligible work. Do not create simultaneous replacement campaign jobs.

Current queued replacement:16184819. First allocation:16163464,PREEMPTED,3633seconds,2.0183333333GPU-hours. The replacement launcher prioritizes the target optimized B1/C4096 smoke/HF gate and broad SM curve; its second device prioritizes the missing physical traffic comparison. The prior real-KV record is historical software validation only, not a fresh-pair handoff claim.

For each new allocation inventory the actual UUIDs and deadline, create a fresh run root and numerical source freeze, and run the required correctness/placement/source/clock/isolation gates. Keep failed and partial captures. Preemption is not evidence that SM restriction failed. Never pool fresh physical devices into an old timing cohort. Publish each complete acquisition bundle/curve and keep raw artifact hashes on persistent storage.

After every interruption, select and validate a concrete next launcher from actual saved evidence. Repeated submission permission does not authorize blindly repeating already completed comparisons. No more-than-two-GPU work or system settings change is authorized here.

Wall-time revision: while still pending, job16184819 was reduced from8h to3h with `scontrol update JobId=16184819 TimeLimit=03:00:00`, verified in scheduler state. Stock target processes took77s clean/89s profiled; applying the conservative curve admission model gives approximately2h for the timing comparison before proofs and closeout. This is a planning proxy, since restricted pilots are not measured yet. The real pilots govern admission. Future bounded comparisons can request shorter slots when their complete cost fits; shorter limits do not change preemption exemption.

## Terminal-allocation closeout

After Slurm reports terminal state and the priority run root exists, save accounting (including the parent job row) under that root. Run `python scripts/h200_review_allocation.py --root <actual-priority-root>` using the pinned runtime. It requires terminal accounting and actual allocated GPU count; it never launches CUDA work. It verifies saved case hashes, UUID/source/producer/isolation metadata before rebuilding curve summaries, retains interrupted process records, and writes `analysis/allocation_closeout.md`, `results/allocation_closeout.json` and a separate reviewed case manifest. Native analysis runs only when all18 producer outputs are present; physical counter captures remain separate from unqualified timing/L2/kernel joins. Original producer outputs and chronological notes are preserved.

Then inspect the specific outstanding gates, choose a concrete next comparison/launcher and commit/push the reviewed root before the next sequential submission. Bootstrap failures without a priority inventory require separate staging review; they cannot be treated as measured cohorts. A complete acquisition set rejected by clock/source gates is retained as a result and is not repeated solely to obtain favorable conditions.

## Staged curve admission

The curve now admits one complete broad acquisition first, using one third of the existing conservative instrumented-pilot estimate. Each broad acquisition still has stock plus full-green controls, all five cuts, interleaved/end references,3warmups and10timed trials per producer. After that first acquisition, the planner checks whether **all remaining acquisitions** fit using the maximum measured acquisition wall time,25percent margin and100seconds per remaining acquisition for publication/analysis. It re-queries the actual Slurm launch cutoff at every stage. Measured wall time includes case supervision, process startup, idle checks and analysis; it is a scheduling input, never an inference-latency result.

This avoids multiplying a cold compilation/profiler startup across the entire sweep before learning the warm cost. It does not relax any gate or call a1/3acquisition curve complete. Every completed bundle is checkpointed, a budget stop remains partial, and no other batch/context is expanded before the core curve qualifies. Future requested wall times will use these observed costs plus setup/proof/closeout allowance; the existing three-hour pending request remains in place.

CPU contracts cover: a first bundle that fits despite an inflated cold full-curve estimate; successful remaining bundles after measuring lower warm cost; no timing acquisition when the first bundle cannot fit; and retention of the completed first bundle when remaining work cannot fit. Scientific qualification/fallback contracts also remain passing.

## Closed-allocation archive

After final CPU closeout/report edits, use `python scripts/archive_h200_priority.py --root <closed-priority-root> --destination /storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives`. The bounded default is256MiB of source files; inspect the storage budget before increasing it. It copies no symlink targets/model cache, preserves original regular-file bytes in a canonical archive, verifies every member, checks the transferred archive hash, and never overwrites a differing archive. The versioned receipt and source manifest describe the captured snapshot; later presentation updates remain separately versioned in Git.

Job16163464 archive:644 members,58781099 original bytes,5489524 compressed bytes. SHA256 `7ffa51f08acb6e3574acc8b89c5a1294b642e532960900a1637f14e4fa83ac38`. Destination `/storage/project/r-dmahajan7-0/jkim4112/h200_priority_archives/20261126_job16163464_priority_98f67d519f20.tar.gz`. Every archive member and destination checksum verified at2026-11-26T23:27:25.048264+00:00. Project filesystem free space was323GiB before this5.5MB transfer. Original workspace files are retained.

## Bounded capture-stream repair v2

The pinned vLLM0.22.1 source creates a capture stream and maintains its own stream tracker. A prepared adapter seeds that tracker after import and supplies the owned green stream to both existing tensor/pipeline group capture contexts. Package files are untouched. This is a source-based repair hypothesis; no new decoder placement result is claimed. See `../results/capture_stream_repair_preparation.json` for source hashes and CPU validation.

The initial optimized proof runs first. If it fails a required gate and has an allocation record, the single bounded repair cycle uses `--rebind-step --bind-vllm-capture`, labeled `capture_bind_v2`. It repeats stock and every supported green-count proof; full-green/restricted timing arms use the same repair route. Every measured decoder kernel must map to the driver-verified green context, graph nodes must be present, and exact outputs must pass. Mixed repair routes cannot qualify a pooled curve. If the repair remains blocked, collect the explicitly labeled same-engine eager curve and continue eligible full-GPU work. A clock-rejected completed acquisition set remains a separate quality outcome.

Six CPU fake tests verify explicit stream routing, preservation of both communicator contexts, restoration on success/error, rejection of changed aliases/unsupported releases and rejection of another device index. They do not validate real CUDA graph behavior. The allocated bootstrap also runs these contracts and the staged-budget contracts before measurement.

## Actual grants and complete references

Full stock and full-green controls must receive the full device resources; a cut must actually reduce the granted SM count. Five saved real stock controls satisfy the new metadata gate. Qualification also requires all three interleaved and all three end references, with full-control SM counts equal to stock. A requested fraction alone never proves restriction.

The terminal closeout requires ten timed trials for every captured producer and all drift references before calling the acquisition set complete. CPU tests reject a synthetic preemption before the last end reference, mismatched full controls, an ignored SM cut and mixed repair routes. The legacy32B evidence remains incomplete/clock-rejected; these forward checks change no measured data and claim no new NVIDIA threshold.
