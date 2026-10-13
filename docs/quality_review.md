# Independent portable H100 package quality review

Reviewed current scripts, model/configuration registry, README, AGENTS contract, playbooks, measurement boundaries, and trace/export contracts. Review was CPU-only: no GPU execution, download, installation, simulator rebuild, or new simulator replay. Only this review document was written. The package prepares future research; no native H100 execution or calibration is certified.

## Verdict

**PASS as a portable preparation package with pending native H100 validation.** The bounded measurement, trace, and export contracts are suitable for shipping with their documented evidence limits. Final source refuses modified tracked framework/backend trees before launching capture, resolving the source-provenance gap. Materialization now requires positive L2 bytes, and cache constructors/analytical CLI reject negative capacities. No material blocker remains in this CPU review.

## Independently executed CPU checks

- validate_package passes all 23 required portable files, Python parsing, shell syntax, status gates, and the implemented host-path scan.
- test_cpu_contracts passes negative replay cases for unsupported-exit-zero, missing instruction statistics, unequal cycle/instruction blocks, and incomplete kernel count.
- Independently exercised real CPU tensor aliases and a noncontiguous storage view: base/tied/view storage identity, byte range, element offset, strides, and logical-versus-allocated bytes are recorded consistently.
- Offline materialization of the actual immutable 1.5B bridge snapshot passes; trying to identify that snapshot as the 7B model fails. Unsupported dense family and active sliding-window analytical adapters are rejected.
- Final CPU contracts also reject negative cache capacity, negative added storage, negative analytical CLI capacity, and negative materialized L2 bytes. Independently reran those tests after the source fixes; the final 23-file package validator also passes.
- Copied and postprocessed the genuine dynamic RTX fixture in a temporary owned output. The export records 20,224 reported warp instructions, positive grouped output, and binary version 120. An earlier genuine range fixture also passed with 3,584 records. Neither is relabeled Hopper; neither proves H100 execution.
- Parsed the existing genuine V100 public replay log: ten kernel statistics sections, 127309 cycles and 418884 active-thread instructions. This checks parser/completion integrity and does not certify H100 calibration or functional replay output.

## Measurement and resource boundaries

The shared GPU selector locks the named physical UUID, checks startup compute contention, chooses numeric CUDA visibility for vLLM, and crosschecks actual CUDA UUID and H100 sm_90 properties before execution. The lock coordinates this package's cooperating processes only; later unrelated activity remains possible and is disclosed. L2 bytes are queried from CUDA properties/driver or remain NA. No RTX capacity is automatically assigned to H100.

Controlled vLLM checks an all-request, exactly-one-token, zero-prefix-hit barrier and synchronizes before measurement. Exactly N−1 subsequent ticks produce B×(N−1) timed tokens. Full outputs, hashes, prompts, selected case identities and warmup/trial repetition are retained. Event elapsed includes launch gaps; host wall includes scheduler/check work. Eager trace timing, graph-optimized baseline timing, and StaticCache controls are distinct scopes. External-profile flags exclude instrumented timings from baseline comparison. Strict StaticCache checks remain intact.

The sensor policy retains raw estimators and requires long windows, observed cadence, and counter/power agreement for qualified J/token. Failure leaves J/token NA. Agreement is explicitly not external absolute calibration or component/system energy. Inventory writes its read-only evidence before the optional CUDA check and preserves check errors. No package operation changes drivers, firmware, clock settings, power limits, or system CUDA.

Model materialization pins immutable revisions or resolves-and-records a commit before downloading. It bounds declared download bytes, checks snapshot identity/model family and stores config hashes/layout. Runtime engines use BF16 and one visible GPU; conditional models remain fit/budget-dependent. Dense reuse rejects unsupported families, active sliding windows, MoE, and unsupported element sizes. Synthetic full-block fills, requested payload, extra exclusive pins, GQA/prefix identities and warmup denominators are separated from measured GPU traffic. Final measured Section 9 consolidation and representative/versioned prompt workloads are explicitly future work.

## Trace and export review

The eager HF route attaches actual pinned module hooks after warmups and completed prefill. Its root-forward capture includes output projection and expressly omits external argmax/copy operations. The eager in-process vLLM full route brackets the entire synchronized post-barrier engine.step, covering the separately executed LM-head/logit/sampling path. It compares full captured output IDs against an uninstrumented same-shape reference. Partial disjoint module routes remain explicitly partial. Native H100 opcode/async/barrier behavior is pending validation.

Parameter manifests record actual underlying storage bases, exclusive range ends, offsets/strides and shared aliases. KV lifecycle snapshots cover before/after each decode step. They do not infer page semantics or count accessed traffic, and the missing activation/workspace/request-to-page mappings are explicit. Allocation snapshots must be validated against actual trace addresses before semantic locality claims.

run_trace rejects inherited simulator/preload/library state, requires explicit tools, precreates nested trace paths, records binary/toolkit/runtime/config/script identities, bounds capture time and observed raw bytes, and terminates its owned process group on failure. It preserves partial failure metadata. Raw limits are sampled rather than a filesystem quota, and postprocessing/replay remain separately bounded; disk headroom and actual expansion still need observation.

export_trace discovers actual stats filenames, demands positive instruction records and nonempty grouped files, checks real architecture headers, and refuses H100 replay unless observed binary versions are exactly {90}. The replay smoke copies backend configuration into an owned working directory, uses both configurations, and validates matched positive kernel/cycle/instruction sections plus completion and rejection-message checks. It is one-kernel parser integrity, not whole-model replay, functional equivalence or same-SKU calibration.

Archive generation includes raw/grouped artifacts and manifests, writes outside the trace tree, refuses overwriting, and provides an SHA256 sidecar. A local checksum is not durable transfer; the documentation correctly requires verification at the destination before rental termination.

## Findings

1. **Resolved source reproducibility:** final scripts/run_trace.py rejects nonempty staged or unstaged tracked framework/backend differences before child launch. Current baseline captures therefore cannot silently use unreconstructable dirty tracked sources. setup_accelsim applies the same clean-source policy. Future intentional architecture changes still require an explicit reviewed policy preserving actual patches/source identity rather than merely weakening this guard.
2. **Resolved input validation:** scripts/materialize_config.py requires L2 bytes greater than zero. ByteLRU/PolicyCache reject negative capacities, and the reuse CLI rejects invalid baseline/extra capacities before processing models or producing analytical output. Final negative CPU regressions pass.

Operational Python imports resolve within the portable package or explicit third-party tools; no parent project/simulator directory is imported. Ignored environments, machine-local configs, models, traces and binaries are distinct from Git-ready templates and historical scoped RTX evidence. The native H100 execution, exact-SKU model, opcode coverage, representative workloads, full measurement table, physical SRAM/NoC/resource costs, and durable transfer remain future research gates.
