# Required H100 evidence: more on-chip memory at fixed bandwidth

**Scope update2026-10-16:** these requirements remain the retention/local-resource branch of [the H200 shoreline assignment](h200_shoreline_jobs.md). They do not replace SM/request-delivery, independent service or joint-bandwidth experiments. Preserve actual H200 baseline/cache identity and collect new matched comparisons under the three-acquisition/ten-trial protocol. Existing evidence is not retroactively qualified.

Read this after `RENTAL_RESUME.md` and before collecting Phase 1 profiles or Phase 2 traces. The companion [HTML explanation](fixed-bandwidth-serving.html) teaches the model. The primary research specification and existing correctness/energy gates still apply. These collection requirements are new; no previously completed run is retroactively declared to satisfy them.

## Question and mathematical contract

Test whether useful on-chip residency or improved utilization can reduce decode time while the external memory interface stays unchanged. Do not change clocks, power limits, driver, GPU allocation, precision or tensor parallelism to manufacture a fixed-bandwidth comparison. Record observed clocks, temperatures and concurrent activity; identical hardware does not guarantee identical achieved bandwidth.

For a declared interval, partition traffic according to the proposed intervention:

```text
D0 = D_fixed + (1-h0) D_reuse
D1 = D_fixed + (1-h1) D_reuse
memory-limited service ratio = D0/D1                 [same effective bandwidth]
ideal dedicated residency h(C) = min(C/W, 1)
memory service time ≈ D_H / (eta × B_peak)
step time ≳ max(F/C_compute, D_H/B_H, D_P/B_P, T_other)
controlled decode throughput = B / step_time
```

`D_fixed` means traffic unaffected by the specified intervention, not inherently all weights. `D_reuse` means reusable requested bytes before accounting for hits; `W` is their working-set size. Repeated requests can make `D_reuse` larger than `W`. `h` is a byte-weighted residency fraction for that class. An aggregate L2 sector-hit percentage is not `h`. The max expression is a simplified overlap bound, not an additive prediction of serial layers. The ratio predicts actual speedup only if delivery remains limiting and other costs are unchanged.

H100 cannot emulate arbitrary additional physical SRAM simply by allocating more HBM. Hardware measurements characterize baseline traffic and limits; capacity sweeps from traces remain analytical until a calibrated simulator/design evaluates them. A software policy or alternate kernel on real H100 is a separate measured control and needs exact implementation/configuration provenance.

## 1. Required allocation and case identity

For every selected case, collect actual SKU/UUID, SM count, queried L2 and per-SM register/shared-memory limits, HBM size, runtime/toolkit/driver, actual backend and FA version, model/config/binary hashes, BF16 weight/KV formats, layers/Hq/Hkv/head dimension, graph/eager policy, prefix-cache policy, batch, prompt IDs/hash and workload, generated/timed counts, and actual per-request query/KV lengths at each sampled step. Record observed memory/core clocks, power, temperature, compute processes and any drift around baseline/profile/control intervals.

Preserve the original staged portfolio in `configs/models.json`: bridge and Qwen7B/Mistral7B B1/4/8, Qwen14B B1/4, conditional-fit Qwen32B B1, and their existing context cases. Start with bridge B1/C128, then select B1/C1024 and B8/C4096 where supported. Prioritize matched small-batch and longer-context larger-model cases after their smoke passes. Additional long-context or larger-batch cases need new recorded configs and actual HBM/workspace/storage fit; do not silently replace missing cases or quantize.

From each resolved immutable configuration, calculate all-layer logical KV payload:

```text
W_KV = N_layers × 2 × bytes_per_element × B × Hkv × L × d
F_attention ≈ 4 × B × Hq × L × d                    [per layer, ordinary decode]
I_KV = F_attention / D_KV = 2 Hq / (bytes_per_element Hkv)
```

These are analytical quantities. Verify tensor storage/strides, allocation padding, aliasing and actual accessed lengths separately. Do not turn allocated capacity into measured traffic. For shared prefixes, count distinct physical storage and per-request accesses separately.

## 2. Required uninstrumented performance and energy

Keep three complete warmups and three valid trials, exact greedy output IDs, the all-request one-token prefill boundary and N−1 timed decode ticks. Collect synchronized wall and GPU-event elapsed, per-step batch latency, aggregate decode tokens/s and denominator B×(N−1). Keep prefill, first token, startup/compilation, scheduler work and decode distinct. Preserve graph-optimized and trace-compatible eager controls with separate labels.

Measure qualified decode-only whole-board energy and raw power/counter estimates in long windows using the existing gates. Report J/token as NA when qualification fails. Clock and temperature observations are required for comparisons. Do not infer SRAM/HBM energy or area from whole-board energy.

If request-level latency distributions or a real server workload are added, save request arrivals/completions and TTFT/TPOT definitions, load/concurrency, prompt/output lengths and server settings as a separate workload. Existing batch-step averages do not supply request p99 or queueing latency. A controlled decode gain alone is not a full serving gain.

## 3. Required kernel and traffic collection

Discover actual decode kernels with Nsight Systems before selecting a bounded Nsight Compute set. Use NVTX/step IDs and request lengths to identify attention forward, split combine, KV update, QKV/output projections, feed-forward projections and remaining work. Preserve early/middle/late layer examples where feasible; never generalize one first-layer sample to the full model without validating representativeness.

For each selected kernel/range, collect the following. Query the installed metric catalog and record exact names, units and active/elapsed denominators. Examples are search hints, not portable guaranteed metric names.

| Required quantity | Purpose and scope |
|---|---|
| DRAM read/write bytes and duration | Actual external traffic; example `dram__bytes_read.sum` / `dram__bytes_write.sum` when supported. Preserve reads and writes separately. |
| DRAM achieved bytes/s and percent sustained | Distinguish reduced traffic from higher utilization. Preserve the hardware metric's denominator; a copy ceiling is a separate control. |
| L2 request/sector counts, hits/misses and traffic | Identify cache pressure and redundant requests. Preserve counter definitions; do not equate sector hits with retained KV bytes. |
| Forward/combine/cache-update grid, block, launch count | Separate CUDA launches, physical CTAs and logical work. FA3 persistent grid alone does not reveal work count. |
| Registers/thread, allocated registers, local-memory load/store traffic | Detect register residency limits and actual spills; a high register count does not prove spilling. |
| Static/dynamic/driver shared allocation and occupancy limits | Record constraints from shared memory, registers, warps, blocks and barriers; do not assume larger storage improves useful occupancy. |
| Achieved occupancy, relevant stall reasons, tensor and LSU activity | Distinguish delivery, latency/MLP, issue and compute limits. Keep each metric's denominator. |
| Shared-memory traffic/bank conflicts and available cache/LSU pressure counters | Identify downstream limits that could replace an HBM bottleneck. Unsupported metrics stay NA. |
| Actual attention tile/stage specialization and split metadata | Record BM/BN, GQA packing, static split upper bound, per-request dynamic splits and productive/empty intervals. If inaccessible, save source/binary identity and explicit NA. |

Record profiling command, source log/report hash, clock/cache control, replay mode/pass count, filtering and layer/launch range. Replay can alter residency. Do not use replayed-kernel duration as ordinary end-to-end timing. If supported, collect a bounded whole-decode-range traffic measurement with clear range boundaries; kernel sampling sums remain a separate estimate with coverage/uncertainty, never a measured full-step total.

Achieved bandwidth computed as bytes/duration is descriptive. Substituting that same value back into bytes/bandwidth reproduces duration by identity; it is not an independent performance prediction. Compare against separately measured workload-appropriate delivery controls, stalls, resource limits and matched interventions. A contiguous useful-copy ceiling is not universal attention bandwidth.

If `ERR_NVGPUCTRPERM` persists, save the denial and exact unavailable fields. Continue independent baseline, timeline and trace work, but keep the hardware-traffic bottleneck and incremental HBM-saving gates incomplete. Do not substitute allocation sizes, useful-copy bytes, token counts or NVBit loads for DRAM counters.

## 4. Required reuse and trace evidence

Collect at least two correct whole-model decode steps for the bridge after the native micro pilot and budget checks. The intervening projection/MLP/other-layer stream must remain present; isolated repeated attention or one layer creates artificial reuse. Extend selected larger-model traces only within measured storage/runtime budgets.

Export parameter ranges/tied aliases, per-layer KV storage/page tables and request-to-page/slot mapping where accessible, actual strides/dtype, KV allocation changes, temporary/activation/split-scratch ranges where identifiable, decode step IDs, query/KV lengths and generated output IDs. Unmapped ranges need explicit labels. Validate address attribution on actual captured records before applying a retention policy.

Trace/analysis must distinguish:

- requested load/store bytes, active masks and repeated accesses;
- distinct weight/KV/activation/scratch working sets;
- byte reuse distances across CTAs, layers and successive full decode steps;
- alias/prefix sharing and new KV writes;
- TMA/async/barrier semantics and the addresses/descriptors actually preserved.

These are requested/logical quantities; counters or a calibrated physical memory model establish HBM transactions. NVBit warp records are not active-thread instruction counts. Existing TRYWAIT-drop warnings remain unresolved and cannot support a calibrated FA3 timing claim.

Save separate analytical capacity curves for ordinary replacement and controlled weight/KV residency, with explicit baseline L2, capacities, granularity, access order, warm/cold initial state, writes and consistency assumptions. Include practical bins such as 128/512/1024 MiB as sensitivity inputs, not claims of feasible area. Charge existing cache/prefix/GQA benefits before incremental savings. Do not assign an invented HBM hit rate to the logical model.

## 5. Required matched controls and interpretation

Compare valid baseline cases with graph/eager execution and representative text at the same hardware settings. A supported split policy, kernel tile or cache-residency control may be added as a distinctly pinned configuration only after correctness. Some APIs reject explicit split overrides; use actual supported interfaces and record the change. Prefix-cache controls need a separately validated measurement boundary; the existing zero-prefix-hit baseline contract cannot be silently weakened.

Separate three hypotheses:

1. **Traffic reduction:** fewer actual off-chip bytes with the same model/request interval; determine which classes change and include control overhead.
2. **Utilization:** comparable traffic but improved achieved bandwidth/latency hiding through occupancy, MLP or scheduling. This does not increase the hardware's peak bandwidth.
3. **Different kernel work:** tile/split/fusion changes alter instruction count, scratch and useful/empty work. Record these changes; a cache-only trace cannot represent a retiled kernel.

Report attention-forward, combine and cache-update effects separately, then the entire attention sublayer including projections, then full decode/serving scope. Use separate denominators for GPU busy-time shares and synchronized wall-time shares. Overlapping summed kernel times cannot provide a wall-time Amdahl fraction. Amdahl estimates must name which fraction and unchanged costs they assume.

For less-compute proposals, collect achieved compute evidence and report the conditional threshold `C_required ≥ I_KV × effective_KV_delivery_bandwidth`. It ignores softmax/issue inefficiency and does not establish a removable Tensor Core fraction. Keep a larger-batch counterexample. No fixed-area/energy claim follows until SRAM/register/fabric PPA and workload calibration exist.

## 6. Mandatory deliverables and acceptance

Under each fresh rental result root, leave:

- `fixed_bandwidth_case_manifest.json`: selected/completed/failed/unavailable cases and links/hashes to raw evidence.
- `fixed_bandwidth_measurements.json`: case/layer/step/range identity and each quantity's value, unit, classification, raw source and denominator. Missing quantities use null with a reason; they are not zeros.
- `attention_launch_inventory.json`: observed backend/version, tiles, resource allocations, splits, useful/empty work and launch identifiers, or explicit unavailable reasons.
- `traffic_attribution_report.md`: measured totals versus sampled estimates versus trace/logical requests; weight/KV/scratch/unknown coverage and discrepancies.
- `residency_opportunity_report.md`: whether traffic, utilization or neither supports an opportunity, including negative cases, fixed-bandwidth limits, cold/warm behavior and whole-model implications.

These new artifacts require collection/adaptation by the future agent; current scripts do not emit all of them. Keep the Section 9 CSV and existing run/trace manifests as well. Update STATUS and the calibration table from actual evidence. Preparation of this checklist does not mark a measurement gate complete.

Before rental termination, archive verified raw traces, sanitized profiler data, metadata and analyses, then verify durable remote checksums. Preserve the existing HF archive and failures. Full fixed-bandwidth validation needs correct optimized baselines, actual traffic/counter evidence, trace attribution and calibration, plus observed full-step behavior. Report unsupported gates as pending rather than declaring architecture success.
