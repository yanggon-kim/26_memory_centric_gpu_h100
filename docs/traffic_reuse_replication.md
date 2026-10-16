# H100 replication: logical inputs versus actual off-chip traffic

**H200 override:** this exact RTX-matched seed20261014 replication is optional supporting work under [the H200 assignment](h200_shoreline_jobs.md). It must not replace J0–J11 or change seed20261013 on the active H200 portfolio. Its older H100 ordering is superseded for this branch.

**Scope/replication update 2026-10-16:** follow [shoreline_hopper_measurements.md](shoreline_hopper_measurements.md) for the governing experiment order and statistical requirements. The 18 baseline+2pressure+2address matrix below describes conditions/minimum pilots. New quantitative baseline/pressure results require three independent acquisitions per condition (54 baseline and6 pressure acquisitions), preserving this supplement's seed and scope. Address traces start as bounded pilots. This retention study is one branch of compute/SRAM/bandwidth research, not the complete H100 assignment.

Added 2026-10-15 after the completed RTX 5080 traffic-reuse study. This is a mandatory supplement to RENTAL_RESUME.md and fixed_bandwidth_collection.md. It specifies new H100 work; none of these H100 measurements is complete merely because this document exists. Keep the earlier Phase 0–2 portfolio and calibration gates. The immediate replication scope is 18 baseline counter captures, two cache-pressure captures and two address captures, not a rerun of every previous RTX study.

## 1. Research question and RTX reference

Can we distinguish the data logically needed by decode from actual off-chip bytes, and determine whether extra on-chip retention could avoid repeated traffic? Test both intra-call duplication and reuse across successive complete decode cycles. Do not assume a larger cache, shared memory or register file improves performance.

The RTX 5080 study qualified all 22 captures. Whole-decode DRAM/main-input ratios were 0.991550–1.003148 per step and approximately three footprints across three steps. Observed layer weight requests were 1.000306/1.000460 times their unique footprint for B1/B8, with KV requests equal to one active footprint. Successive calls reused 100% of weight addresses and 99.9028%/99.9756% of the next KV footprint. A completed 256MiB competing read increased measured layer reads by 1.53884%/0.11742%. All baseline measured local load/store sectors were zero. These are historical RTX observations, not expected H100 outcomes or architectural proof.

The portable numerical reference is [rtx5080_traffic_reuse_reference.json](rtx5080_traffic_reuse_reference.json). It contains the exact models, 18 scope results, pressure and trace summaries, prompt/output fingerprints and provenance. Raw RTX profiler files and traces are not shipped in this repository and are not required to collect H100 data. Do not access sibling directories or assume the RTX workstation's filesystem exists on a rental.

## 2. Prepare the rental and implement the missing adapters

Follow RENTAL_RESUME.md Sections 1–2 first: new allocation inventory, explicit physical UUID, compatible pinned runtime, immutable snapshots, bridge smoke, correctness and counter-permission probe. Query actual H100 SKU, full-GPU versus MIG, SM count, L2, HBM and per-SM shared/register limits. Do not assume the historical 132-SM/50MiB device matches the new rental. Use only the authorized allocation; do not alter host profiling policy or clocks.

**Implementation gap:** the current H100 package has baseline and whole-step trace runners; it does not yet ship the RTX study's layer hooks, replay-safe publications, requested-sector classifier or traffic-comparison analyzer. Implement portable adapters in this repository before launching this matrix. Do not claim the existing runners already produce this evidence, and do not invent CLI flags for them. Use their actual `--help` and source interfaces. Start from scripts/controlled_vllm_experiments.py, tensor_manifest.py, profile_eager_kernels.py, trace_vllm_decode.py and run_trace.py. Preserve historical runners/results or version new entrypoints separately.

The new adapters must:

1. Wrap original decoder forwards solely to collect CPU-side metadata and NVTX boundaries; execute original kernels and all intervening layers. Inspect actual vLLM APIs rather than assume cross-version compatibility. Avoid device-to-host metadata copies in the captured range; use verified existing CPU sequence/page-table mirrors or record unavailable fields.
2. Preserve every application-replay process in its own immutable output directory and publish a pass index. Mark it running before each pass; any failed/incomplete pass invalidates overall completion. Never let a later successful pass hide a failed one.
3. Serialize GPU producers with an owned lock plus idle check and a process monitor throughout execution (RTX used 10 seconds idle and 50ms sampling). Verify UUID and account for owned descendants. A startup snapshot alone is insufficient. Do not terminate unrelated processes; defer and record contamination.
4. Save command/environment, source/config/runtime/model hashes, logs, raw reports, pass metadata, exact outputs, range/call ledgers and monitor records. Retries use fresh job IDs and directories; later failure cannot be replaced by an older success when selecting evidence.
5. Validate CPU accounting and parsers before GPU use: tied/overlapping views, activation value versions, current KV, sector units, wrong-layer/range rejection, failed replay invalidation, malformed/truncated trace rejection, preload controls, exact output mismatch and pressure placement. Use meaningful failure cases.

Keep a fresh `results/<rental>/traffic_reuse/` root and matching trace/profile roots. Extend the machine-readable case manifest with planned/running/qualified/failed/unsupported/permission-denied dispositions. Preparation and CPU tests never count as H100 measurement completion.

## 3. Reproduce these exact six cases first

Use the model revisions in the portable reference; do not confuse Qwen2.5-3B-Instruct with Qwen3. Add a separate versioned replication config for that model; the original portfolio did not include it. Retain BF16 weights and KV, TP1, eager in-process vLLM, ordinary caching, prefix caching off, synchronous scheduling, greedy temperature 0, repetition penalty 1, ignore EOS, no detokenization, and 64 output tokens. Prefer vLLM 0.22.1/compatible recorded runtime; a runtime change creates a separately labeled comparison.

| Model | B | Nominal prompt C | Selected scopes |
|---|---:|---:|---|
| Qwen2.5-1.5B-Instruct | 1 | 1024 | layer 0, layer 27, whole decode |
| Qwen2.5-1.5B-Instruct | 1 | 4096 | layer 0, layer 27, whole decode |
| Qwen2.5-1.5B-Instruct | 8 | 4096 | layer 0, layer 27, whole decode |
| Qwen2.5-3B-Instruct | 1 | 4096 | layer 0, layer 35, whole decode |
| Qwen2.5-3B-Instruct | 8 | 4096 | layer 0, layer 35, whole decode |
| Mistral-7B-Instruct-v0.2 | 1 | 4096 | layer 0, layer 31, whole decode |

This replication uses **seed 20261014**, an explicit exception to the older portfolio's seed 20261013. Do not change old configs or relabel their results. Generate synthetic token IDs on CPU exactly as:

```python
prompts = torch.randint(0, model_config['vocab_size'], (B, C),
    generator=torch.Generator().manual_seed(20261014)).tolist()
```

Save complete prompt IDs and their SHA256 over row-major little-endian signed int64 values; compare with the reference fingerprints. Save complete outputs using the same fingerprint convention. Exact output equality is mandatory among H100 warmups, control, profile, pressure and trace executions at matched conditions. Cross-GPU output differences must be reported/investigated, but never repaired by weakening within-H100 correctness.

RTX settings: max_model_len 8192, max_num_seqs 8, max_num_batched_tokens 32768, gpu_memory_utilization 0.70, block size 16, chunked prefill on, compilation disabled for eager, generation_config='vllm'. Use a bounded KV pool per case: `num_gpu_blocks_override = B * ceil((C+64)/16) + B` (unless a larger explicitly declared override is required). Record actual allocation/layout. Match control and pressure pools; do not allow the new buffer to silently change automatic KV capacity.

Run at least three complete warmup trials. The prefill engine step produces the first output token outside pure decode. Decode tick j is the j-th engine step after that barrier: verify query length 1 and actual attention length C+j for each request. Profile ticks **4, 5, 6**, after ticks 1–3; trace ticks **4, 5**. Check the runtime ledger rather than trusting an ordinal if the runner uses another convention. Keep each 64-token trial's complete outputs and prefill-barrier proof.

Hopper may choose FA3, persistent/TMA kernels or a different split policy. Record actual backend/version, source/binary, attention tile/split/launch details and resources. Do not force RTX FA2 behavior and call it the native H100 baseline. If supported, a separately named FA2 control can help distinguish hardware and backend differences; it does not replace the native baseline.

## 4. Collect 18 real counter captures

First run one bridge layer-0 pilot, qualify it, then the remaining 17 captures. Selected layers are measured during complete-model execution; never repeatedly call an isolated layer. Preserve three layer ranges ordered at ticks 4/5/6, and a separate whole-decode aggregate covering all three ticks including intervening layer work, logits and sampling. Exclude prefill and startup. Validate exported NVTX identities exactly; layer_0 cannot match layer_01. Validate every replay's logical geometry and exact outputs.

Use bounded NVTX filtering, application-range replay where supported, **cache-control none**, **clock-control none**, and a short explicit metric list. Never use all-kernel `--set full`. Discover supported names/units on this H100. RTX metric names below are candidate names, not guaranteed H100 interfaces:

```text
dram__bytes_op_read.sum
dram__bytes_op_write.sum
lts__t_sectors_op_read.sum
lts__t_sectors_op_read_lookup_miss.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_access.sum
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
smsp__inst_executed.sum
gpu__time_duration.sum
```

Save raw metric catalog and NCU command/version/report plus CPU-import CSV in base units. Preserve DRAM reads/writes, L2 requests/read misses, global/async loads and local loads/stores separately. Multiply sector counts by the verified hardware sector size (RTX used 32B), not by tensor dtype width. On Hopper, legacy LSU/LDGSTS counters may omit TMA paths: collect supported TMA/interface counters as needed, document coverage and use null for unavailable fields. Do not treat absent counters as zero traffic.

Profiler durations are instrumented quantities, not normal serving latency or independent replicates. Collect fresh uninstrumented controls with ≥3 warmups and ≥3 measured trials separately if making a timing claim. If NCU permissions fail, save the exact denial and continue independent work; physical traffic comparison remains unavailable, not analytically filled in.

## 5. Calculate U, measure D, and distinguish R

For a layer at tick j, the main useful external input footprint is:

```text
U_layer,j = W_layer + 2 * B * (C+j-1) * Hkv * d * KV_element_bytes
            + distinct layer-entry input values
A_step = sum(D_j over ticks 4/5/6) / sum(U_j over ticks 4/5/6)
A_interval = sum(D_j) / U_interval_distinct
```

D is measured off-chip read bytes. R is observed implementation request traffic from address instrumentation; neither allocation spans nor L2 sector counts are exact logical payload R. Read/write budgets remain separate.

Compute active dense BF16 weights from immutable safetensors headers and reconcile runtime views/dtypes/strides/padding. Include QKV/O/MLP/norm/bias and full LM-head projection. Deduplicate tied weights and overlapping views; untied input embeddings contribute selected lookup rows, not the whole vocabulary table. Record reconciliation differences, not an assumed header/runtime identity.

At tick j, attention reads C+j KV positions including newly produced current K/V. Only C+j-1 positions are historical external inputs for that call. For the three-step interval, initial external KV has C+3 positions; subsequent new KV values are generated inside the interval. Do not count new values as both compulsory external reads and internal outputs.

Layer interval-distinct U counts weights once, KV present at interval entry once, and each tick's incoming hidden/residual/position value versions separately (union aliases within a tick). Whole-decode U counts active model parameters once and initial KV once; internal activations are not external inputs. Track selected untied embedding rows with actual input token IDs. Use actual query/KV lengths and CPU page tables to distinguish active KV from allocated unused blocks. Sliding windows, prefixes or exotic layouts require verified bounds; never guess.

Main U excludes small RoPE/scheduler tables, materialized intermediates, sector overfetch and spill traffic; D includes physical traffic from these where present. Warm hits can lower D. Therefore U is not a strict cold DRAM lower bound: D/U>1 does not prove eviction, and D/U<=1 does not exclude rereads offset by other hits. Approximately three interval footprints can reflect recurrence across tokens without intra-tick amplification.

## 6. Collect two controlled-pressure captures

For Qwen 1.5B B1/C1024 and B8/C4096, layer 0 only, allocate one random float32 CUDA buffer of **256MiB**, seed **20261014**, before warmups and reuse it. Immediately before each selected layer call at ticks 4/5/6, execute an eager reduction reading the full buffer and synchronize. Complete this outside the selected layer NVTX range, then execute the original layer. The rest of the model still runs.

Profile only layer_0, not the full-decode parent range containing the pressure operation. Do not include the competing read's own 256MiB traffic in the measured layer. Publish buffer shape/dtype/stride/logical/span bytes, seed, per-call pressure-completed-before-layer ledger and immutable per-replay metadata/hashes. Match exact full outputs, runtime/model/prompt/backend/settings and KV allocation against the corresponding baseline. Restore wrappers/settings on all failure paths.

Report absolute and percentage change in DRAM reads, writes, L2 requests/misses and local traffic. Call this cache competition, not a guaranteed full flush, enlarged SRAM, isolated cache-miss mechanism or measured larger-cache benefit. The original RTX study has one acquisition per condition; repeat independent control/pressure acquisitions on H100 if interpreting small differences, preserving the core 22 captures as a separate base matrix. Reserve a buffer in an additional matched control if needed to separate memory-allocation effects.

## 7. Collect two real-layer address traces

After a native sm_90 dynamic-input micro trace and tool/space pilot, trace Qwen 1.5B layer 0 at ticks 4/5 for B1/C1024, then B8/C4096. Use fresh same-shape eager references. Original forwards and intervening layers must execute; only selected calls are instrumented. These address captures complement, and do not replace, the separately required whole-model phase-3 traces.

RTX used the existing Accel-Sim NVBit tool with ACTIVE_FROM_START=1 and exported enable_nvbit_instrumentation/disable_nvbit_instrumentation controls. ACTIVE_FROM_START=0 was reset per kernel in that tool version. NVBit removed LD_PRELOAD after startup: prove the exact whitelisted binary is already loaded with RTLD_NOLOAD plus required exported controls, without runtime-loading fallback. Verify behavior on the pinned H100 tool; do not assume a different tool revision behaves identically. Disable instrumentation before Torch/CUDA initialization; synchronize before enabling and before disabling.

Record each call's exact byte offsets in closed statistics files before enable and after completion; statistics include untraced kernels, so preserve launch-ID gaps and map trace filenames/header IDs from those exact intervals. Never partition kernels by equal counts. Save final statistics sizes/hashes and every raw kernel hash. Use SPINLOCK_HANDLING_MODE=0 for a raw parsed-record count equality check; different compression/replay modes need a separately validated parser.

The parser must support the actual native trace format and fail on truncation/unknown memory encoding. Verify parsed records equal reported emitted records per kernel. **This establishes emitted-record integrity, not complete executed-memory coverage.** RTX's default ENABLE_WATCHDOG=1 could skip instruction regions before emission. Preserve all instrumentation flags; audit watchdog, predication, TMA/async, WGMMA descriptors and mbarrier handling on Hopper. Mark unsupported coverage explicitly. Do not silently disable watchdog protection or drop instructions to obtain a favorable total.

Classify observed 32B sectors using actual weight views/aliases, KV allocation plus request-to-page mapping, metadata, identifiable activation/scratch ranges and explicit unmapped ranges. Derive active KV sectors from verified strides/pages/lengths. Separate current-token KV stores from historical values. Count per-call requested sectors, unique sectors, repetition histograms, sectors requested ≥2 times and cross-call overlap for each class. Do not classify generic/unmapped addresses or shared sides of asynchronous copies as global/off-chip traffic without proof. Activation addresses can recur with different values; static weights can recur unchanged.

Validate each trace against its matching counter case: same layer, model, B/C, prompts, full generated token arrays, hardware/software/backend/engine, and exact ticks 4/5. Acquisitions are separate; explicitly set physical-address DRAM miss identity to unmeasured. Raw receive order is not a globally ordered cache-access timeline. A partial selected-layer trace cannot supply a full-model LRU miss curve because the intervening stream was executed but not traced.

Budget from the pilot: the two RTX raw captures together were about 9.4GiB, with the entire evidence set about 13.7GB before compression. Hopper kernels can differ substantially. Reserve grouping/archive/transfer headroom using measured H100 expansion. Keep unsupported native instructions and semantic warnings visible; simulator replay/calibration is a separate gate from request-sector analysis.

## 8. Expand only after the comparable matrix

Then extend the original portfolio to larger models and longer contexts within fit/budget, preserving BF16 and the single authorized GPU. Add early/middle/late layers if the first/last samples differ. Separate native H100 backend effects from a matched supported FA2 control. Include at least one larger-batch compute-active case. Larger models do not automatically imply a greater fractional benefit from a fixed SRAM budget.

Potential retention analysis may use `D_new ≈ D_base - S_effectively_retained` and memory-service ratio `D_base/D_new` at unchanged achieved BW. S_effectively_retained is avoided physical traffic, not SRAM capacity. Perfect pinning is an analytical scenario; LRU cyclic eviction may save little. End-to-end Amdahl estimates require a separable measured time component; overlapping compute/memory requires a suitable overlap model. No GPU-byte measurement establishes SM-to-SRAM area ratio, energy benefit or feasible capacity. Do not remove compute and assume unchanged memory delivery.

## 9. Required deliverables and completion gates

Inside the fresh traffic_reuse root, save:

- A 22-capture manifest with all attempts, raw paths/hashes and final qualification reasons; separately index any added statistical repetitions and model extensions.
- Immutable per-replay profile, runtime tensor/page metadata, exact prompt/output arrays and source/config hashes; selected call and pressure ledgers; sampled isolation monitors.
- `logical_accounting.json`, `counter_comparison.json`, `pressure_comparison.json`, and two `trace_*_qualified.json` files. Each scalar needs scope, unit, measured/analytical classification and source. Missing values are null with reasons.
- Raw NCU reports/import CSVs/catalog/commands, raw traces/statistics/boundary proofs and all coverage warnings. Preserve failures and incomplete captures.
- `rtx5080_vs_h100.md` plus machine-readable comparison and a plot showing both D/summed-per-step U and D/interval-distinct U. Report actual B/C, backend, SKU, warm/cache/replay controls and comparability differences. Reference RTX values without treating them as H100 targets.
- A conclusion explicitly answering: is there intra-call request amplification; does it reach HBM; what recurs across full decode cycles; does cache competition alter HBM reads; are spills significant; and which larger-cache benefits remain conditional? If physical miss addresses remain unavailable, say so.
- CPU/independent review records, source audit, a portable artifact manifest, and local plus durable remote archive checksum/readback verification before rental termination. Never infer completion from exit code alone.

All 18 baseline scopes plus both pressure captures and both address traces must pass their applicable numerical/source/scope/isolation checks for full replication. Permission denial, unsupported trace semantics or incomplete coverage remain explicit limits; do not replace physical counters with analytical bytes. Do not require simulator timing calibration merely to report qualified hardware counter results, but do not use uncalibrated traces to claim architecture performance.

Update STATUS.json, the rental case manifest, calibration table and opportunity report only from actual H100 evidence. This handoff update authorizes preparation/continuation under the existing research contract; it records no new H100 execution and does not erase historical failures or unfinished Phase 0–2 work.
