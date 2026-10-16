# Decode-only Hopper trace collection and export

**Current H200 direction:** follow [h200_shoreline_jobs.md](h200_shoreline_jobs.md), especially J11, and [common Hopper methods](shoreline_hopper_measurements.md). Native trace preparation is bounded by correctness, semantic, allocation/storage and closeout gates. Independent SM/service calibration must accompany relevant traces. Requested addresses are not HBM bytes; the stock H100 parser smoke is not H200 timing/power calibration. The old H100 procedures below do not authorize an immediate broad trace campaign.

For continuation, read `RENTAL_RESUME.md` first. The two-step whole-model HF eager capture and verified private archive transfer are complete as recorded in `results/archive_transfer.json`; full semantic/calibrated replay and optimized-vLLM collection are incomplete. Recover existing evidence before considering recapture.

The installed source inspected for this scaffold is `util/tracer_nvbit/others/torch_hook/pytorch_profiler.py` at the pinned framework SHA. It supplies `TorchModelHookWrapper(model)`, `hook_nvbit_to_layer(wrapper, module_name)`, and exported tracer callbacks `enable_nvbit_instrumentation`, `disable_nvbit_instrumentation`, `set_nvbit_instrumentation_tag`. Hooks silently ignore a missing tracer; our runner checks that the callback library actually loaded. The optional Python `nvtx` dependency must be present.

For fixed-bandwidth analysis, also follow `fixed_bandwidth_collection.md`: preserve full inter-step working sets, collect page/slot and scratch attribution, and distinguish requested trace bytes from measured HBM transactions.

## Prepare the external pinned source

Clone the release into an external path, verify framework/backend pins, build CPU simulator/postprocessor and NVBit 1.8 under an explicitly selected matching toolkit. The local H100 execution applies only the one-line `patches/accelsim_nvbit_missing_spinlock_entry.patch`: the original optional spinlock lookup dereferenced a missing map entry and reproducibly segfaulted even on the dynamic-input micro kernel. `configs/simulator_pin.json` pins original, patched and patch hashes; setup and capture reject other tracked source differences. No driver is downloaded or built. Example source location in the original workspace was `03_simulation/accel_sim2/vendor/accel-sim-framework`; portability uses an explicit `--accel-root`.

Read upstream licenses/citation instructions. Preserve source diffs, compiler versions, tracer shared-library hash and build logs. Do not run the upstream destructive NVBit installer over an existing installation without reviewing its behavior. Validate a dynamic-input small kernel before an LLM; a constant-folded arithmetic test is inadequate.

## Capture a bounded actual decode sequence

The trace wrapper precreates the complete output parent and nested `raw/traces` directory, sets USER_DEFINED_FOLDERS=1, and prepends `CUDA_HOME/bin` so an older PATH `cuobjdump` cannot silently reject a newer cubin. It starts NVBIT_INSTRUMENTATION_ENABLED=0, uses ACTIVE_FROM_START=1 and runtime module hooks. It deliberately does not use ACTIVE_FROM_START=0/cudaProfilerStart because the pinned upstream has the reproduced empty-capture bug. Dynamic kernel filtering is removed for the runtime-hook mode.

```bash
.venv/bin/python scripts/run_trace.py \
  --accel-root /external/pinned/accel-sim-framework \
  --config results/bridge_smoke.json --gpu-uuid GPU-ALLOCATED-UUID \
  --output traces/bridge_full_decode --steps 2 --layers full \
  --budget-gib 40 --timeout-s 1800 --replay-smoke \
  --archive-path /persistent-export/bridge_full_decode.tar
```

No trace hook exists during three warmups and the final prefill. After all requests' prefill and first selected token finish, the eager HF runner attaches hooks and executes exactly the requested 2–3 whole-model forward decode steps. Metadata logs prompt IDs/hash, immutable model revision, request batch/context, output IDs/hash, selected modules, steps and generated-token denominators. Full coverage means root model forward including output projection; non-model argmax/copies are omitted and disclosed. It is BF16 eager SDPA/dynamic KV, a different kernel/program path from optimized vLLM/CUDA graphs. Calibrate replay against the same eager route; use graph baselines as separate controls.

Whole-model bridge sequences preserve distinct layer addresses and the entire weight working set between steps. A single-layer repetition can falsely manufacture cache reuse and is prohibited as a whole-model cache argument. For 7B/14B/32B first collect disjoint early/middle/late modules, then bounded full-model sequences if storage/runtime budgets permit. Partial traces must retain `coverage=selected modules` and cannot claim full-model LLC residency or end-to-end cache benefit. Increasing budget/steps is an explicit decision, not automatic. Reserve at least raw+grouped headroom (2× raw budget), or raw+grouped+archive (3×), and monitor actual expansion/free space. Byte/time limits terminate the owned process group and mark partial_failed; partial traces are retained but not validated baselines.

The trace state uses ALLOW_REG_VAL_TRACING=1, SKIP_TMA_MEM=0, SPINLOCK_HANDLING_MODE=2 and ENABLE_WATCHDOG=1, preserving descriptors/register operands and marked barrier polling behavior. These flags do not prove a selected kernel uses TMA/WGMMA or prove parser accuracy. Inventory observed opcodes/active masks and test async/barrier semantics. Inspect upstream source for features actually captured. Native H100 support remains a test, not a header relabel.

The runnable `--engine vllm` route is mandatory for optimized-kernel calibration after the HF bridge smoke. It uses the inspected public `llm.collective_rpc` worker model API, InprocClient, uni executor, async_scheduling=False, enforce_eager=True and compilation_config={'mode':0}. Three uninstrumented same-shape complete references precede capture; generated IDs must match exactly. The all-request first-token barrier must pass before instrumentation. This route is implemented but native H100 execution remains pending.

For full vLLM coverage, runtime callbacks bracket each complete pure-decode `engine.step`, synchronize, then disable. No nested module hooks run in that mode: Qwen's model.forward returns hidden states and its separate LM-head/logit processor can bypass module.forward. Full step coverage therefore includes output projection, sampling and staging kernels, preserving the whole weight set. Partial mode attaches disjoint named module hooks through the actual torch_hook API and declares partial coverage. Root+child overlap is rejected. Graph replay does not re-run Python hooks; compare this eager vLLM trace against separately measured eager vLLM and retain graph-optimized baselines as distinct controls.

Both trace routes export parameter storage bases/end ranges, dtype, shape/strides/offsets and shared/tied aliases. HF exports dynamic KV before/after each step because allocations move; vLLM exports actual kv_caches/attention-layer cache ranges plus observed page/group specs, without guessing axes. Activation/temporary/workspace addresses and request-to-page slot tables remain explicitly unmapped. Allocated bytes are not accessed traffic. The initial H100 smoke must parse actual trace addresses and prove attribution to saved ranges before weight-pinning or KV-locality claims; extend request/page mapping when needed.

Use a baseline-stage config to enable B8 after B1 pilots, then `--engine vllm --batch 8 --context 4096` within the explicit time/disk budget. Do not assume HF SDPA contains the same FA/TMA/WGMMA kernels. H100 trace completion requires this actual vLLM route and observed relevant Hopper opcode coverage/replay; HF-only collection is insufficient. Do not reuse the upstream example unchanged, because it attaches hooks before generation and lacks a decode-only barrier.

## Verify before rental termination

1. Native completion is insufficient. Positive `total_reported_insts`, nonempty context kernel lists under the actual nested output directory and trace files are mandatory. The upstream postprocessor may also exit successfully on an empty input.
2. Run the pinned `post-traces-processing <directory> -j 2`; it produces grouped `.tracez` and context `kernelslist*.g`. `export_trace.py` checks every listed grouped file exists and is nonempty, and hashes the raw/grouped outputs and metadata.
3. Replay an actual sm_90 trace kernel using both H100 configs. The bounded rental smoke selects one real captured kernel; positive cycles, active-thread instructions and kernel statistics are required, with no unsupported-architecture/opcode messages. Stock132SM/SXM-like H100 config makes this a parser/completion integrity smoke, not exact-SKU or full-model calibration. Derive actual allocation configs before calibrated timing comparisons. Extend to observed Hopper async kernels and matched eager measurements before architectural conclusions.
4. Retain opcode coverage, register/descriptor/barrier data, parser logs, NVBit/toolkit/source/runtime hashes and all failures. Do not compare NVBit warp records directly with simulator active-thread instructions; use real active masks.
5. Create the lossless tar archive outside the trace tree, preserving raw and grouped files and manifests. Check the SHA256 sidecar, transfer it to durable storage, and recompute the checksum there. A local archive alone is not successful transfer. Allow disk for raw + grouped + archive and time for hashing/transfer. Git intentionally excludes all raw traces, profiler databases and archives.
6. Keep rental running until actual replay, transfer integrity and manifest checks finish or explicitly report the failed/pending gate. A successful V100 fixture or an unsupported sm_120 exit0 cannot substitute for this H100 check.
