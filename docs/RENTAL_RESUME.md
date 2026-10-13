# Resume H100 research on a new rental

This handoff records the evidence pulled at commit `48672f7` on 2026-10-13. Continue Phase 0–2 and native trace collection; do not begin physical architecture changes. The question remains whether reducing compute and adding memory resources improves decode under a realistic fixed-area and energy budget. Existing results do not establish that benefit.

## Read first

Read `AGENTS.md`, primary specification Sections 0 and 14 followed by the full specification, `STATUS.json`, `docs/opportunity_report.md`, `docs/calibration_table.md` and `results/case_manifest.json`. Use `docs/phase02_playbook.md` for measurement methodology and `docs/trace_collection.md` for capture details. Treat tracked run files as evidence and human summaries as interpretations. Do not overwrite historical artifacts.

## What already exists

| Work | Recorded disposition | How to use it |
|---|---|---|
| Prior allocation | H100 80 GB HBM3, 132 SMs, 50 MiB L2; driver 570.195.03 | Historical environment only; discover the new UUID/SKU and remeasure controls. |
| BF16 HF bridge | Qwen2.5-1.5B revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; compiled B1/C128 and B1/C1024 pass; eager B1/C128 and B8/C4096 pass | Reusable historical software controls; fresh short correctness/timing controls are required for the new runtime/device. No optimized-vLLM result exists. |
| Compiled B8/C4096 | Two attempts failed exact output consistency | Keep excluded from valid performance comparisons; diagnose without weakening the gate. Eager B8 success does not repair compiled correctness. |
| Eager energy | B1/C1024, three roughly 33-second windows, about 2.55 board J/token; counter/power agreement within 0.32% | Internally qualified only; remeasure on the new allocation. No HBM/component energy or absolute calibration. |
| Profiling | Eager PyTorch inventory; Nsight Systems control timeline works; NCU and Systems GPU metrics denied | Hardware HBM/L2/tensor/LSU counters remain missing. Timeline availability alone does not classify the bottleneck. |
| Roofline controls | 718.40 BF16 TFLOP/s and 3032.09 useful-copy GB/s | Prior measured controls; decode intensity is analytical. Recheck new allocation and obtain actual decode counters. |
| Logical reuse | 4 KiB/64 KiB/2 MiB studies; 64 KiB B1/C1024 LRU savings near zero, selective pinning 4.30% at 128 MiB and 17.22% at 512 MiB | Analytical, capacity-only bounds. No measured HBM saving or feasible SRAM/PPA result. |
| Native traces | Patched micro and layer pilot; two-step whole-model HF eager capture: 2,336 grouped sm_90 kernels, 316,425,467 warp records | Correct output IDs and positive one-kernel replay are recorded. Full replay, async semantics and same-program timing remain unvalidated. |
| Eager archive | 5,135,554,560 bytes; local, Hub metadata and full remote readback SHA256 matched | Recover from private storage if needed. It is not included in Git. |
| vLLM runtime | Six failed CUDA 12.8 source builds; next attempt recorded as running | Historical unresolved state. No installed runtime, successful build, smoke or baseline can be inferred. |
| Real text | Four versioned passages and B1/B4 C128 configuration; runner supports `workload_file` | Prepared, unmeasured. Correct the old model path through fresh materialization and run it after synthetic smoke. |

Larger Qwen models, Mistral and conditional MoE experiments have no completed baseline evidence in this snapshot. The original multi-model matrix is still required within fit/time budgets.

## 1. Choose a usable allocation and record new Phase 0

Prefer a single explicitly allocated H100 with a complete CUDA development toolkit, a driver compatible with the pinned CUDA 13 runtime, Python 3.10 development headers, and provider-enabled hardware profiling. Ask the provider about counter access before spending rental time; the previous machine denied it. Do not change drivers, clocks, power limits or host profiling policy yourself. A denied counter route leaves the quantitative bottleneck gate incomplete, while independent baseline/trace tasks may continue.

The previous 25 GiB overlay was inadequate for larger models and simultaneous raw/grouped/archive storage. Plan at least 200 GiB usable persistent workspace as a starting reservation, then calculate the actual requirement from selected weights, environments/builds and pilots; this is not a guarantee for large full-model traces. Keep transfer time before the rental deadline. For traces reserve at least 2× raw budget for grouping, or 3× with a local archive, plus other files and actual expansion headroom. The prior HF archive was about 5.14 GB; optimized kernels can differ greatly. Do not reuse the prior 6.62 GiB free-space snapshot.

Clone this repository, record its HEAD, select the new physical UUID, and use a fresh output root such as `results/rental_YYYYMMDD_instance/` and matching `traces/` root. Inventory OS, GPU SKU/HBM/SM/L2, observed clocks/power, sharing, CPU/RAM, toolkit/runtime versions, free space and rental deadline. Run CPU contracts. Once the environment is installed, use:

```bash
python3 scripts/validate_package.py
python3 scripts/test_cpu_contracts.py
CUDA_HOME=/existing/cuda13 PYTHON_BIN=python3.10 scripts/setup.sh --install
export CUDA_HOME=/existing/cuda13
export PATH="$CUDA_HOME/bin:$PATH"
.venv/bin/python scripts/inventory.py --gpu-uuid GPU-NEW-ALLOCATED-UUID \
  --check-cuda --output results/rental_YYYYMMDD_instance/environment.json
```

Create the fresh result root before inventory. Replace placeholders with real paths/UUIDs. Never reuse the prior UUID or `/root/26_memory_centric_gpu_h100/...` model paths. Resolve/download pinned snapshots and generate new configs with `materialize_config.py`; old committed configs are historical inputs, not portable execution commands. Record actual runtime/config hashes before each comparison.

## 2. Establish the optimized runtime before expanding

Prefer the existing pinned CUDA 13 install route on a compatible provider image. `configs/runtime.lock.txt` targets torch 2.11.0+cu130 and vLLM 0.22.1; the prior CUDA 12.8 HF lock is a separate runtime. Check imports, extension loading and actual GPU execution, not just package installation. Do not install cu130 blindly on the prior R570 image.

If the new image only supports CUDA 12.8, create a separately recorded runtime and validate it. Consult `results/vllm_source_manifest_cu128.json`, `results/runtime_build/attempt_manifest.json`, the six build logs and `patches/vllm_py310_spinloop.patch`. They record old CMake, missing NVRTC link, Python limited-API incompatibility and incomplete math headers. The last proposed fix used absolute header paths in `NVCC_PREPEND_FLAGS` and CPATH; it has no successful result in Git. Obtain complete development dependencies and immutable source identities; do not assume the previous build process or its files exist on this rental. Prefer a provider image that avoids reconstructing a partial toolkit. A new version/ABI route needs its own pinned lock, code/API checks and correctness evidence.

Materialize and run the B1/C128 bridge smoke, with three warmups and three measured trials:

```bash
.venv/bin/python scripts/materialize_config.py --model bridge --stage smoke \
  --download --output results/rental_YYYYMMDD_instance/bridge_smoke.json
.venv/bin/python scripts/controlled_vllm_experiments.py \
  --config results/rental_YYYYMMDD_instance/bridge_smoke.json \
  --gpu-uuid GPU-NEW-ALLOCATED-UUID --mode timing \
  --output results/rental_YYYYMMDD_instance/vllm_smoke
```

The runner validates completion automatically; it has no `--validate` CLI mode. Require the all-request one-token prefill boundary, exactly N−1 measured decode ticks and identical complete output IDs across warmups/trials. Use `scripts/compare_vllm_hf_tokens.py --help` to configure independent HF comparison; investigate mismatches before expansion. Record the actual optimized features and eager/graph configuration. The eager trace path is a distinct same-program calibration control, not automatically the fastest software baseline.

## 3. Complete Phase 1 models, workloads and counters

Follow `fixed_bandwidth_collection.md` as the mandatory measurement/trace supplement. Collect baseline HBM read/write bytes, effective bandwidth, weight/KV/scratch attribution, actual attention tiles/splits, register spills/shared-memory and occupancy limits, full-step timing, observed clock conditions and cross-token reuse. The requested new artifacts and acceptance rules are listed there. These are future collection requirements, not evidence already obtained.

After smoke, materialize baseline configs and execute staged synthetic cases: bridge/Qwen7B/Mistral7B B1/4/8 × C128/1024/4096; Qwen14B B1/4 × those contexts; conditional-fit Qwen32B B1 × C128/1024. Start every model at B1/C128. Pin each new model revision, record actual layout and memory use, and preserve OOM/failure cases. BF16, one allocated GPU, TP1 remain required. Do not automatically quantize or enlarge GPU scope to obtain fit. Complete the dense portfolio before conditional Qwen3-30B-A3B; report routed-expert behavior using representative prompts, and do not send MoE through the dense reuse adapter.

Run `representative_text_v1` at B1/B4 C128 after synthetic bridge correctness. Build a fresh config from the new local snapshot, set `workload_file` to `configs/representative_text_v1.json`, and retain prompt IDs, truncation policy and hashes. The four saved passages support C128; do not silently pad them into representative C1024/C4096 workloads. Longer real-text contexts require separately versioned data/configs.

Measure uninstrumented timing first, then long decode-only energy, then separate profiling. Preserve event elapsed, scheduler wall, aggregate token counts and raw sensors. Reapply all energy gates on this rental; primary J/token stays NA on qualification failure. Rerun the control ceilings using `scripts/measure_roofline_ceilings.py --help`. Obtain Nsight Systems decode timelines and select observed kernels for bounded Nsight Compute traffic/L2/tensor/LSU/issue metrics. Query available metrics on this machine and preserve permission diagnostics. Do not substitute analytical read bytes or device-copy bandwidth for actual decode HBM counters.

Investigate compiled B8/C4096 separately if that control remains in scope: reproduce with the exact immutable model, prompts, cache boundaries and preserved complete token IDs; isolate eager versus compiled behavior. A fresh runtime pass establishes a new configuration, not retroactive validity of failed trials. Record the larger-batch compute-active counterexample before evaluating reduced arithmetic capacity.

Acceptance: schema-complete valid/failed cases across the selected portfolio, representative text results, a quantitative bottleneck classification supported by actual counters, and explicit remaining limitations. A partial matrix remains partial.

## 4. Recover the existing archive and validate native trace semantics

Read `results/archive_transfer.json`. The private Hugging Face dataset is `yang-gon/h100-memory-centric-accelsim-traces`, immutable revision `55688d40408d0684f1f9d1f458cd070ada4bc588`, object `archives/bridge_full_cu128.tar`. Use user-provided authenticated access; do not place tokens in Git or logs. Require exactly 5,135,554,560 bytes and SHA256 `8c57824a7b034caf6651cbd81b156dc6169da6862a0288dce0ee12991374f9a5` before extracting into an owned directory. Inspect archive paths before extraction. If credentials/artifact access are unavailable, mark recovery blocked and continue independent measurements; do not claim the raw trace is present in this clone.

Some `case_manifest.json` paths, including native micro/layer pilot files and profiler reports, were not committed. Missing files are unavailable evidence, not proof of hash corruption. Check tracked source hashes that are locally present; recover archive-contained files where available. Git contains whole-model manifests/metadata and selected result files, not the 5.14 GB archive or the external tools. Initial `DELIVERY_MANIFEST.json` describes package preparation, not a complete historical trace inventory.

Build pinned Accel-Sim/NVBit/postprocessing tools with `scripts/setup_accelsim.sh` and an explicit complete toolkit in a child shell. Verify `configs/simulator_pin.json` and the reviewed missing-spinlock-entry patch; reject unrelated source changes. Run a fresh dynamic-input micro capture and positive native sm_90 replay to validate this rental/toolchain. Keep clean native execution shells; simulator environment/preload must not contaminate real-GPU measurements.

Recovered full HF traces already grouped and passed a one-kernel parser smoke, but contain 1,568 recorded `mbarrier` TRYWAIT-drop warnings. Audit the affected kernels, preserved registers/descriptors/active masks, and postprocessor/simulator treatment before claiming async correctness. Expand replay beyond the first kernel to selected attention/GEMM/async paths, then a bounded full sequence if runtime permits; preserve unsupported opcodes, warnings and timeouts. Capture completion and positive cycles do not establish functional semantics.

Derive a separately versioned actual-SKU configuration and perform measured-versus-simulated comparison for the same eager program, model, shape and decode interval. The stock 132-SM config is only parser evidence. Address the instrumented path, gaps and instruction denominators; NVBit warp records differ from simulator active-thread instructions. If semantics require simulator changes, specify and validate a separate corrective patch before architectural experiments; never silently drop barriers or relabel architectures. Accel-Sim H100 power remains uncalibrated.

## 5. Collect the missing optimized traces and extend Phase 2

Preserve complete inter-layer/weight traffic between successive decode steps for the fixed-bandwidth reuse analysis. Collect actual request-to-KV-page/slot and scratch mappings where accessible, and mark unmapped ranges explicitly. Follow the supplement’s mandatory metadata/export and physical-counter distinction.

Only after valid vLLM smoke/baselines, run the implemented `scripts/run_trace.py --engine vllm` route, beginning with a bounded B1/C128 pilot. It uses eager in-process execution and full-step instrumentation including LM head/sampling; compare against uninstrumented same-shape eager vLLM references. Preserve a separate graph-optimized baseline. Use fresh budgets and `--help` for exact capture/export options.

After the pilot, capture 2–3 full-model bridge decode steps to preserve the entire inter-step weight stream, then selected early/middle/late larger-model layers and bounded full-model cases where measured budgets permit. Keep B8/long-context coverage conditional on fit. Partial-layer repetition cannot establish whole-model cache benefit. Save weight/KV storage ranges and aliases, parse actual addresses to validate attribution, and extend request/page/temporary mappings where needed. Record observed TMA/WGMMA/mbarrier coverage; presence alone is not semantic validation.

Every new capture needs correct reference IDs, nonzero warp records, nonempty kernel lists, all grouped files, actual sm_90 replay with positive kernel/cycle/instruction statistics, source/runtime/model/config hashes and verified durable transfer. Any dropped operations keep semantic calibration incomplete. Allow more than the previous 300-second grouping cap when a pilot justifies it; record a new bound rather than overwriting the recovered timeout. Hash/read back the transferred archive before removing owned local raw files.

Extend logical reuse analyses for supported dense models with measured L2 provenance, LRU/pin/prefix controls and block-granularity/recurrence sensitivities. Relate logical savings to actual trace attribution/counters; retain analytical labels and account for existing cache/software features. MoE requires a separately validated routed-access adapter. SRAM/NoC area, leakage, port/AGU throughput and memory energy are still needed before fixed-area or lower-energy claims.

## 6. Close out before the rental ends

Update `STATUS.json`, `results/case_manifest.json`, schema-complete measured rows, experiment log, calibration table and opportunity report with completed, failed, partial and unavailable gates. Use the consolidation/manifest scripts only after reviewing their input scope; do not convert absence into completion. Retain old results and label new rental/runtime/model/workload identities separately.

Export relevant sanitized logs/metadata, source patches/locks, profiler artifacts and traces to durable storage with local and independent remote checksum verification. Raw profiler databases can embed credentials or environment data; the previous Systems report was withheld from Git after push protection detected sensitive content. Keep secrets out of artifacts/publication, inspect files before adding ignored outputs explicitly, and preserve private storage where appropriate. Record exact artifact locations and access requirements without credentials.

Report what the new evidence supports, what falsifies the hypothesis, and what remains unknown. Do not declare Phase 0–2 or calibration complete if baseline, counter, semantic or matrix gates are missing. Proceed toward architectural design only when the opportunity report supports it and calibration/PPA prerequisites are satisfied; this handoff authorizes software research and trace preparation, not physical architecture implementation.
