# Portable dense logical-reuse model

**New direction:** this logical-reuse abstraction remains one sub-study. Follow [the H200 jobs](h200_shoreline_jobs.md) for independent P/beta, useful versus issued work and c/r/f sensitivity. SRAM bytes do not establish eliminated physical traffic, SM fraction does not establish arithmetic-rate fraction, and nominal stack bandwidth does not establish delivered service.

This is an analytical access-order/block-fill study, not a GPU memory trace or simulator. It reads actual safetensors headers and configuration from the selected immutable local snapshot. The current adapter supports dense `qwen2` and `mistral` naming/layout only, rejects active sliding-window attention and MoE, and requires two-byte checkpoint weights for the BF16-size accounting. Unsupported families need a separately validated adapter; absence of an exception is not sufficient evidence of coverage.

Each decode step reads declared layer weights and output projection, one embedding row per request, appends one K/V row and scans valid unique KV heads. Shared-prefix K/V ranges alias across requests. Actual config determines layers/heads/head_dim. An LM-head alias is introduced only when actual tie_word_embeddings is true and the independent head tensor is absent. Tied storage must not be double-counted, but its two access roles remain distinct.

The baseline L2 is an explicit byte value from the actual selected SKU/query, with provenance. No 64MiB RTX default or ambiguous marketing50MB is inherited. Added storage is diagnostic0/8/32/128/512MiB; it is not a feasible equal-area or latency/power assertion. Fully associative LRU, pinned weights and pinned shared-prefix policies charge full allocated synthetic blocks on first read miss, separate requested payload from read-fill traffic, and use write-through/no-write-allocate. Pinning excludes unchanged baseline L2 capacity. Policy order is declared, not learned as an oracle.

The default run is3 total logical steps:1warmup and2measured steps. Repeat with longer recurrence and 4KiB/64KiB/2MiB blocks. Embedding/prefix tails still pay full allocated-block fills. KV growth, prefix fraction, policy order, GQA sharing, logical write traffic and eviction changes must remain explicit. Actual cache lines, coalescing, rereads, L2 banks/NoC/MLP, occupancy, placement, pin-control/coherence costs and HBM transactions are outside this abstraction.

Example after actual L2 query and immutable bridge materialization:

```bash
.venv/bin/python scripts/materialize_config.py --model bridge --stage baseline \
  --snapshot /cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
  --output results/bridge_baseline.json --reuse-output results/bridge_reuse.json \
  --l2-bytes ACTUAL_QUERIED_BYTE_VALUE --l2-source 'inventory selected_cuda.l2_bytes and source'
.venv/bin/python scripts/reuse_analysis.py --config results/bridge_reuse.json --output-dir results/reuse_64k
.venv/bin/python scripts/reuse_analysis.py --config results/bridge_reuse.json --chunk-bytes 4096 --output-dir results/reuse_4k
.venv/bin/python scripts/reuse_analysis.py --config results/bridge_reuse.json --chunk-bytes 2097152 --output-dir results/reuse_2m
```

Outputs are `per_tensor_manifest.json`, `trace_sample.jsonl` (logical), `traffic_bound_curves.csv`, `reuse_distance.csv` (sampled LRU capacity/hit curve, not exact instruction reuse distance), `remote_energy_break_even.csv` (dimensionless illustrative sensitivity) and `analysis_metadata.json`. All stay analytical. No measured joules or latency are inferred. MoE timing/trace is allowed within the existing allocation; MoE logical analysis remains unsupported until actual routed experts and representative saved prompts are mapped.
