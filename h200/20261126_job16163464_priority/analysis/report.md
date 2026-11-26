# Two-H200 priority campaign

No decoder-curve measurements yet. Primary: Qwen2.5-32B BF16/TP1 B1/C4096 SM curve. Secondary UUID: independent native controls.

- Limitation P0_inventory_startup_IO: Fresh device-query process required several minutes; process snapshot showed cl_sync_io_wait, then query completed and paired pilot launched without repair. Evidence: h200/20261126_job16163464_bootstrap_repair1/environment/coordinator_start_processes.txt; h200/20261126_job16163464_bootstrap_repair1/environment/coordinator_query_io.txt; environment/allocated_devices.json. Tried: Observed the owned process and retained both snapshots; no process termination or system changes. Resume: Budget from actual measured producer pilots; startup waiting is not decoder latency and is not a GPU measurement.

## P0 synthetic peer-copy control

The allocated pair completed3 warmups and10 instrumented128MiB BF16 known-value copies. Correctness passed, peer access was available in both directions, and the sampled ownership monitor recorded no failures. The pilot copies in one direction; bidirectional peer capability is not a bidirectional bandwidth measurement. Nsight Systems produced `profiles/P0_p2p.nsys-rep`; raw results and scope are in `results/P0_p2p.json`. Timing is an instrumented single-acquisition pilot, with no primary transfer-latency, physical-link-byte or serving claim. Real all-layer KV handoff is a separate pending pilot.

## P0 real all-layer BF16 KV handoff

Functional pilot passed at B1/C128 for the pinned32B model: all64 layers/128 KV tensors (32MiB logical KV plus8-byte first token) were copied from GPU0 to GPU1 and reconstructed. Three warmups plus three validation iterations produced exact64-token outputs matching the untransferred source and independent HF generate. The receiver performed63 one-token queries and never prefilled. See `results/P0_real_hf_kv.json`, `results/P0_real_hf_kv_review.json`, and the captured handoff in `profiles/P0_real_hf_kv.nsys-rep`. This validates this HF dynamic-cache handoff only. The vLLM connector, loaded serving, physical link bytes and primary transfer latency remain unvalidated. Raw copy/rebuild/event observations are retained as unqualified pilot diagnostics.
