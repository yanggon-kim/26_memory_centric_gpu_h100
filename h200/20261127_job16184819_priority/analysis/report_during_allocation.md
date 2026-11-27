# Two-H200 priority campaign

No measurements yet. Primary: Qwen2.5-32B BF16/TP1 B1/C4096 SM curve. Secondary UUID: independent native controls.

- Limitation P0_new_pair_KV: Fresh paired HF handoff deferred until after first complete SM sweep; prior adapter validation applies only to its recorded physical pair. Evidence: results/P0_prior_KV_software_validation.json. Tried: Validated unchanged adapter/model source hashes; fresh allocated-pair peer-copy smoke remains separate. Resume: At a safe completed-curve boundary, run the existing real-KV pilot on the new allocated pair; never relabel prior UUIDs.

- Limitation model_startup_io_wait: Both first model producers spend minutes before decode; owned process snapshot shows cl_sync_io_wait and package-bytecode descriptors. No decoder latency is inferred from startup wall time.. Evidence: environment/early_model_process_snapshot.txt; environment/early_model_io_detail.txt; gpu0/logs/graph_b1_c4096_stock_f100_a0_proof.log; gpu1/logs/graph_b1_c4096_stock_f100_a0_traffic_reference.log. Tried: Retained live process/I/O evidence and versioned1200s producer budgets; pinned sources/versions unchanged. Resume: Require complete run publications, exact outputs and measured placement; use actual observed costs and allocation cutoff for curve admission.

- Prepared bounded import-order correction: results/capture_binding_import_preflight_preparation.json. Nine CPU contracts passed; no GPU validation, active frozen worker unchanged. Any use requires a fresh identified cohort and fresh placement proofs.

- GPU1 completed a three-acquisition full-GPU physical DRAM capture comparison: [report](../gpu1/analysis/P2a_triplicate_counter_report.md). Replay/source/reader gates pass; clean-clock/L2/kernel/restricted-arm joins remain incomplete. No qualified clean bandwidth or latency inference.

- GPU1 native controls complete: all10 condition curves qualified across132/112/96/80/56/40SM,3acquisitionsx10trials each. [Native report](native_controls_report.md). Useful payload is not physical HBM service or a decoder threshold.

- Limitation allocation_preemption: Parent job16184819 PREEMPTED at2026-11-27T01:21:12 Eastern after3619s, including staging and idle:2.0105555556 allocatedGPU-hours. No GPU allocation remains. GPU0 traffic acquisition1 and GPU1 corrected-cohort eager full-green proof were interrupted.. Evidence: environment/final_slurm_accounting.psv; logs/worker0.log; logs/gpu1_import_preflight_supervisor.log. Tried: Saved three GPU1 physical captures, ten qualified native curves, stock decoder/HF proofs, failed optimized context evidence and all partial cases; no continuation on expired allocation. Resume: CPU reconcile every cohort and archive; prepare a fresh two-H200 job prioritizing the corrected separate eager route and eligible full-GPU long-context work; fresh UUIDs/source gates required.
