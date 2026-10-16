# H200 research assumptions

The user authorizes this H200 allocation and fresh Phase 0–1 measurements. Execution exposes only GPU-a613c806-89b1-05d1-f71c-3d52ab46d246. Startup process checks and a cooperating lock are repeated; they do not exclude unrelated future activity.

Model weights and KV remain BF16, TP1, with immutable revisions, seed 20261013, greedy EOS-ignored generation, and at least three complete warmups and measured trials. All requests must finish the one-token prefill boundary before pure decode. Exact token discrepancies remain failures to investigate.

Graph-optimized vLLM, eager vLLM, HF StaticCache, compiled controls, and profiled execution are separate programs. Uninstrumented wall time and GPU-event elapsed use explicit token denominators. Profiled time cannot replace ordinary latency.

Whole-board energy retains raw counter and sampled-power estimates. Qualification requires the declared minimum window, observed counter updates, and agreement. SRAM/HBM component energy, external absolute accuracy, and architectural area remain uncalibrated.

This session uses the current allocation only. No additional Slurm jobs, system changes, quantization, tensor parallelism, full-model instruction traces, or physical architecture experiments are included. Unfinished cases retain explicit status and resume instructions.
