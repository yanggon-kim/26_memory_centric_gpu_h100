# Current research assumptions and boundaries

- The named H100 UUID is exclusively allocated by the user. Every GPU run must expose only that physical GPU and repeat a startup contention check.
- BF16 weights and KV, immutable model revision, seed 20261013, greedy EOS-ignored outputs, and the strict all-request first-token barrier remain fixed. Any runtime adaptation is a newly recorded software configuration, not an alteration of the original baseline.
- The 23 GiB free-space limit is hard. Keep model, environment, raw/grouped traces and archive budgets explicit; a partial or empty capture is retained as failure evidence.
- CUDA 13 on the R570 driver is not accepted as an ordinary compatible path. The separate CUDA 12.8 bridge runtime is pinned in `configs/runtime_cu128.lock.txt`; it can provide a measured software bridge if H100 kernels and the exact output contract pass, but cannot certify the pinned CUDA 13/vLLM route.
- Logical full-block reuse is analytical. A real sm_90 NVBit trace, positive replay and matched eager calibration are separate requirements. Stock H100 simulator configuration does not establish exact-SKU timing or power.
- A local archive checksum is not a verified transfer to durable storage. Under workspace-only storage, that closeout gate remains pending.
- The experiment can conclude with a justified stop or pivot if Phase 2 opportunity, runtime correctness, calibration or storage gates fail. No physical architecture claim follows from a software proxy alone.
