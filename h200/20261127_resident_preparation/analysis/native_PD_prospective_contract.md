# Prospective native 1P1D pilot: CPU contract only

No native GPU transfer has been executed. The earlier all-layer HF handoff remains valid only in its recorded software/physical-pair scope.

The installed P2pNccl connector sends the producer prompt KV during prefill. Its cached producer-request path assumes a chunked-prefill entry; an ordinary producer continuing64-output generation cannot be substituted unchanged for this first pilot. The planned producer therefore requests one fixed first token p0. Its peer decoder receives prompt plus p0 and generates63more outputs. The native consumer advertises `len(prompt)-1` external tokens, so the appended input makes that external count equal the original context C. The receiver executes p0 at original position C: one query classified as prefill relative to its enlarged prompt, but canonical decode relative to the original request. The remaining62queries are post-prompt. Combined outputs must be exactly64, compared against a full-worker oracle and both strict independent HF references. No claim of zero receiver queries is permitted.

| Quantity | Planned first pilot |
|---|---:|
| Original context |128|
| Producer generated outputs |1|
| Consumer prompt length |129|
| Consumer generated outputs |63|
| Combined generated outputs |64|
| Canonical decode queries |63|
| Consumer first-query prefill classification |1|
| Consumer later post-prompt queries |62|
| Analytical requested BF16 KV tensor bytes at16-token blocks |33,554,432|

Runtime block size, actual cache shape/strides/dtype, all64packed K+V layer tensors, sender completion, receiver load/injection and query counts require actual observations. Padding bytes are distinct from meaningful context bytes and neither is physical HBM/NVLink traffic. CPU contracts reject missing, duplicated, corrupt, wrong-dtype, uncompleted and uninjected layer ledgers.

Use globally unique identical request IDs with exact producer/consumer addresses and the pinned compatibility flag disabling per-engine random suffixes; verify actual internal IDs and all layer keys. Each process must expose only its owned physical UUID. Remain TP1/BF16 on both devices. Native GPU staging defaults to1e9bytes and the host pool to32GiB per worker. A full context4096payload is analytically1,073,741,824bytes, so staging/host spill must be explicitly measured under the chosen concurrency. Do not equate buffer size, KV capacity, logical demand and physical link counters.

Source cleanup limitation: base connector shutdown is a no-op; the separate transport engine close method joins infinite listener/sender loops without a stop flag. The prospective owned-process protocol must drain and record all sends/loads, publish results, run normal engine/process-group cleanup and verify the owned worker exits. It must not call an unbounded close and infer success, or leave a daemon transfer unobserved. No actual deadlock or successful native transfer has been observed by this inspection.

Resume after current batch8/prefill work: implement the versioned native worker/coordinator with bounded startup/IPC/producer deadlines, fresh owned-pair inventory and no-proxy local transport, run a tiny full-worker oracle plus all-layer handoff, inspect first mismatching output and payload evidence if it fails, and only then design loaded3x10paired comparisons. These CPU contracts are not that implementation or GPU qualification.

## Known-data transport component prepared

`scripts/h200_native_transport_probe.py` is a separate, unexecuted component probe. It requires two exact owned role UUIDs/job/hostname, loopback peer addresses and the pinned native-engine SHA. Each of six trials (three warmup, three diagnostic) sends64deterministic full BF16 block tensors totaling32MiB; the receiving GPU must match every source hash and full known-value tensor. These are synthetic fills, not generated model KV, cache injection, an instruction trace, physical-link bytes or calibrated bandwidth. Its explicit small host pool and GPU staging threshold are component-test settings, not the deployment defaults.

A watchdog bounds native blocking calls by480seconds and the earlier descriptor/session deadline; both roles must publish completion before their owned processes exit. It does not invoke the transport engine's unbounded close. Source inspection confirmed that wait_for_sent only waits for an empty queue, which can occur after the last item is popped but before NCCL completes. The prepared probe instead requires every post-send completion ID plus verified receiver acknowledgement; a CPU test rejects an empty queue with the last completion missing. The second CPU test rejects wrong job/host/role/UUID pairs and nonlocal descriptors. Actual GPU transport and full native KV integration remain pending.
