# H200 decode traffic attribution

81 sampled kernel launches currently provide 1377 raw counter observations. Reads and writes remain separate.

Kernel roles identify linear, attention-forward, combine, and KV-update work. A linear kernel can read weights, activations, and temporary data; its DRAM count is not automatically weight-only traffic. Allocation manifests identify possible owners without proving accessed or off-chip bytes.

No full-step traffic is assigned to the ordinary 63-step timing trials. Summing sampled layers or kernels would be an estimate requiring coverage and uncertainty; this report does not present such a sum as measurement. Bounded whole-engine three-step range attempts have separate scope and output-prefix proofs.

Executed attention query and KV lengths and allocated request/page maps are saved separately. Raw input-batch counters can lag execution. Persistent KV capacity includes unused pages and is not the active KV working set.

Raw percentages outside their physical range remain preserved but are excluded from derived interpretation. No sensor or counter is rescaled to force agreement.

## Whole-engine range attempts

| Model and shape | Status | HBM read bytes/token | HBM write bytes/token |
|---|---|---:|---:|
| bridge B1/C1024 | failed | NA | NA |
| bridge B8/C4096 | failed | NA | NA |

Ranges cover three engine.step iterations after the one-token prefill barrier, including graph execution and LM head/sampling. Replay mode is recorded per attempt. Application-range replay re-executes the native program with a separate output directory per pass; native output-prefix proofs and raw metrics are required for acceptance. Instrumented range duration is not ordinary decode latency. Failed attempts retain NA with the raw failure. [Nsight Compute documents these replay scopes and limitations](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-range-replay).
