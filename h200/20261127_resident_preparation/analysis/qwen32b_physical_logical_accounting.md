# Saved Qwen32B traffic versus analytical demand

Analytical useful inputs/FLOPs are separate from measured DRAM/L2. Aggregate near-one reads/logical demand does not prove address-level exactly-once reads or absence of reuse. Clock-rejected clean timing is not joined to these counters; no architectural benefit or clean bandwidth inferred.

GPU `GPU-4ec297c0-8010-9dee-aae9-54a7ad6eb4ad`; model revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`.

| Acquisition | DRAM read / sum of per-tick logical inputs | DRAM read / window-unique inputs | Useful FLOP / DRAM R+W byte |
|---|---:|---:|---:|
| 0 | 1.000547 | 3.001629 | 1.064672 |
| 1 | 1.000545 | 3.001622 | 1.064685 |
| 2 | 1.000545 | 3.001622 | 1.064685 |

The analytical decoder weight input is 63,970,617,344 bytes per tick, including norms/biases and the untied output head. The untied input embedding contributes only the actual selected row. Historical KV is approximately1.075GB per tick at these lengths. Geometry comes from the pinned checkpoint:64layers,40query heads,8KV heads,128head dimension; sliding windows are disabled.

Three-tick logical input sum: 195,136,253,952 bytes. Window-unique logical inputs: 65,045,700,608 bytes. Shared weights are counted once in the latter, explaining why a repeated weight stream can yield a ratio near3without address-level rereading beyond the normal token recurrence.

Limitations: useful FLOPs omit non-main operations and actual issued/padded work. Logical inputs are not a compulsory-DRAM oracle. L2 requested bytes are not a hit rate. The signed residual after subtracting three weight streams is not measured KV traffic. No confidence interval, clean GB/s or latency threshold is derived from these clock-unqualified joins. Kernel attribution, restricted-side counters and grid retuning remain pending.

All exact model/header, physical report, replay output and selected-interval checksums are retained in `../results/qwen32b_physical_logical_accounting.json`. This analysis adds no GPU measurement and does not alter the archived original run.
