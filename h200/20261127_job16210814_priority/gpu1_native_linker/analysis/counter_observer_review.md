# Full-device physical traffic and scheduler-observer review

These are saved measurements from this UUID/source cohort, audited on CPU after preemption. No new GPU execution occurred.

GPU `GPU-4ec297c0-8010-9dee-aae9-54a7ad6eb4ad`; source freeze `bf8ee96b404f3d101113532ebc7c5e5959ec055225f7d1ee4a7a1d1b54ac5c34`.

## Three DRAM/L2 acquisitions

| Acquisition | DRAM read B/token | DRAM write B/token | L2 requested B/token | Instrumented 3-step ms |
|---|---:|---:|---:|---:|
| 0 | 65081029546.7 | 50400256.0 | 100288585760.0 | 80.8071 |
| 1 | 65080872192.0 | 49780736.0 | 100285238581.3 | 82.5940 |
| 2 | 65080873728.0 | 49780224.0 | 100284761408.0 | 79.8162 |

Three physical DRAM/L2 totals for full optimized decode, three ticks each. L2 metric is total requested bytes. No clean bandwidth, address-level exactly-once claim, architectural SRAM saving or restricted-arm explanation is established.

Limitation: the clean timing reference failed its clock envelope. Physical totals cannot be joined to qualified clean latency or GB/s. The native-linker repair made collection functional; restricted-side counters, kernel-grid retuning and complete attribution remain unfinished. Resume with those separately qualified comparisons; retain all captures and original failed linker attempt.

## Three scheduler-observer pairs

All six independent producers completed exact outputs with three warmups and ten timed trials each. Each observed producer retained13scheduler files,832step records,53,248prompt input queries and819post-prompt input queries, with no mixed steps or preemptions. Checksums and source identities are in `../results/P3a_observer_review.json`.

B1/C4096 decode-only observer validation with one full prefill step and63decode steps per request. Not loaded serving, tuned policies, mixed arrivals, held-out SLOs or demonstrated negligible observer intrusion. Clock gates failed; all raw trials retained.

Limitation: pooled SM clock ratio is 1.5714285714285714; the5%clock gate fails. No observer latency ratio is qualified. Resume with explicit prefill-only/mixed/load adapters and matched intrusion review before serving conclusions.
