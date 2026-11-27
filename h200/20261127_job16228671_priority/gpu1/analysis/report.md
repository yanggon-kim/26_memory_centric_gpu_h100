# GPU 1: GPU-28fa5fe0-0a71-78dc-ff99-a1fb75e9d459

## Bounded clock-conditioning diagnostic

One independent producer per arm, three warmups and three timed trials; no three-acquisition latency claim. memory_settle_v3 is a fixed one-second private-buffer workload after prefill, outside decode timing; it changes cache history and TTFT. Any future conditioned SM curve needs its own placement, HF and3x10/source/clock/isolation proof and must stay separate from unconditioned results.

Evidence: `../results/clock_conditioning_pilot.json`. All failed windows are retained; no clock or power settings were changed. This diagnostic does not qualify clean bandwidth, eager placement or thresholds.
