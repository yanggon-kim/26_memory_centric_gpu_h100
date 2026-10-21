# Consolidated H200 evidence and gap diagnosis

independent H200 allocation cohorts, BF16/TP1/seed20261013; no cross-allocation acquisition pooling.

## Accepted results

- Engine: bridge B1/C1024 graph/eager wall ratio0.292817; software-engine comparison only.
- SM restriction: qualified eager B1/C1024 comparisons at104/72/40SMs; upper95% regression bounds1.27%/1.40%/1.18%; no iso-area inference.
- Native service:12 source/clock-qualified J5 joins; streaming approximately3.97/3.97/4.01/3.71TB/s at132/104/72/40SMs. No automatic decoder transfer.
- Physical traffic: Qwen14B graph B1/C4096 reads approximately1.00054× summed per-tick demand, approximately3.0016× window-unique demand across3ticks; aggregate ratios do not prove address-level exactly-once access.

| Treatment | Shape | Ratio |95%CI|
|---|---|---:|---|
|J8 normal_eventsempty|B1/C1024|1.020587|[1.000044196676094, 1.0370078678492038]|
|J8 normal_eventscomplete|B1/C1024|1.000338|[0.974662443637027, 1.0378737809062966]|
|J7 pressure_eventsnone|B1/C1024|0.995943|[0.9845382848268049, 1.0054621873050098]|
|J7 pressure_eventsnone|B1/C2048|0.984141|[0.9605971830898663, 1.0126069543113985]|

J7 pressure timing includes intervention cost; pressure is competition, not a cache flush or larger SRAM. J8 whole-model perturbation does not bound local event intrusion. No significant pressure slowdown is established for the qualified B1 controls.

## Board energy

| Family | Qualified | J/timed token (95% CI) |
|---|---|---|
|J10_bridge_graph_b1_c1024_n2048|True|{'value': 0.7757462139716658, 'ci95': [0.7549294088910601, 0.7836499755740107], 'bootstrap_draws': 10000, 'seed': 20261013, 'statistic': 'median of acquisition medians', 'resampling': 'acquisitions then trials within chosen acquisition'}|
|J10_bridge_graph_b8_c4096_n2048|False|None|
|J10_mistral7b_graph_b1_c4096_n1024|False|None|
|J10_mistral7b_graph_b8_c4096_n1024|False|None|
|J10_qwen14b_graph_b1_c4096_n1024|False|None|

Only sensor/output/source/isolation/pooled-clock-qualified families receive a primary interval. Counter and sampled-power agreement is internal consistency, not external absolute calibration. Longer-run energy scope is separate from64-token latency.

## Diagnoses and actions

1. J7: all18 counter producers exited0 with exact replay output, but each exported6 outer ranges while the reader required1. Pressure callbacks caused warmup and other trials to emit the outer range. Keep original rejections; correct the producer and collect one bounded pilot before expansion. B1/C4096 also failed the strict HF/engine output gate and is excluded, with failure retained.
2. J6: native runs reached complete output and NSYS observed both contexts and overlap, but interpreter teardown aborted with destroyed CUDA contexts. Retain both contexts/streams for process lifetime; require clean producer exit, exact outputs and correct context per kernel role before a small fixed-partition treatment family.
3. J8/J10 clock failures remain rejected. No new sweep is justified merely to search for acceptable clocks. Bridge B1 energy is separately qualified after all three long-output acquisitions agree. For excluded B1/C4096, HF and vLLM have the same prompt/model and internally deterministic outputs but diverge at generated token6. Missing logits prevent a definitive numerical-versus-cache diagnosis; exact correctness stays closed.
4. J11 full tracing stays blocked: preserving raw waits is mandatory, and the postprocessor/simulator semantics remain incomplete. No GPU recapture can substitute for that implementation.
5. Private archive upload/readback returns HTTP404 under existing authentication. Shared raw archives and SHA manifests persist; independent private readback is unverified. Authorized dataset access or a corrected private destination is needed; no credentials are published.

## Targeted follow-up

One versioned H200 repair job will run corrected J6 pilots/treatments and J7 physical counters for already output-qualified shapes, keeping three acquisitions/ten ordinary trials and the existing correctness/clock/isolation gates. It will reuse all other job dispositions, retain failures and commit/push at acquisition boundaries. It does not repeat the completed matrix or clock-rejected families.

The architecture motivation remains compute/request-delivery co-design with workload-specific retention. Area, placeable shoreline, package, energy advantage and validated decoder speedup predictions remain unproven.

## Current repair/diagnostic cohort

h200/20261020_job14215309

| Comparison | Shape | Qualified | Ratio95%CI |
|---|---|---|---|
|J8 normal_eventsempty/normal_eventsnone|B1/C1024|False|None|
|J8 normal_eventscomplete/normal_eventsnone|B1/C1024|False|None|
|J8 normal_eventsempty/normal_eventsnone|B8/C4096|False|None|
|J8 normal_eventscomplete/normal_eventsnone|B8/C4096|False|None|
|J8 normal_eventsempty/normal_eventsnone|B1/C4096|False|None|
|J8 normal_eventscomplete/normal_eventsnone|B1/C4096|False|None|
|J8 normal_eventsempty/normal_eventsnone|B8/C4096|False|None|
|J8 normal_eventscomplete/normal_eventsnone|B8/C4096|False|None|
|J8 normal_eventsempty/normal_eventsnone|B1/C4096|False|None|
|J8 normal_eventscomplete/normal_eventsnone|B1/C4096|False|None|
|J6 memory50/idle|B1/C1024|False|None|
|J6 memory100/idle|B1/C1024|False|None|
|J6 arithmetic50/idle|B1/C1024|False|None|
|J6 arithmetic100/idle|B1/C1024|False|None|
|J6 memory50/idle|B8/C4096|False|None|
|J6 memory100/idle|B8/C4096|False|None|
|J6 arithmetic50/idle|B8/C4096|False|None|
|J6 arithmetic100/idle|B8/C4096|False|None|

### Latest diagnostic evidence

The gap list above describes the original cohorts. Current diagnostics below supersede its historical statements about missing logits and pending repair commands; scientific acceptance remains separate.

- h200/20261020_job14215309/results/campaign_queue.json (SHA256 0e28d7e714caa15d6f572cc447bbf9262e79cb7fb9957abb441032418198ae28)
  Original recovery status: needs_diagnosis. The B8 exact reference and CPU-import failure are retained; later CPU consolidation does not relabel the batch successful.

- h200/20261020_job14215309/analysis/correctness_audit_v4/audit.json (SHA256 e1d34752dbcb03d36335e273cc5ca2c80e2ac3c66bdb9af9416b6256448209ef)
  Legacy eager/static used a Boolean 4D mask; pinned Qwen2 eager attention adds the mask to scores, so unused KV slots were not excluded. Its agreement is invalid correctness evidence. Original SDPA/Boolean mask remains valid for that interface; its mismatch remains unresolved.

Physical J7 counter dispositions: {}
repair cohort remains separate from earlier timing/service acquisitions; counter replay can perturb cache history and does not establish a cross-GPU timing join Explicit post-prefill memory conditioning changes cache history and adds untimed TTFT; these are conditioned-decode results, not repaired original unconditioned latency.

Retained allocation cohort: h200/20261019_job14182525
Source: h200/20261019_job14182525/analysis/consolidated/results/qualified_evidence.json (SHA256 55225bd6583b68972903633a883590e900149dbc11a12af44fdc03f2f13afdd5).
Qualified J6 comparisons: 4
Retained physical counter dispositions: {'complete': 15}
These acquisitions remain separate; no cross-allocation timing or clock pooling.

Retained allocation cohort: h200/20261020_job14201590
Source: h200/20261020_job14201590/analysis/consolidated/results/qualified_evidence.json (SHA256 6b7afebc9c8ced54b722ecee2c6b3c12d1ab4a414ea270b602f84ad571b23cac).
Qualified J6 comparisons: 0
Retained physical counter dispositions: {'complete': 3}
These acquisitions remain separate; no cross-allocation timing or clock pooling.
