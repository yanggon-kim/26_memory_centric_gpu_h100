# goal_v1 smoke review — jobs 502423 (gfx942, MI300X) and 502424 (gfx950, MI350X)

Date: 5 November 2026. Release `fab0ea00` (plan: `docs/amd/goal_followup_execution_plan_20261105.md`). One GPU each
(`mi3001x`, `mi3501x`), 54 steps per smoke, one acquisition of every step kind. Smoke numbers are pilots (one block,
one acquisition) and are not G1–G8 results.

Published: `results_published/amd_20261105_mi300x_goal_v1_smoke_job502423/`,
`results_published/amd_20261105_mi350x_goal_v1_smoke_job502424/` (goal tables/summary re-generated with the fixed
analyzer at `2d81405a`; `raw_SHA256SUMS` + `raw_root.txt` identify the untouched raw roots in `$WORK/results_amd`).

## Step outcomes

| | 502423 MI300X | 502424 MI350X |
|---|---|---|
| wall time | 56 min | 42 min |
| native steps rc=0 / check passed | 30/30 | 30/30 |
| vLLM steps rc=0 (incl. route traces, G7 eager counters, energy) | 13/13 | 13/13 |
| `serve` (mixed serving, C=8; plain and energy) | 2/2 | 2/2 |
| shell (G4 audit, ISA, e4) | 3/3 | 3/3 |
| failed acquisitions | 0 | 0 |
| in-job goal report | crashed (analyzer bug 1) | crashed (analyzer bug 1) |

Gates on the smoke roots: `lifecycle_ok`, `outputs_ok` true; `goal_isolation` = leak v4 (tuned decoder) **sensitive and
clean on both** (positive control flags 48/48 probe CUs on MI300X and 8/32 on MI350X, tuned decoder 0 flags, natives
0). `placement_ok` is false only because a smoke probes 4 of the plan's levels; every full plan probes every level and
the competitor pool on all 8 lanes (checked), so the production gate is reachable.

## Bugs found and disposition

1. **analyze_goal `natives()`** spread `analyze.metric()`'s tuple as a mapping → report crash. Analysis only; fixed
   (`2d81405a`). Main-job reports will be regenerated with the fixed analyzer from the raw roots.
2. **G7 per-tick traffic** summed all 6 decode marker ranges (3 warmups + 3 trials) and divided by one trial's ticks (6×
   overstatement). Analysis only; fixed: median over ranges. Corrected FETCH_SIZE per decode tick is 63.6 M KiB on both
   cohorts (range spread < 0.3 %), i.e. 1.001× the logical compulsory 65.04 GB (weights + KV at 4096).
3. **ISA metadata (G6)** produced no kernels: the bundler cannot read the `.so`; the code object is in the
   `.hip_fatbin` section. Fixed (`llvm-objcopy --dump-section` first) and re-run as a CPU job (devel) for both
   architectures: 392 matching kernels each, no errors (`isa_meta_cpu_rerun.json`). Static: wvSplitK variants use
   45–52 VGPRs, no spills, 512-thread workgroups; LDS reservation 160 KiB on gfx950 vs 64 KiB on gfx942.
4. **Counter-audit candidate ordering (G4)** assumed `_sum` names; this rocprofv3 lists per-block names without `_sum`,
   so after FETCH_SIZE/WRITE_SIZE the 16 slots fill alphabetically (including stall/credit counters and a bare `DRAM`
   token). Fixed for future plans. In the `fab0ea00` release the gfx942 picks still include TCC_EA0_RDREQ, _32B,
   _DRAM, WRREQ, WRREQ_64B, WRREQ_DRAM and RD_UNCACHED_32B, and the gfx950 picks include RDREQ_32B/64B/128B and
   RDREQ_DRAM; L2 hit/miss and DRAM-qualified write variants are missing.

## Decision

The owner's additive OC assignment (`11a639d2`) asks to preserve the queued G1–G8 releases rather than rewrite them.
None of the four bugs invalidates an acquisition the production stage depends on (1–2 are analysis-only; 3 is a CPU-only
static step now done separately; 4 narrows but does not void the audit). The held productions were therefore
**released unchanged on `fab0ea00`**: 502427 (MI350X) at 14:54 (started 14:54), 502425/502426 (MI300X/MI325X) at 14:55 CST. The
missing counter coverage is carried into OC0/OC3 under a separate plan and release.

## Early observations (pilot, one acquisition — not results)

- **FETCH_SIZE scale differs by architecture.** gfx942: 1.00× useful bytes (KiB) for 128 MiB×16, 1 GiB×4, 4 GiB×1
  reads — including the MALL-resident 128 MiB load, so it is a pre-MALL request-side counter. gfx950: 0.50× for the
  same loads while WRITE_SIZE is 1.00×; consistent with the derived formula assuming ≤64-B requests while the
  streaming kernel issues 128-B requests (gfx950 exposes TCC_EA0_RDREQ_128B). The decoder's FETCH_SIZE per tick is
  ≈ compulsory on both, so its requests are counted at full size. To be settled by the production audit's per-size
  request counters.
- Override reach (route traces): Qwen32B B1 decode skinny-GEMM share 0.83 (MI300X) / 0.81 (MI350X); Qwen7B B8 decode 0
  (hipBLASLt) — the override is no intervention there, as predicted.
- G1 pilot: MI300X mask/grid 128 R = 0.990; MI350X 192 R = 0.987 (one block, clock-confounded flag as pre-registered).
- Mixed serving C=8 (stock, full device): 182 tok/s MI300X, 314 tok/s MI350X; output agreement
  is recorded per request (a single condition here, so no cross-condition comparison yet).
