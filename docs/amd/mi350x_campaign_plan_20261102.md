# MI350X campaign — what we repeat, why, and the jobs (2 November 2026)

**Access:** the owner was granted access to `mi3501x` (1× MI350X slice) and `mi3508x` (8× MI350X) on
2026-11-02. A toy job (496003) confirmed the hardware and software:
- AMD Instinct MI350X, gfx950, 256 CUs, 288 GiB;
- a gfx950 HIP binary built on the login node ran with no errors;
- about 6.45 TB/s on a vector add.

**Cohort rule:** MI350X is a **third, separate cohort**. Its results are never pooled with MI300X or MI325X;
comparisons across cohorts are made only at matched per-XCD CU fractions and labelled as such.

## 1. Why MI350X matters to the project

The question in plan 2.0 is how compute and memory bandwidth must be balanced. The project asks this
"shoreline" question: if compute moves toward the memory (SRAM, or HBM/PHY bandwidth), what CU count does
memory-bound decode actually need?

The two cohorts so far share the same CU structure but differ in HBM. MI350X changes both, and in opposite
directions (vendor specifications):

| | MI300X | MI325X | MI350X |
|---|---:|---:|---:|
| CUs (XCDs × CUs per XCD) | 304 (8 × 38) | 304 (8 × 38) | 256 (8 × 32) |
| HBM peak | 5.3 TB/s | 6.0 TB/s | 8.0 TB/s |
| HBM peak per CU | 17 GB/s | 20 GB/s | **31 GB/s** |
| LDS per CU | 64 KiB | 64 KiB | **160 KiB** |
| HBM capacity | 192 GB | 256 GB | 288 GB |

This makes MI350X a natural test of the two findings so far:

1. **Memory-level parallelism (MLP).** MI325X job 492957 showed that the CU-restriction penalty in decode comes
   from fewer HBM reads in flight per CU, not from more bytes moved. With about 1.6–1.8× more bandwidth per CU,
   each CU must keep more requests in flight to saturate HBM. If the MLP explanation is right, a CU restriction
   should cost *more* on MI350X at the same fraction of CUs. If it costs the same or less, the explanation is
   incomplete.
2. **Kernel geometry (`wvSplitK`).** The non-monotone curve on MI325X came from one kernel launching 304
   workgroups that each use all 64 KiB of a CU's LDS, so one fits per CU.
   - On gfx950, a 64 KiB workgroup leaves room for two per CU in 160 KiB of LDS. vLLM's gfx950 path may also use
     a different kernel or sizing.
   - If the curve's shape changes on MI350X, that is direct evidence it is a software-geometry artifact, not a
     property of the hardware's compute/bandwidth balance.

A third bandwidth-per-CU point also tests the E7 service-response model on new hardware. E7 failed its blind
check on MI325X.

## 2. Experiments repeated on MI350X (same definitions as the follow-up assignment)

| ID | What | Why it is needed | MI350X specifics |
|---|---|---|---|
| E0 | Capability audit (rocprofv3 counter list, amd-smi, native counter validation) | gfx950 counter names and rocprofv3 support may differ; every counter claim needs this | first run on gfx950 |
| E1 | Placement proof per GPU and level; leak tests v1.1 and v2 (per-CU probe) | the CU mask must give exactly the requested CUs, balanced per XCD; E3 needs proof the decoder does not leak onto competitor CUs | probe reserves all 160 KiB of LDS (bench v4.1) so one probe workgroup still fills a CU |
| E2 | Decode latency R(n) = T(n)/T(full), Qwen32B (8 blocks) and Qwen14B (3 blocks), B1/C4096 graph mode | the core shoreline measurement: penalty vs CU fraction | nominal levels 256/192/168/128/80 plus fine levels 232/208/184/152 (same per-XCD fractions as MI300X) |
| E3 | Fixed partition: decoder on n CUs, owned competitor on CUs 192–255 (memory 50/100%, arithmetic) | bandwidth contention vs compute contention at a fixed CU split | minimal grid at n = 192 and 128; released only if leak test v2 passes (the MI300X rule) |
| E4 | Service curves (streaming read, ILP, grid, stride, gather, pointer chase, reuse, MFMA/FMA) at each level, 3 acquisitions | measures how each memory/compute service scales with CUs; the inputs to E7 | grid/chase sizes scale with 256 |
| E5/E6 | Kernel-trace timelines at every fine level; kernel identification; `wvSplitK` sized for the active CUs; kernel-shape controls; pre-MALL counters (FETCH_SIZE, RDREQ, RDREQ_LEVEL, GRBM) | tests the MLP and kernel-geometry mechanisms directly | gfx950 may dispatch a different skinny-GEMM kernel |
| E7 | Service-response model fit and held-out check (CPU, after the job) | does a bandwidth-scaling model predict R(n) on new hardware? | fitted on this cohort only |
| E8 | Guardrails: B8 decode, prefill, 16k context, Mistral-7B, matched energy at full/boundary/lowest level | a CU reduction must not secretly hurt other workloads or energy | boundary level 168 (21/32 per XCD, ≈ 25/38 on MI300X) |

**Not planned:**
- The full E3 grid: it needs the clock gate, which neither earlier cohort passed.
- Fixed-clock qualification: clock and power changes are not authorized.

All timing stays observational unless the pooled clock gate passes.

## 3. Jobs

**Code (cubw v1.4):** per-cohort hardware table in `plan.py` (`HW`, `configure`).
- The analyzer takes the full CU count from the plan.
- Preflight takes the architecture and CU count as arguments, and the job script and submit wrapper take them
  from the plan.
- Bench v4.1 adds `--lds` to cuprobe; its default is the device's LDS per CU, so the gfx942 behaviour is
  unchanged.
- Regenerated MI300X/MI325X extension plans are identical to the frozen ones, and re-analysis of 492957 gives
  identical tables.

**gfx950 binaries:** `$WORK/bin/svc_bench_gfx950{,_v2,_v3,_v4}`, with SHA-256 in `svc_bench.sha256`.
`_v2` is built from `scripts/amd/svc_bench.hip` at HEAD; preflight uses only its copy and FMA modes.

1. **Smoke, one GPU (`mi3501x`, 0.175×):** `--kind pilot --smoke`, 37 steps, about 1 h estimated, about
   1.7 units. It checks on real gfx950:
   - placement and the 160 KiB probe;
   - the counter list;
   - vLLM decode at 256/192/128, plus the sized kernel;
   - a kernel trace, shapes, counters and a few E4 rows.

   Its findings can still change the 8-GPU job before that job starts.
2. **Campaign, 8 GPUs (`mi3508x`, 1.4×):** `--kind auto --campaign`, 727 steps.
   - Stages: pilot, then the gate (`analyze.py gate`), then production.
   - The longest lane is estimated at 2.95 h. The limit is 3:45:00, at up to about 42 units (8 × 1.4 per
     node-hour; earlier jobs used about half their estimate).

**Budget at submission:** 318.6 of 450 units used. MI300X job 492956 (up to about 22 units) is still queued.
