# AMD MI300X experiment log (allocation 468885, node k002-003, 2026-10-18)

All times CDT. Result root: `results/amd_20261018_mi300x/` (large raw files stay in `$WORK`/results, see `docs/amd/artifacts.md`).

## Allocation and environment (J0)

- SLURM job 468885, partition `mi3008x`, node `k002-003`, 8× AMD Instinct MI300X (gfx942, 304 CUs, 192 GB HBM3, 750 W cap), started 12:11:55, limit 12 h.
- Host: ROCm 7.2.0 default, amdgpu 6.16.13; HF route uses the `pytorch/2.10.0` module (torch 2.10.0+rocm7.1, Python 3.12, transformers 4.57.6, numpy 2.4.3 vs lock 2.2.6).
- vLLM route: AMD container `rocm/vllm:rocm7.14.1_cdna_ubuntu24.04_py3.14_pytorch_2.11_vllm_0.23.0` (SHA256 017554e0…632e), vLLM 0.23.1.dev1+g9ddef7117, torch 2.11.0+rocm7.14.1, transformers 5.16.1, run via apptainer.
- GPU identity: amdsmi `hip_uuid` differs between amdsmi versions (host 26.1 returns the ASIC serial; the container returns torch's HIP UUID), and torch's ROCm `uuid` is not the amdsmi UUID. The package now uses `GPU-<ASIC serial>` and verifies the HIP device by PCI bus.
- GPUs used: 0 `GPU-e5d1f2010c8b902a` (M1), 1 `GPU-ff01d97deccdb374` (HF smoke, profiling, counters), 2 `GPU-a72ad0fc3c4efc83` (M4/M7 service), 3 `GPU-870c8c5c21427604` (M6), 4 `GPU-3bb38eeb949c256a` (M8 energy), 5 `GPU-05c7e838addcf74b` (profiling, counters), 6 `GPU-77e3d22110fc21c1` (rocprof-compute), 7 `GPU-3c9417ac76059843` (M5 pressure, then M6 for larger models).
- Concurrency disclosure: lanes ran simultaneously on different GPUs of the same node (shared host CPU, PCIe root complexes and node power). No foreign users were on the node. ROCr opens a small context (~27 KB VRAM, no GFX time) on every GPU for any HIP process, so the foreign-use check rejects only ≥64 MiB VRAM or GFX activity and records the rest.

## Tool findings

| Finding | Evidence | Consequence |
|---|---|---|
| Counter access is permitted | `rocprofv3 --pmc` works on compute nodes | Hardware traffic gates are reachable on AMD (they were blocked on H100) |
| Counters validated on a known-traffic copy | 1 GiB copy: FETCH_SIZE 1,048,588 KB, WRITE_SIZE 1,048,576 KB; TCC_EA0_RDREQ_DRAM×128 B and WRREQ_DRAM×64 B ≈ 1.07 GB each | FETCH/WRITE_SIZE are trusted for L2→memory bytes |
| No Infinity Cache (MALL) or data-fabric counters exposed | `rocprofv3-avail` list | Measured bytes are L2→memory-side traffic, an upper bound on HBM bytes; MALL effects come from timing controls |
| FETCH_SIZE + WRITE_SIZE in one pass exceeds the counter budget | "Request exceeds the capabilities of the hardware", then rocprofv3 hangs in its signal handler | One derived counter per pass; hard-kill timeouts and a watchdog |
| Per-dispatch counters crash/hang on real LLM workloads | vLLM eager: "AQL packet is malformed" / illegal address right after runtime Triton loads; HF: crash at lazily loaded `attn_fwd`; simple HIP microbenchmarks always fine | Exclude `^(_\|triton)` kernels from collection, up to 4 attempts per group, all attempts kept; rocprof-compute filtered with `-k Cijk` and eager attention |
| `HSA_CU_MASK` device index is the physical (pre-HIP-masking) index | Mask `0:` had no effect on GPU 2; `2:` worked | Lanes use the physical index |
| `HSA_CU_MASK` ignores comma lists; ranges work | every-8th list gave full FMA rate | Only contiguous ranges used |
| CU indices interleave across XCDs | per-stream exact bitmask: 38 contiguous CUs → 15.9 TFLOP/s; every-8th (all on one XCD) → full rate, i.e. an XCD left with an empty mask ignores it | Contiguous k (multiple of 8) masks trim every XCD equally; masks leaving an XCD empty are invalid |
| ROCm 7.1 `amd-smi` refuses `/dev/dri` access; mixed 7.1/7.2 amdsmi libraries break the CLI | rocprof-compute "No GPU detected" | rocprof-compute uses a clean-environment ROCm 7.2 `amd-smi` shim |
| Shared Triton cache across concurrent container processes | "hip_utils.so could not be loaded", unspecified launch failure (qwen14b cu232 a0) | From ~13:35, per-process Triton/vLLM/inductor caches; earlier completed runs passed exact-output validation |
| MI300X clocks are power-managed | GFX clock rises when CUs are masked (≈1650 → ≈1950 MHz for 7B B1); memory clock ramps 900→1300 MHz at trial start | Contract clock envelope (≤5%) often fails; reported as declared, with a separately labelled interior-sample diagnostic |

## Timeline

- 12:12 job start; J0 inventory, identity, counter, CU-mask, container checks.
- 12:39 vLLM graph smoke (bridge B1/C128): exact outputs, 2.18 ms/step.
- 12:40 M1 lane (GPU 0); 12:43 HF StaticCache gate passed (GPU 1).
- 12:46 rocprofv3 trace of vLLM eager works, with per-step and per-layer markers.
- 12:47 service sweep (GPU 2); 12:48 profiling lanes (GPU 1, 5), CU-mask lane (GPU 3), energy lane (GPU 4).
- 12:50 M5 pressure (GPU 7), complete at 13:05 (15 acquisitions).
- 12:53–13:30 counter-collection crash investigation (see tool findings); lanes restarted with exclusion and retries.
- 13:06 M6 extension lane on GPU 7 (Mistral, 14B, 32B).
