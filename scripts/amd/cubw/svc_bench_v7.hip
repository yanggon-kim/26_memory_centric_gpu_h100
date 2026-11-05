// Independent service microbenchmarks for gfx942 (MI300X/MI325X): contract M4 and A-D, reused for M7.
// Each run prints one JSON line with event-timed trials and a correctness check.
// v3 (compute/bandwidth follow-up E1/E3/E4) adds, without changing the v2 kernels:
//   read_ilp   K independent 16-B loads in flight per thread per iteration (--ilp 1|2|4|8|16)
//   placement  each workgroup records HW_ID/XCC_ID; reports the distinct physical CUs that ran work
//   idle       competitor mode only: an allocated process that launches nothing (process-overhead control)
//   competitor windows carry realtime and monotonic stamps, completed launches, and started/stopped
//   events; SIGTERM/SIGINT end the run after the in-flight launch so cleanup is observable.
// v4 (follow-up extension, leak test v2) adds, without changing the v3 kernels:
//   cuprobe    competitor mode only: FMA workgroups that each reserve the full 64 KiB LDS of a CU (one resident
//              workgroup per CU, like vLLM's wvSplitK) and count completed workgroups per physical CU (key as
//              in placement). Windows carry the per-CU counts; a foreign workgroup resident on a probe CU makes
//              that CU, not the whole partition, complete fewer probe workgroups.
// v4.1 (MI350X port): cuprobe reserves --lds bytes (default: the device's LDS per CU, 64 KiB on gfx942 as in v4,
//              160 KiB on gfx950) so one probe workgroup still fills a CU; the started event records the size.
// v5 (cubw v1.5, MI350X follow-up), without changing the v4 kernels:
//   cuprobe_t  competitor mode only: duration probe for leak test v3. No LDS; each thread runs a dependent chase
//              through an L2-resident 256 KiB ring plus 4 FMA chains. Thread 0 adds the workgroup's residency time
//              (s_memrealtime, constant clock) and one completion to its physical CU. A foreign workgroup sharing a
//              probe CU (SIMD issue, vector-memory path) lengthens that CU's workgroups even when the dispatcher
//              assigns every CU the same number of workgroups per launch, which made v4 completion counts blind
//              on gfx950 (job 496074: counts per CU within 18 of 506,270 alone and beside an overlapping probe).
//   dispatch   per-workgroup physical CU, start and end time for a wvSplitK-shaped launch: --blocks workgroups of
//              --threads (default 1024 = 16 waves) holding --lds bytes (default: the device LDS per CU), each
//              streaming its own contiguous chunk of a --bytes buffer. Reports how the workgroups land on CUs and
//              shader engines and how many start only after another workgroup on the same CU has finished.
// v6 (cubw v1.6, leak test v4 on gfx950), without changing the v5 kernels:
//   gapprobe   competitor mode only: start-delay probe. Each launch has --blocks workgroups of --threads (default
//              1024 = 16 waves) holding the full LDS of a CU, sized so that one lands on every CU of the probe mask;
//              each workgroup runs --iters FMA iterations and thread 0 records its physical CU, start and end. The host
//              sleeps --gap-us between launches, so a foreign workgroup can take a probe CU while the probe is idle.
//              A full-LDS workgroup cannot co-reside with a probe workgroup, so a foreign one holding a probe CU
//              delays the probe's start there (or pushes it to another CU). Per window and CU: launches in which the
//              CU ran a probe workgroup (cu_blocks), in which its first workgroup started more than --late-ns after the
//              launch's first start (cu_late), and in which it ran two or more (cu_multi). v3 measured how long a
//              probe workgroup takes, and the probe won the CU against a full-LDS intruder (smoke 497081); v6
//              measures when it can start.
// v7 (overlap/service calibration OC-v0, docs/amd/overlap_calibration_followup_20261105.md), without changing the v6
// kernels. Route-matched references for vLLM's bf16 wvSplitK, whose arithmetic is v_mfma_f32_4x4x4_16b_bf16 fed by
// global_load_dwordx4 (weights) and ds_read_b128 (activation in LDS) on gfx942 and gfx950 (disassembly, 2026-11-05):
//   mfma4      register-operand v_mfma_f32_4x4x4_16b_bf16, 8 independent accumulators per wave (arithmetic reference)
//   mfma4_lds  the same, each MFMA's A operand read with ds_read_b128 from --lds bytes of LDS (on-chip delivery + MFMA)
//   split      known-schedule control: even workgroups stream the buffer (read), odd workgroups run mfma4; each
//              workgroup holds --lds bytes so at most one is resident per CU (default: the device LDS per CU)
//   fused      known-schedule control: every wave keeps one 16-B load in flight while issuing --intensity mfma4 that
//              do not depend on it (loads and MFMA overlap inside the wave by construction)
// Useful FLOPs count 2*4*4*4*16 = 2048 per wave-level 4x4x4_16b instruction.
// Measured-real-GPU when run on the declared GPU; physical HBM bytes need rocprofv3 counters separately.
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <chrono>
#include <csignal>
#include <map>
#include <set>
#include <unistd.h>
#include <thread>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#define CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

struct Args {
  std::string mode = "read";
  size_t bytes = 1ull << 30;      // footprint
  int blocks = 0;                 // 0 -> 8 * CUs
  int threads = 256;
  int reps = 20;                  // kernel launches per trial
  int trials = 10;
  int warmups = 3;
  int stride = 1;                 // elements of 16 B for "strided"
  int chains = 1;                 // independent pointer chains (chase)
  long steps = 1 << 20;           // dependent loads per chain (chase)
  int line = 128;                 // chase node spacing in bytes
  int intensity = 0;              // FMAs per 16 B loaded ("mixed")
  long iters = 4096;              // compute loop length (fma, mfma)
  int passes = 1;                 // in-kernel passes over the buffer ("read"), for cache-resident footprints
  double seconds = 0;             // >0: competitor mode, run "read" or "fma" for this long and report windows
  double duty = 1.0;              // competitor busy fraction (host sleeps between launches)
  unsigned long long seed = 20261013ull;
  std::string cu_mask;            // hex mask, LSB = CU0; empty = default stream
  int ilp = 4;                    // independent loads per thread per iteration (read_ilp)
  int expect_cus = 0;             // placement: CUs the active mask should expose (0 = all)
  long spin = 200000;             // placement: cycles each workgroup stays resident
  int lds = 0;                    // cuprobe: dynamic LDS bytes per workgroup (0 = device LDS per CU)
  double gap_us = 200;            // gapprobe: host sleep between launches
  double late_ns = 5000;          // gapprobe: a CU's first start later than this after the launch's first start is late
};

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

__global__ void k_read(const float4* __restrict__ a, size_t n, float* out, int passes) {
  float4 s = {0, 0, 0, 0};
  for (int p = 0; p < passes; ++p)
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  float t = s.x + s.y + s.z + s.w;
  if (t == 1234.5678f) out[0] = t;  // keeps loads live without a reduction
}

__global__ void k_write(float4* __restrict__ a, size_t n, float v) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    a[i] = make_float4(v, v, v, v);
}

__global__ void k_copy(const float4* __restrict__ a, float4* __restrict__ b, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) b[i] = a[i];
}

// Reads n/stride float4 elements spaced stride apart (coalescing degrades as stride grows).
__global__ void k_strided(const float4* __restrict__ a, size_t n, int stride, float* out) {
  float s = 0;
  size_t m = n / stride;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i * stride]; s += v.x + v.w;
  }
  if (s == 1234.5678f) out[0] = s;
}

// Random 16-B gathers through an index array (index bytes are part of the traffic).
__global__ void k_gather(const float4* __restrict__ a, const uint32_t* __restrict__ idx, size_t m, float* out) {
  float s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < m; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[idx[i]]; s += v.x + v.w;
  }
  if (s == 1234.5678f) out[0] = s;
}

// Dependent pointer chase: each chain follows next[] for `steps` loads.
__global__ void k_chase(const uint32_t* __restrict__ next, const uint32_t* __restrict__ starts, int chains,
                        long steps, uint32_t* out) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= chains) return;
  uint32_t p = starts[c];
  for (long i = 0; i < steps; ++i) p = next[p];
  out[c] = p;
}

__global__ void k_fma(float* out, long iters, float x) {
  float a0 = x + threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  const float m = 0.999999f, c = 1e-7f;
  for (long i = 0; i < iters; ++i) {
    a0 = fmaf(a0, m, c); a1 = fmaf(a1, m, c); a2 = fmaf(a2, m, c); a3 = fmaf(a3, m, c);
    a4 = fmaf(a4, m, c); a5 = fmaf(a5, m, c); a6 = fmaf(a6, m, c); a7 = fmaf(a7, m, c);
  }
  float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (s == 1234.5678f) out[0] = s;
}

// K independent loads per thread per iteration (memory-level parallelism sweep).
template <int K>
__global__ void k_read_ilp(const float4* __restrict__ a, size_t n, float* out) {
  float s = 0;
  size_t g = (size_t)gridDim.x * blockDim.x;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += K * g) {
    float4 v[K];
#pragma unroll
    for (int k = 0; k < K; ++k) { size_t j = i + k * g; v[k] = j < n ? a[j] : make_float4(0, 0, 0, 0); }
#pragma unroll
    for (int k = 0; k < K; ++k) s += v[k].x + v[k].w;
  }
  if (s == 1234.5678f) out[0] = s;
}

// Physical placement probe: wave 0 of each workgroup records HW_ID (CU/SH/SE) and XCC_ID, then the
// workgroup spins so concurrently resident groups spread over every CU the queue may use.
__global__ void k_placement(uint32_t* rec, long spin) {
  if (threadIdx.x == 0) {
    rec[2 * blockIdx.x] = __builtin_amdgcn_s_getreg((31 << 11) | 4);       // HW_REG_HW_ID
    rec[2 * blockIdx.x + 1] = __builtin_amdgcn_s_getreg((31 << 11) | 20);  // HW_REG_XCC_ID (gfx940+)
  }
  long t0 = clock64();
  while (clock64() - t0 < spin) {}
}

// Per-CU probe: full-LDS workgroup, 8 independent FMA chains per thread, completion counted per physical CU.
__global__ void k_cuprobe(uint32_t* counts, long iters) {
  extern __shared__ float lds[];
  lds[threadIdx.x] = (float)threadIdx.x;
  __syncthreads();
  float a0 = lds[threadIdx.x], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, a4 = a0 + 4, a5 = a0 + 5, a6 = a0 + 6, a7 = a0 + 7;
  for (long i = 0; i < iters; ++i) {
    a0 = fmaf(a0, 0.999999f, 1e-7f); a1 = fmaf(a1, 0.999999f, 1e-7f); a2 = fmaf(a2, 0.999999f, 1e-7f);
    a3 = fmaf(a3, 0.999999f, 1e-7f); a4 = fmaf(a4, 0.999999f, 1e-7f); a5 = fmaf(a5, 0.999999f, 1e-7f);
    a6 = fmaf(a6, 0.999999f, 1e-7f); a7 = fmaf(a7, 0.999999f, 1e-7f);
  }
  float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  if (s == 1234.5678f) lds[0] = s;
  if (threadIdx.x == 0) {
    uint32_t hw = __builtin_amdgcn_s_getreg((31 << 11) | 4), xcc = __builtin_amdgcn_s_getreg((31 << 11) | 20) & 0xf;
    uint32_t key = xcc << 8 | ((hw >> 13) & 7) << 5 | ((hw >> 12) & 1) << 4 | ((hw >> 8) & 0xf);
    atomicAdd(&counts[key & 0xfff], 1u);
  }
}

__device__ inline uint32_t cu_key() {
  uint32_t hw = __builtin_amdgcn_s_getreg((31 << 11) | 4), xcc = __builtin_amdgcn_s_getreg((31 << 11) | 20) & 0xf;
  return xcc << 8 | ((hw >> 13) & 7) << 5 | ((hw >> 12) & 1) << 4 | ((hw >> 8) & 0xf);
}

// v5 duration probe (leak test v3): acc[2*key] += residency (wall-clock ticks), acc[2*key+1] += 1.
__global__ void k_cuprobe_t(unsigned long long* acc, const uint32_t* __restrict__ ring, uint32_t ring_mask, long iters) {
  unsigned long long t0 = wall_clock64();
  uint32_t q = (blockIdx.x * blockDim.x + threadIdx.x) & ring_mask;
  float a0 = (float)threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  for (long i = 0; i < iters; ++i) {
    q = ring[q];
    a0 = fmaf(a0, 0.999999f, 1e-7f); a1 = fmaf(a1, 0.999999f, 1e-7f);
    a2 = fmaf(a2, 0.999999f, 1e-7f); a3 = fmaf(a3, 0.999999f, 1e-7f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t key = cu_key() & 0xfff;
    atomicAdd(&acc[2 * key], wall_clock64() - t0);
    atomicAdd(&acc[2 * key + 1], 1ull);
  }
  if (q == 0xffffffffu && a0 + a1 + a2 + a3 == 1234.5678f) acc[0] = 0;  // keeps the chain and FMAs live
}

// v5 dispatch probe: rec[3*b] = CU key, rec[3*b+1] = start, rec[3*b+2] = end (wall-clock ticks) of workgroup b.
__global__ void k_dispatch(unsigned long long* rec, const float4* __restrict__ a, size_t per, float* out) {
  unsigned long long t0 = wall_clock64();
  const float4* b = a + blockIdx.x * per;
  float s = 0;
  for (size_t i = threadIdx.x; i < per; i += blockDim.x) { float4 v = b[i]; s += v.x + v.w; }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[3 * blockIdx.x] = cu_key();
    rec[3 * blockIdx.x + 1] = t0;
    rec[3 * blockIdx.x + 2] = wall_clock64();
  }
  if (s == 1234.5678f) out[0] = s;
}

// v6 start-delay probe (leak test v4): rec[3*b] = CU key, rec[3*b+1] = start, rec[3*b+2] = end of workgroup b.
__global__ void k_gapprobe(unsigned long long* rec, long iters) {
  extern __shared__ float lds[];
  unsigned long long t0 = wall_clock64();
  lds[threadIdx.x] = (float)threadIdx.x;
  __syncthreads();
  float a0 = lds[threadIdx.x], a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3;
  for (long i = 0; i < iters; ++i) {
    a0 = fmaf(a0, 0.999999f, 1e-7f); a1 = fmaf(a1, 0.999999f, 1e-7f);
    a2 = fmaf(a2, 0.999999f, 1e-7f); a3 = fmaf(a3, 0.999999f, 1e-7f);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    rec[3 * blockIdx.x] = cu_key();
    rec[3 * blockIdx.x + 1] = t0;
    rec[3 * blockIdx.x + 2] = wall_clock64();
  }
  if (a0 + a1 + a2 + a3 == 1234.5678f) lds[0] = 0;  // keeps the FMAs live
}

typedef __attribute__((__vector_size__(4 * sizeof(short)))) short bf16x4;
typedef __attribute__((__vector_size__(16 * sizeof(float)))) float floatx16;

// Register-resident BF16 MFMA throughput: 32x32x8 per instruction per wave, 4 independent accumulators.
__global__ void k_mfma(float* out, long iters) {
  bf16x4 a = {(short)0x3f80, (short)0x3f80, (short)0x3f80, (short)0x3f80};
  bf16x4 b = {(short)0x3c00, (short)0x3c00, (short)0x3c00, (short)0x3c00};
  floatx16 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
  for (long i = 0; i < iters; ++i) {
    c0 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c0, 0, 0, 0);
    c1 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c1, 0, 0, 0);
    c2 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c2, 0, 0, 0);
    c3 = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a, b, c3, 0, 0, 0);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (s == 1234.5678f) out[0] = s;
}

typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4;
#define MF4(c) c = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a, b, c, 0, 0, 0)

__device__ __forceinline__ void mfma4_body(long iters, float* out) {
  bf16x4 a = {(short)0x3f80, (short)0x3f80, (short)0x3f80, (short)0x3f80};
  bf16x4 b = {(short)0x3c00, (short)0x3c00, (short)0x3c00, (short)0x3c00};
  floatx4 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0}, c4 = {0}, c5 = {0}, c6 = {0}, c7 = {0};
  for (long i = 0; i < iters; ++i) { MF4(c0); MF4(c1); MF4(c2); MF4(c3); MF4(c4); MF4(c5); MF4(c6); MF4(c7); }
  float s = c0[0] + c1[1] + c2[2] + c3[3] + c4[0] + c5[1] + c6[2] + c7[3];
  if (s == 1234.5678f) out[0] = s;
}

__global__ void k_mfma4(float* out, long iters) { mfma4_body(iters, out); }

__global__ void k_mfma4_lds(float* out, long iters, int lds_words) {
  extern __shared__ uint4 sm[];
  for (int i = threadIdx.x; i < lds_words; i += blockDim.x) sm[i] = make_uint4(0x3f803f80u, 0x3f803f80u, i, 0);
  __syncthreads();
  bf16x4 b = {(short)0x3c00, (short)0x3c00, (short)0x3c00, (short)0x3c00};
  floatx4 c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
  int j = threadIdx.x % lds_words;
  for (long i = 0; i < iters; ++i) {
    uint4 v = sm[j]; j += 64; if (j >= lds_words) j -= lds_words;
    bf16x4 a0 = {(short)v.x, (short)(v.x >> 16), (short)v.y, (short)(v.y >> 16)};
    bf16x4 a1 = {(short)v.z, (short)(v.z >> 16), (short)v.w, (short)(v.w >> 16)};
    c0 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a0, b, c0, 0, 0, 0);
    c1 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a1, b, c1, 0, 0, 0);
    c2 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a0, a1, c2, 0, 0, 0);
    c3 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a1, a0, c3, 0, 0, 0);
  }
  float s = c0[0] + c1[1] + c2[2] + c3[3];
  if (s == 1234.5678f) out[0] = s;
}

__global__ void k_split(const float4* __restrict__ a, size_t n, long iters, float* out) {
  extern __shared__ uint4 sm[];
  if (threadIdx.x == 0) sm[0] = make_uint4(blockIdx.x, 0, 0, 0);  // keeps the LDS reservation live
  if (blockIdx.x % 2) { mfma4_body(iters, out); return; }
  float4 s = {0, 0, 0, 0};
  size_t half = (gridDim.x + 1) / 2, w = blockIdx.x / 2;
  for (size_t i = w * blockDim.x + threadIdx.x; i < n; i += half * blockDim.x) {
    float4 v = a[i]; s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  float t = s.x + s.y + s.z + s.w + (float)sm[0].y;
  if (t == 1234.5678f) out[0] = t;
}

__global__ void k_fused(const float4* __restrict__ a, size_t n, int intensity, float* out) {
  bf16x4 x = {(short)0x3f80, (short)0x3f80, (short)0x3f80, (short)0x3f80};
  bf16x4 y = {(short)0x3c00, (short)0x3c00, (short)0x3c00, (short)0x3c00};
  floatx4 c0 = {0}, c1 = {0};
  float4 s = {0, 0, 0, 0};
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i];  // in flight while the independent MFMA below issue
    for (int k = 0; k < intensity; k += 2) {
      c0 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(x, y, c0, 0, 0, 0);
      c1 = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(y, x, c1, 0, 0, 0);
    }
    s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  float t = s.x + s.y + s.z + s.w + c0[0] + c1[1];
  if (t == 1234.5678f) out[0] = t;
}

// Concurrent memory + arithmetic in one kernel: `intensity` dependent FMAs per 16 B streamed.
__global__ void k_mixed(const float4* __restrict__ a, size_t n, int intensity, float* out) {
  float s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = a[i];
    float t = v.x;
    for (int k = 0; k < intensity; ++k) t = fmaf(t, 0.999999f, v.y);
    s += t + v.z + v.w;
  }
  if (s == 1234.5678f) out[0] = s;
}

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i += 2) {
    std::string k = argv[i];
    if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(1); }
    std::string v = argv[i + 1];
    if (k == "--mode") a.mode = v;
    else if (k == "--bytes") a.bytes = std::stoull(v);
    else if (k == "--blocks") a.blocks = std::stoi(v);
    else if (k == "--threads") a.threads = std::stoi(v);
    else if (k == "--reps") a.reps = std::stoi(v);
    else if (k == "--trials") a.trials = std::stoi(v);
    else if (k == "--warmups") a.warmups = std::stoi(v);
    else if (k == "--stride") a.stride = std::stoi(v);
    else if (k == "--chains") a.chains = std::stoi(v);
    else if (k == "--steps") a.steps = std::stol(v);
    else if (k == "--line") a.line = std::stoi(v);
    else if (k == "--intensity") a.intensity = std::stoi(v);
    else if (k == "--iters") a.iters = std::stol(v);
    else if (k == "--passes") a.passes = std::stoi(v);
    else if (k == "--seconds") a.seconds = std::stod(v);
    else if (k == "--gap-us") a.gap_us = std::stod(v);
    else if (k == "--late-ns") a.late_ns = std::stod(v);
    else if (k == "--duty") a.duty = std::stod(v);
    else if (k == "--seed") a.seed = std::stoull(v);
    else if (k == "--cu-mask") a.cu_mask = v;
    else if (k == "--ilp") a.ilp = std::stoi(v);
    else if (k == "--expect-cus") a.expect_cus = std::stoi(v);
    else if (k == "--spin") a.spin = std::stol(v);
    else if (k == "--lds") a.lds = std::stoi(v);
    else { fprintf(stderr, "unknown option %s\n", k.c_str()); exit(1); }
  }
  return a;
}

static std::vector<uint32_t> parse_mask(const std::string& hex, int cus) {
  std::vector<uint32_t> words((cus + 31) / 32, 0);
  std::string h = hex.rfind("0x", 0) == 0 ? hex.substr(2) : hex;
  int bit = 0;
  for (int i = (int)h.size() - 1; i >= 0 && bit < cus; --i) {
    int d = std::stoi(std::string(1, h[i]), nullptr, 16);
    for (int b = 0; b < 4 && bit < cus; ++b, ++bit) if (d >> b & 1) words[bit / 32] |= 1u << (bit % 32);
  }
  return words;
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  int cus = p.multiProcessorCount;
  if (a.blocks == 0) a.blocks = 8 * cus;
  if (a.lds == 0) a.lds = (int)p.maxSharedMemoryPerMultiProcessor;
  if (a.mode == "cuprobe" && a.lds > 65536)
    CHECK(hipFuncSetAttribute((const void*)k_cuprobe, hipFuncAttributeMaxDynamicSharedMemorySize, a.lds));
  if (a.mode == "dispatch" && a.lds > 65536)
    CHECK(hipFuncSetAttribute((const void*)k_dispatch, hipFuncAttributeMaxDynamicSharedMemorySize, a.lds));
  if (a.mode == "gapprobe" && a.lds > 65536)
    CHECK(hipFuncSetAttribute((const void*)k_gapprobe, hipFuncAttributeMaxDynamicSharedMemorySize, a.lds));
  if (a.mode == "split" && a.lds > 65536)
    CHECK(hipFuncSetAttribute((const void*)k_split, hipFuncAttributeMaxDynamicSharedMemorySize, a.lds));
  if (a.mode == "mfma4_lds" && a.lds > 65536)
    CHECK(hipFuncSetAttribute((const void*)k_mfma4_lds, hipFuncAttributeMaxDynamicSharedMemorySize, a.lds));
  int wall_khz = 0;
  CHECK(hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0));
  const uint32_t ring_mask = (1u << 16) - 1;  // cuprobe_t: 65,536-entry (256 KiB) ring, L2-resident
  hipStream_t s = nullptr;
  int mask_cus = cus;
  if (!a.cu_mask.empty()) {
    auto words = parse_mask(a.cu_mask, cus);
    mask_cus = 0;
    for (auto w : words) mask_cus += __builtin_popcount(w);
    CHECK(hipExtStreamCreateWithCUMask(&s, words.size(), words.data()));
  } else {
    CHECK(hipStreamCreate(&s));
  }
  float* out; CHECK(hipMalloc(&out, 4096 * sizeof(float)));
  double useful_bytes = 0, useful_flops = 0, per_access = 0;
  bool ok = true;
  std::string check = "none";
  void *buf = nullptr, *buf2 = nullptr, *idx = nullptr, *starts = nullptr;
  size_t n = a.bytes / 16;
  std::vector<uint32_t> host_next;
  auto launch = [&]() {
    if (a.mode == "read") hipLaunchKernelGGL(k_read, a.blocks, a.threads, 0, s, (const float4*)buf, n, out, a.passes);
    else if (a.mode == "write") hipLaunchKernelGGL(k_write, a.blocks, a.threads, 0, s, (float4*)buf, n, 1.0f);
    else if (a.mode == "copy") hipLaunchKernelGGL(k_copy, a.blocks, a.threads, 0, s, (const float4*)buf, (float4*)buf2, n);
    else if (a.mode == "strided") hipLaunchKernelGGL(k_strided, a.blocks, a.threads, 0, s, (const float4*)buf, n, a.stride, out);
    else if (a.mode == "gather") hipLaunchKernelGGL(k_gather, a.blocks, a.threads, 0, s, (const float4*)buf, (const uint32_t*)idx, n, out);
    else if (a.mode == "chase") hipLaunchKernelGGL(k_chase, (a.chains + 63) / 64, 64, 0, s, (const uint32_t*)buf, (const uint32_t*)starts, a.chains, a.steps, (uint32_t*)buf2);
    else if (a.mode == "fma") hipLaunchKernelGGL(k_fma, a.blocks, a.threads, 0, s, out, a.iters, 1.0f);
    else if (a.mode == "mfma") hipLaunchKernelGGL(k_mfma, a.blocks, a.threads, 0, s, out, a.iters);
    else if (a.mode == "mfma4") hipLaunchKernelGGL(k_mfma4, a.blocks, a.threads, 0, s, out, a.iters);
    else if (a.mode == "mfma4_lds") hipLaunchKernelGGL(k_mfma4_lds, a.blocks, a.threads, a.lds, s, out, a.iters, a.lds / 16);
    else if (a.mode == "split") hipLaunchKernelGGL(k_split, a.blocks, a.threads, a.lds, s, (const float4*)buf, n, a.iters, out);
    else if (a.mode == "fused") hipLaunchKernelGGL(k_fused, a.blocks, a.threads, 0, s, (const float4*)buf, n, a.intensity, out);
    else if (a.mode == "mixed") hipLaunchKernelGGL(k_mixed, a.blocks, a.threads, 0, s, (const float4*)buf, n, a.intensity, out);
    else if (a.mode == "read_ilp") {
      switch (a.ilp) {
        case 1: hipLaunchKernelGGL(k_read_ilp<1>, a.blocks, a.threads, 0, s, (const float4*)buf, n, out); break;
        case 2: hipLaunchKernelGGL(k_read_ilp<2>, a.blocks, a.threads, 0, s, (const float4*)buf, n, out); break;
        case 4: hipLaunchKernelGGL(k_read_ilp<4>, a.blocks, a.threads, 0, s, (const float4*)buf, n, out); break;
        case 8: hipLaunchKernelGGL(k_read_ilp<8>, a.blocks, a.threads, 0, s, (const float4*)buf, n, out); break;
        case 16: hipLaunchKernelGGL(k_read_ilp<16>, a.blocks, a.threads, 0, s, (const float4*)buf, n, out); break;
        default: fprintf(stderr, "unsupported --ilp %d\n", a.ilp); exit(1);
      }
    }
    else if (a.mode == "placement") hipLaunchKernelGGL(k_placement, a.blocks, a.threads, 0, s, (uint32_t*)buf, a.spin);
    else if (a.mode == "cuprobe") hipLaunchKernelGGL(k_cuprobe, a.blocks, a.threads, a.lds, s, (uint32_t*)buf, a.iters);
    else if (a.mode == "cuprobe_t") hipLaunchKernelGGL(k_cuprobe_t, a.blocks, a.threads, 0, s, (unsigned long long*)buf,
                                                       (const uint32_t*)buf2, ring_mask, a.iters);
    else if (a.mode == "dispatch") hipLaunchKernelGGL(k_dispatch, a.blocks, a.threads, a.lds, s, (unsigned long long*)buf2,
                                                      (const float4*)buf, n / a.blocks, out);
    else if (a.mode == "gapprobe") hipLaunchKernelGGL(k_gapprobe, a.blocks, a.threads, a.lds, s, (unsigned long long*)buf2, a.iters);
    else if (a.mode == "idle") {}
    else { fprintf(stderr, "unknown mode %s\n", a.mode.c_str()); exit(1); }
  };
  std::mt19937_64 rng(a.seed);
  if (a.mode == "dispatch") {
    // Per-workgroup CU and timing over --trials launches (after --warmups); no bandwidth is reported.
    CHECK(hipMalloc(&buf, n * 16)); CHECK(hipMemset(buf, 0, n * 16));
    CHECK(hipMalloc(&buf2, (size_t)a.blocks * 24));
    std::vector<unsigned long long> rec((size_t)a.blocks * 3);
    for (int w = 0; w < a.warmups; ++w) launch();
    CHECK(hipStreamSynchronize(s));
    hipEvent_t d0, d1; CHECK(hipEventCreate(&d0)); CHECK(hipEventCreate(&d1));
    printf("{\"classification\":\"measured-real-GPU\",\"mode\":\"dispatch\",\"device\":\"%s\",\"arch\":\"%s\",\"cus\":%d,"
           "\"hsa_cu_mask\":\"%s\",\"blocks\":%d,\"threads\":%d,\"lds_bytes\":%d,\"footprint_bytes\":%zu,\"wall_clock_khz\":%d,"
           "\"key_layout\":\"xcc<<8|se<<5|sh<<4|cu\",\"trials\":[",
           p.name, p.gcnArchName, cus, getenv("HSA_CU_MASK") ? getenv("HSA_CU_MASK") : "", a.blocks, a.threads, a.lds,
           a.bytes, wall_khz);
    bool complete = true;
    for (int t = 0; t < std::max(a.trials, 1); ++t) {
      CHECK(hipMemset(buf2, 0xff, (size_t)a.blocks * 24));
      CHECK(hipEventRecord(d0, s)); launch(); CHECK(hipEventRecord(d1, s)); CHECK(hipEventSynchronize(d1));
      CHECK(hipGetLastError());
      float km; CHECK(hipEventElapsedTime(&km, d0, d1));
      CHECK(hipMemcpy(rec.data(), buf2, rec.size() * 8, hipMemcpyDeviceToHost));
      unsigned long long t_min = ~0ull;
      for (int b = 0; b < a.blocks; ++b) { if (rec[3 * b] == ~0ull) complete = false; t_min = std::min(t_min, rec[3 * b + 1]); }
      printf("%s{\"kernel_ms\":%.6f,\"wg\":[", t ? "," : "", km);
      for (int b = 0; b < a.blocks; ++b)
        printf("%s[%llu,%.1f,%.1f]", b ? "," : "", rec[3 * b], (rec[3 * b + 1] - t_min) * 1e6 / wall_khz,
               (rec[3 * b + 2] - t_min) * 1e6 / wall_khz);
      printf("]}");
    }
    printf("],\"wg_fields\":\"[cu key, start ns, end ns] relative to the first start\",\"check\":\"every workgroup recorded\","
           "\"check_passed\":%s}\n", complete ? "true" : "false");
    return complete ? 0 : 3;
  }
  if (a.mode == "placement") {
    // Distinct physical CUs observed over several launches; no timing is reported for this mode.
    a.threads = 64;
    CHECK(hipMalloc(&buf, (size_t)a.blocks * 8));
    std::set<uint32_t> cu_keys;
    std::map<int, std::set<uint32_t>> per_xcc;
    std::vector<uint32_t> rec((size_t)a.blocks * 2);
    for (int t = 0; t < std::max(a.trials, 1); ++t) {
      CHECK(hipMemset(buf, 0xff, (size_t)a.blocks * 8));
      launch();
      CHECK(hipStreamSynchronize(s));
      CHECK(hipGetLastError());
      CHECK(hipMemcpy(rec.data(), buf, rec.size() * 4, hipMemcpyDeviceToHost));
      for (int b = 0; b < a.blocks; ++b) {
        uint32_t hw = rec[2 * b], xcc = rec[2 * b + 1] & 0xf;
        uint32_t key = xcc << 8 | ((hw >> 13) & 7) << 5 | ((hw >> 12) & 1) << 4 | ((hw >> 8) & 0xf);
        cu_keys.insert(key); per_xcc[(int)xcc].insert(key);
      }
    }
    int expect = a.expect_cus ? a.expect_cus : mask_cus;
    printf("{\"classification\":\"measured-real-GPU\",\"mode\":\"placement\",\"device\":\"%s\",\"arch\":\"%s\",\"cus\":%d,"
           "\"stream_cu_mask\":\"%s\",\"hsa_cu_mask\":\"%s\",\"blocks\":%d,\"launches\":%d,\"spin_cycles\":%ld,"
           "\"distinct_cus\":%zu,\"expected_cus\":%d,\"per_xcc\":{",
           p.name, p.gcnArchName, cus, a.cu_mask.c_str(), getenv("HSA_CU_MASK") ? getenv("HSA_CU_MASK") : "",
           a.blocks, std::max(a.trials, 1), a.spin, cu_keys.size(), expect);
    bool first = true;
    for (auto& kv : per_xcc) { printf("%s\"%d\":%zu", first ? "" : ",", kv.first, kv.second.size()); first = false; }
    printf("},\"cu_keys\":[");
    first = true;
    for (auto k : cu_keys) { printf("%s%u", first ? "" : ",", k); first = false; }
    bool okp = (int)cu_keys.size() == expect;
    printf("],\"key_layout\":\"xcc<<8|se<<5|sh<<4|cu from HW_REG_HW_ID/HW_REG_XCC_ID\",\"check\":\"distinct CUs equal expected\","
           "\"check_passed\":%s}\n", okp ? "true" : "false");
    return okp ? 0 : 3;
  }
  if (a.mode == "read" || a.mode == "write" || a.mode == "strided" || a.mode == "mixed" || a.mode == "gather" || a.mode == "copy"
      || a.mode == "read_ilp" || a.mode == "split" || a.mode == "fused" || (a.mode == "idle" && a.seconds > 0)) {
    CHECK(hipMalloc(&buf, n * 16));
    CHECK(hipMemset(buf, 0, n * 16));
    if (a.mode == "copy") { CHECK(hipMalloc(&buf2, n * 16)); CHECK(hipMemset(buf2, 0x7f, n * 16)); }
    if (a.mode == "gather") {
      std::vector<uint32_t> h(n);
      std::uniform_int_distribution<uint32_t> d(0, (uint32_t)(n - 1));
      for (auto& x : h) x = d(rng);
      CHECK(hipMalloc(&idx, n * 4));
      CHECK(hipMemcpy(idx, h.data(), n * 4, hipMemcpyHostToDevice));
    }
    double per = a.mode == "copy" ? 2.0 * n * 16 : a.mode == "strided" ? (double)(n / a.stride) * 16
               : a.mode == "gather" ? n * 20.0 : n * 16.0;
    useful_bytes = per * a.reps * (a.mode == "read" ? a.passes : 1) * (a.mode == "idle" ? 0 : 1);
    if (a.mode == "mixed") useful_flops = 2.0 * a.intensity * n * a.reps;
    if (a.mode == "split") useful_flops = 2048.0 * 8 * a.iters * (double)(a.blocks / 2) * (a.threads / 64) * a.reps;
    if (a.mode == "fused") useful_flops = 2048.0 * ((a.intensity + 1) / 2 * 2) * ((double)n / 64) * a.reps;
  } else if (a.mode == "chase") {
    size_t nodes = a.bytes / a.line;
    size_t spacing = a.line / 4;
    std::vector<uint32_t> order(nodes);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);  // one random cycle through all nodes
    host_next.assign(nodes * spacing, 0);
    for (size_t i = 0; i < nodes; ++i) host_next[order[i] * spacing] = (uint32_t)(order[(i + 1) % nodes] * spacing);
    std::vector<uint32_t> st(a.chains);
    for (int c = 0; c < a.chains; ++c) st[c] = (uint32_t)(order[(size_t)c * nodes / a.chains] * spacing);
    CHECK(hipMalloc(&buf, host_next.size() * 4));
    CHECK(hipMemcpy(buf, host_next.data(), host_next.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&starts, a.chains * 4));
    CHECK(hipMemcpy(starts, st.data(), a.chains * 4, hipMemcpyHostToDevice));
    CHECK(hipMalloc(&buf2, a.chains * 4));
    a.reps = 1;
  } else if (a.mode == "cuprobe") {
    CHECK(hipMalloc(&buf, 4096 * 4)); CHECK(hipMemset(buf, 0, 4096 * 4));
  } else if (a.mode == "cuprobe_t") {
    CHECK(hipMalloc(&buf, 4096 * 2 * 8)); CHECK(hipMemset(buf, 0, 4096 * 2 * 8));
    std::vector<uint32_t> ord(ring_mask + 1), ring(ring_mask + 1);
    std::iota(ord.begin(), ord.end(), 0);
    std::shuffle(ord.begin(), ord.end(), rng);  // one random cycle through the ring
    for (size_t i = 0; i < ord.size(); ++i) ring[ord[i]] = ord[(i + 1) % ord.size()];
    CHECK(hipMalloc(&buf2, ring.size() * 4));
    CHECK(hipMemcpy(buf2, ring.data(), ring.size() * 4, hipMemcpyHostToDevice));
  } else if (a.mode == "gapprobe") {
    CHECK(hipMalloc(&buf2, (size_t)a.blocks * 24));
  } else if (a.mode == "fma") {
    useful_flops = 2.0 * 8 * a.iters * (double)a.blocks * a.threads * a.reps;
  } else if (a.mode == "mfma4") {
    useful_flops = 2048.0 * 8 * a.iters * (double)a.blocks * (a.threads / 64) * a.reps;
  } else if (a.mode == "mfma4_lds") {
    useful_flops = 2048.0 * 4 * a.iters * (double)a.blocks * (a.threads / 64) * a.reps;
  } else if (a.mode == "mfma") {
    useful_flops = 2.0 * 32 * 32 * 8 * 4 * a.iters * (double)a.blocks * (a.threads / 64) * a.reps;
  }
  if (a.seconds > 0) {
    // Competitor mode (E3): owned background load on its own CU mask. Emits a "started" event after the
    // first completed launch (proof of activity), one JSON line per ~0.25 s window, and a "stopped" event
    // after SIGTERM/SIGINT or the time limit. Realtime stamps join to decoder trial windows (same host
    // clock); monotonic stamps detect realtime steps.
    using clk = std::chrono::steady_clock;
    auto rt_ns = [] { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count(); };
    auto mono_ns = [] { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          clk::now().time_since_epoch()).count(); };
    signal(SIGTERM, on_signal); signal(SIGINT, on_signal);
    const char* mask_env = getenv("HSA_CU_MASK") ? getenv("HSA_CU_MASK") : "";
    bool idle = a.mode == "idle";
    bool probe = a.mode == "cuprobe", probe_t = a.mode == "cuprobe_t", gap = a.mode == "gapprobe";
    if (!idle && !probe && !probe_t && !gap && a.mode != "read" && a.mode != "fma") {
      fprintf(stderr, "competitor supports read, fma, cuprobe, cuprobe_t, gapprobe, idle\n"); exit(1);
    }
    std::vector<unsigned long long> grec((size_t)a.blocks * 3);
    std::vector<long> g_seen(4096, 0), g_late(4096, 0), g_multi(4096, 0);
    std::vector<unsigned long long> g_first(4096);
    std::vector<int> g_count(4096, 0);
    const unsigned long long late_ticks = (unsigned long long)(a.late_ns * wall_khz / 1e6);
    std::vector<uint32_t> cu_now(4096, 0), cu_prev(4096, 0);
    std::vector<unsigned long long> ct_now(8192, 0), ct_prev(8192, 0);
    auto end = clk::now() + std::chrono::duration<double>(a.seconds);
    double per_bytes = a.mode == "read" ? (double)n * 16 * a.passes : 0;
    double per_flops = (a.mode == "fma" || probe) ? 2.0 * 8 * a.iters * (double)a.blocks * a.threads
                     : probe_t ? 2.0 * 4 * a.iters * (double)a.blocks * a.threads : 0;
    double busy = 0, bytes = 0, flops = 0; long launches = 0, total_launches = 0;
    bool started = false;
    auto win = clk::now(); long long w0 = rt_ns(), m0 = mono_ns();
    auto emit = [&](bool final_window) {
      double w = std::chrono::duration<double>(clk::now() - win).count();
      long long w1 = rt_ns(), m1 = mono_ns();
      if (w > 0) printf("{\"t0_unix_ns\":%lld,\"t1_unix_ns\":%lld,\"t0_mono_ns\":%lld,\"t1_mono_ns\":%lld,\"mode\":\"%s\","
                        "\"duty_requested\":%.3f,\"busy_fraction\":%.4f,\"launches\":%ld,\"gb_per_s\":%.3f,\"tflop_per_s\":%.3f,"
                        "\"hsa_cu_mask\":\"%s\",\"final\":%s}\n", w0, w1, m0, m1, a.mode.c_str(), a.duty, busy / w, launches,
                        bytes / w / 1e9, flops / w / 1e12, mask_env, final_window ? "true" : "false");
      if (w > 0 && probe) {  // per-CU completed workgroups in this window, as a separate line joined by t1
        CHECK(hipMemcpy(cu_now.data(), buf, 4096 * 4, hipMemcpyDeviceToHost));
        printf("{\"cu_window_t1_unix_ns\":%lld,\"cu_blocks\":{", w1);
        bool first = true;
        for (int k = 0; k < 4096; ++k) if (cu_now[k] != cu_prev[k]) {
          printf("%s\"%d\":%u", first ? "" : ",", k, cu_now[k] - cu_prev[k]); first = false;
        }
        printf("}}\n");
        cu_prev = cu_now;
      }
      if (w > 0 && probe_t) {  // per-CU completions and mean residency (ns) in this window, joined by t1
        CHECK(hipMemcpy(ct_now.data(), buf, 8192 * 8, hipMemcpyDeviceToHost));
        printf("{\"cu_window_t1_unix_ns\":%lld,\"cu_blocks\":{", w1);
        bool first = true;
        for (int k = 0; k < 4096; ++k) if (ct_now[2 * k + 1] != ct_prev[2 * k + 1]) {
          printf("%s\"%d\":%llu", first ? "" : ",", k, ct_now[2 * k + 1] - ct_prev[2 * k + 1]); first = false;
        }
        printf("},\"cu_mean_ns\":{");
        first = true;
        for (int k = 0; k < 4096; ++k) if (ct_now[2 * k + 1] != ct_prev[2 * k + 1]) {
          double ticks = (double)(ct_now[2 * k] - ct_prev[2 * k]) / (double)(ct_now[2 * k + 1] - ct_prev[2 * k + 1]);
          printf("%s\"%d\":%.1f", first ? "" : ",", k, ticks * 1e6 / wall_khz); first = false;
        }
        printf("}}\n");
        ct_prev = ct_now;
      }
      if (w > 0 && gap) {  // per-CU launches with a workgroup, late launches, launches with >= 2 workgroups
        auto dump = [&](const char* name, std::vector<long>& v, bool lead) {
          printf("%s\"%s\":{", lead ? "" : "},", name);
          bool first = true;
          for (int k = 0; k < 4096; ++k) if (g_seen[k] || v[k]) { printf("%s\"%d\":%ld", first ? "" : ",", k, v[k]); first = false; }
        };
        printf("{\"cu_window_t1_unix_ns\":%lld,\"gap_launches\":%ld,\"late_ns\":%.0f,", w1, launches, a.late_ns);
        dump("cu_blocks", g_seen, true); dump("cu_late", g_late, false); dump("cu_multi", g_multi, false);
        printf("}}\n");
        std::fill(g_seen.begin(), g_seen.end(), 0); std::fill(g_late.begin(), g_late.end(), 0);
        std::fill(g_multi.begin(), g_multi.end(), 0);
      }
      fflush(stdout); win = clk::now(); w0 = w1; m0 = m1; busy = bytes = flops = 0; launches = 0;
    };
    while (!g_stop && clk::now() < end) {
      if (idle) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      } else if (gap) {
        auto t0 = clk::now();
        launch();
        CHECK(hipStreamSynchronize(s));
        busy += std::chrono::duration<double>(clk::now() - t0).count(); ++launches; ++total_launches;
        CHECK(hipMemcpy(grec.data(), buf2, grec.size() * 8, hipMemcpyDeviceToHost));
        unsigned long long t_min = ~0ull;
        std::vector<uint32_t> keys;
        for (int b = 0; b < a.blocks; ++b) {
          uint32_t key = (uint32_t)grec[3 * b] & 0xfff;
          t_min = std::min(t_min, grec[3 * b + 1]);
          if (g_count[key]++ == 0) { g_first[key] = grec[3 * b + 1]; keys.push_back(key); }
          else g_first[key] = std::min(g_first[key], grec[3 * b + 1]);
        }
        for (auto key : keys) {
          ++g_seen[key];
          if (g_first[key] - t_min > late_ticks) ++g_late[key];
          if (g_count[key] >= 2) ++g_multi[key];
          g_count[key] = 0;
        }
        std::this_thread::sleep_for(std::chrono::duration<double>(a.gap_us * 1e-6));
      } else {
        auto t0 = clk::now();
        launch();
        CHECK(hipStreamSynchronize(s));
        double k = std::chrono::duration<double>(clk::now() - t0).count();
        busy += k; bytes += per_bytes; flops += per_flops; ++launches; ++total_launches;
        if (a.duty < 1.0 && a.duty > 0) std::this_thread::sleep_for(std::chrono::duration<double>(k * (1.0 - a.duty) / a.duty));
      }
      if (!started) {
        started = true;
        printf("{\"event\":\"started\",\"pid\":%d,\"unix_ns\":%lld,\"mono_ns\":%lld,\"mode\":\"%s\",\"duty_requested\":%.3f,"
               "\"hsa_cu_mask\":\"%s\",\"footprint_bytes\":%zu,\"lds_bytes\":%d}\n", (int)getpid(), rt_ns(), mono_ns(), a.mode.c_str(), a.duty,
               mask_env, idle ? (size_t)0 : a.bytes, (a.mode == "cuprobe" || gap) ? a.lds : 0);
        fflush(stdout);
      }
      if (std::chrono::duration<double>(clk::now() - win).count() >= 0.25) emit(false);
    }
    emit(true);
    CHECK(hipGetLastError());
    printf("{\"event\":\"stopped\",\"pid\":%d,\"unix_ns\":%lld,\"mono_ns\":%lld,\"reason\":\"%s\",\"total_launches\":%ld}\n",
           (int)getpid(), rt_ns(), mono_ns(), g_stop ? "signal" : "time_limit", total_launches);
    fflush(stdout);
    return 0;
  }
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
  for (int w = 0; w < a.warmups; ++w) { for (int r = 0; r < a.reps; ++r) launch(); }
  CHECK(hipStreamSynchronize(s));
  CHECK(hipGetLastError());
  std::vector<double> ms;
  for (int t = 0; t < a.trials; ++t) {
    CHECK(hipEventRecord(e0, s));
    for (int r = 0; r < a.reps; ++r) launch();
    CHECK(hipEventRecord(e1, s));
    CHECK(hipEventSynchronize(e1));
    float m; CHECK(hipEventElapsedTime(&m, e0, e1));
    ms.push_back(m);
  }
  CHECK(hipGetLastError());
  if (a.mode == "copy") {
    std::vector<float> h(1024);
    CHECK(hipMemcpy(h.data(), buf2, 4096, hipMemcpyDeviceToHost));
    ok = std::all_of(h.begin(), h.end(), [](float x) { return x == 0.0f; });
    check = "destination prefix equals zero-filled source";
  } else if (a.mode == "write") {
    std::vector<float> h(1024);
    CHECK(hipMemcpy(h.data(), buf, 4096, hipMemcpyDeviceToHost));
    ok = std::all_of(h.begin(), h.end(), [](float x) { return x == 1.0f; });
    check = "buffer prefix equals written value";
  } else if (a.mode == "chase") {
    // Host replay of chain 0 for the same number of steps.
    std::vector<uint32_t> dev(a.chains);
    CHECK(hipMemcpy(dev.data(), buf2, a.chains * 4, hipMemcpyDeviceToHost));
    uint32_t q = 0;
    { std::vector<uint32_t> st(1); CHECK(hipMemcpy(st.data(), starts, 4, hipMemcpyDeviceToHost)); q = st[0]; }
    for (long i = 0; i < a.steps; ++i) q = host_next[q];
    ok = dev[0] == q;
    check = "chain0 endpoint equals host replay";
    per_access = 0;
  }
  std::vector<double> sorted = ms;
  std::sort(sorted.begin(), sorted.end());
  double med = sorted[sorted.size() / 2];
  printf("{\"classification\":\"measured-real-GPU\",\"mode\":\"%s\",\"device\":\"%s\",\"arch\":\"%s\",\"cus\":%d,"
         "\"stream_cu_mask\":\"%s\",\"stream_cus\":%d,\"hsa_cu_mask\":\"%s\",\"footprint_bytes\":%zu,\"blocks\":%d,\"threads\":%d,"
         "\"reps\":%d,\"warmups\":%d,\"stride\":%d,\"chains\":%d,\"steps\":%ld,\"line\":%d,\"intensity\":%d,\"iters\":%ld,\"passes\":%d,"
         "\"seed\":%llu,\"trial_ms\":[",
         a.mode.c_str(), p.name, p.gcnArchName, cus, a.cu_mask.c_str(), mask_cus,
         getenv("HSA_CU_MASK") ? getenv("HSA_CU_MASK") : "", a.bytes, a.blocks, a.threads, a.reps, a.warmups,
         a.stride, a.chains, a.steps, a.line, a.intensity, a.iters, a.passes, a.seed);
  for (size_t i = 0; i < ms.size(); ++i) printf("%s%.6f", i ? "," : "", ms[i]);
  printf("],\"median_ms\":%.6f", med);
  if (useful_bytes > 0) printf(",\"useful_bytes\":%.0f,\"useful_gb_per_s\":%.3f", useful_bytes, useful_bytes / med / 1e6);
  if (useful_flops > 0) printf(",\"useful_flops\":%.0f,\"useful_tflop_per_s\":%.3f", useful_flops, useful_flops / med / 1e9);
  if (a.mode == "chase") printf(",\"ns_per_dependent_load\":%.3f,\"total_dependent_loads\":%ld", med * 1e6 / a.steps, a.steps * (long)a.chains);
  printf(",\"check\":\"%s\",\"check_passed\":%s,\"scope\":\"useful bytes/flops are logical; physical HBM traffic requires counters\"}\n",
         check.c_str(), ok ? "true" : "false");
  return ok ? 0 : 3;
}
