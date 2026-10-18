// Independent service microbenchmarks for gfx942 (MI300X/MI325X): contract M4 and A-D, reused for M7.
// Each run prints one JSON line with event-timed trials and a correctness check.
// Measured-real-GPU when run on the declared GPU; physical HBM bytes need rocprofv3 counters separately.
#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <chrono>
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
};

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
    else if (k == "--duty") a.duty = std::stod(v);
    else if (k == "--seed") a.seed = std::stoull(v);
    else if (k == "--cu-mask") a.cu_mask = v;
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
    else if (a.mode == "mixed") hipLaunchKernelGGL(k_mixed, a.blocks, a.threads, 0, s, (const float4*)buf, n, a.intensity, out);
    else { fprintf(stderr, "unknown mode %s\n", a.mode.c_str()); exit(1); }
  };
  std::mt19937_64 rng(a.seed);
  if (a.mode == "read" || a.mode == "write" || a.mode == "strided" || a.mode == "mixed" || a.mode == "gather" || a.mode == "copy") {
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
    useful_bytes = per * a.reps * (a.mode == "read" ? a.passes : 1);
    if (a.mode == "mixed") useful_flops = 2.0 * a.intensity * n * a.reps;
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
  } else if (a.mode == "fma") {
    useful_flops = 2.0 * 8 * a.iters * (double)a.blocks * a.threads * a.reps;
  } else if (a.mode == "mfma") {
    useful_flops = 2.0 * 32 * 32 * 8 * 4 * a.iters * (double)a.blocks * (a.threads / 64) * a.reps;
  }
  if (a.seconds > 0) {
    // Competitor mode (contract J6): owned background load on its own CU mask. Emits one JSON line per
    // ~0.25 s window with wall timestamps so a concurrently measured decoder can be joined to it.
    using clk = std::chrono::steady_clock;
    auto now_ns = [] { return (long long)std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch()).count(); };
    auto end = clk::now() + std::chrono::duration<double>(a.seconds);
    auto win = clk::now(); double busy = 0, bytes = 0, flops = 0; long long w0 = now_ns();
    double per_bytes = a.mode == "read" ? (double)n * 16 * a.passes : 0;
    double per_flops = a.mode == "fma" ? 2.0 * 8 * a.iters * (double)a.blocks * a.threads : 0;
    while (clk::now() < end) {
      auto t0 = clk::now();
      launch();
      CHECK(hipStreamSynchronize(s));
      double k = std::chrono::duration<double>(clk::now() - t0).count();
      busy += k; bytes += per_bytes; flops += per_flops;
      if (a.duty < 1.0) std::this_thread::sleep_for(std::chrono::duration<double>(k * (1.0 - a.duty) / a.duty));
      double w = std::chrono::duration<double>(clk::now() - win).count();
      if (w >= 0.25) {
        long long w1 = now_ns();
        printf("{\"t0_unix_ns\":%lld,\"t1_unix_ns\":%lld,\"mode\":\"%s\",\"duty_requested\":%.3f,\"busy_fraction\":%.4f,"
               "\"gb_per_s\":%.3f,\"tflop_per_s\":%.3f,\"hsa_cu_mask\":\"%s\"}\n", w0, w1, a.mode.c_str(), a.duty, busy / w,
               bytes / w / 1e9, flops / w / 1e12, getenv("HSA_CU_MASK") ? getenv("HSA_CU_MASK") : "");
        fflush(stdout); win = clk::now(); w0 = w1; busy = bytes = flops = 0;
      }
    }
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
