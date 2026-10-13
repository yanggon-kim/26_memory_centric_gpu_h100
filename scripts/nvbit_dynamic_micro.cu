// Dynamic-input tracer health check; not an LLM workload or precision comparison.
#include <cuda_runtime.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

__global__ void add_dynamic(const uint32_t* a, const uint32_t* b, uint32_t* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  if (argc != 3 || std::strncmp(argv[1], "GPU-", 4) != 0) return 1;
  char lock_path[256];
  std::snprintf(lock_path, sizeof(lock_path), "/tmp/h100_research_%s.lock", argv[1]);
  int lock_fd = open(lock_path, O_CREAT | O_RDWR, 0600);
  if (lock_fd < 0 || flock(lock_fd, LOCK_EX | LOCK_NB) != 0) return 1;
  int count = 0;
  CHECK(cudaGetDeviceCount(&count));
  if (count != 1) return 1;
  cudaDeviceProp props{};
  CHECK(cudaGetDeviceProperties(&props, 0));
  char actual[64];
  auto* u = reinterpret_cast<const unsigned char*>(props.uuid.bytes);
  std::snprintf(actual, sizeof(actual),
      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  if (std::strcmp(actual, argv[1]) != 0 || props.major != 9 || props.minor != 0) {
    std::fprintf(stderr, "wrong selected physical GPU: %s\n", actual);
    return 1;
  }
  constexpr int n = 1024;
  std::vector<uint32_t> a(n), b(n), out(n);
  uint32_t state = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  for (int i = 0; i < n; ++i) {
    state = 1664525u * state + 1013904223u; a[i] = state;
    state = 1664525u * state + 1013904223u; b[i] = state;
  }
  uint32_t *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(cudaMalloc(&da, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&db, n * sizeof(uint32_t)));
  CHECK(cudaMalloc(&dc, n * sizeof(uint32_t)));
  CHECK(cudaMemcpy(da, a.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice));
  add_dynamic<<<4,256>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(out.data(), dc, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) if (out[i] != a[i] + b[i]) return 3;
  CHECK(cudaFree(da)); CHECK(cudaFree(db)); CHECK(cudaFree(dc));
  std::printf("dynamic micro PASS uuid=%s elements=%d seed=%s\n", actual, n, argv[2]);
  return 0;
}
