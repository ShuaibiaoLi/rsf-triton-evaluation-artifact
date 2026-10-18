// Frozen v2 implementation is included unchanged so that its 60-step kernel,
// independent scalar reference, and event timer remain the control condition.
#include <cstring>
#define main rsf_c3_fixed_replay_v2_entry_point
#include "rsf_c3_fixed_replay_native_v2.cpp"
#undef main

__global__ void bounded_replay22_kernel(const Input* in, double* v, double* mu,
                                        double* g, int* iterations,
                                        int* converged, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Input x = in[i];
  double z = fmax(x.prev, 1e-45);
  double C = .5 / x.v0 * exp((x.f0 + x.b * log(x.v0 * x.psi / x.L)) / x.a);
  double inv_eta = 1.0 / x.eta;
  for (int k = 0; k < 22; k++) {
    bool evaluating = k < x.actual;
    if (!evaluating) continue;
    double q = z * C, s = sqrt(q * q + 1.0);
    double m = x.a * log(q + s), d = x.a * C / s;
    double r = (x.tau - x.sigma * m) * inv_eta - z;
    bool last = k + 1 == x.actual;
    if (!last || !x.converged)
      z = fmax(1e-45, z - r / (-(x.sigma * inv_eta) * d - 1.0));
  }
  double q = z * C, m = x.a * asinh_equivalent(q);
  v[i] = z;
  mu[i] = m;
  g[i] = (x.tau - x.sigma * m) * inv_eta - z;
  iterations[i] = x.actual;
  converged[i] = x.converged;
}

int main(int argc, char** argv) {
  int n = argc > 1 ? std::atoi(argv[1]) : 65537;
  int warmup = argc > 2 ? std::atoi(argv[2]) : 10;
  int repeat = argc > 3 ? std::atoi(argv[3]) : 30;
  bool smoke = argc > 4 && std::strcmp(argv[4], "smoke") == 0;
  if (n < 1 || warmup < 0 || repeat < 2) return 2;
  const Input base[] = {
    {8e6,1e7,.2,.01,.015,.6,1e-6,.02,4e6,1e-6,1e-8,60,8,1},
    {5.8e6,1e7,2e4,.012,.018,.6,1e-6,.02,4e6,0,1e-8,60,22,1},
    {1e5,1e7,2e4,.012,.018,.6,1e-6,.02,4e6,1e-12,1e-8,60,6,1},
    {1.2e7,1e7,.02,.008,.012,.58,1e-6,.01,2e6,1,1e-8,60,4,1},
    {2e4,1e3,2e4,.01,.015,.6,1e-6,.02,4e6,1e-6,1e-8,60,4,1},
    {7e6,1e7,.2,.004,.018,.6,1e-6,.01,3e6,1e-5,1e-8,60,6,1},
    {8e6,1e7,.2,.01,.015,.6,1e-6,.02,4e6,1e-4,1e-8,60,6,1},
    {8e6,1e7,.2,.01,.015,.6,1e-6,.02,4e6,1e-45,1e-8,1,1,0},
    {6.5e6,9e6,.05,.009,.014,.59,1e-6,1e-4,3.5e6,1e-3,1e-8,60,6,1}
  };
  std::vector<Input> host(n);
  for (int i = 0; i < n; i++) host[i] = base[i % 9];
  Input* input;
  double *v, *mu, *g;
  int *iterations, *converged;
  HIP_OK(hipMalloc(&input, (size_t)n * sizeof(Input)));
  HIP_OK(hipMalloc(&v, (size_t)n * sizeof(double)));
  HIP_OK(hipMalloc(&mu, (size_t)n * sizeof(double)));
  HIP_OK(hipMalloc(&g, (size_t)n * sizeof(double)));
  HIP_OK(hipMalloc(&iterations, (size_t)n * sizeof(int)));
  HIP_OK(hipMalloc(&converged, (size_t)n * sizeof(int)));
  HIP_OK(hipMemcpy(input, host.data(), (size_t)n * sizeof(Input), hipMemcpyHostToDevice));
  hipStream_t stream;
  HIP_OK(hipStreamCreate(&stream));
  dim3 block(256), grid((n + 255) / 256);
  auto pre60 = [&]() { fixed_replay_kernel<<<grid, block, 0, stream>>>(
      input, v, mu, g, iterations, converged, n); };
  auto bound22 = [&]() { bounded_replay22_kernel<<<grid, block, 0, stream>>>(
      input, v, mu, g, iterations, converged, n); };

  auto check = [&](auto launch, const char* variant) {
    launch();
    HIP_OK(hipStreamSynchronize(stream));
    std::vector<double> hv(n), hm(n), hg(n);
    std::vector<int> hi(n), hc(n);
    HIP_OK(hipMemcpy(hv.data(), v, (size_t)n * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(hm.data(), mu, (size_t)n * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(hg.data(), g, (size_t)n * sizeof(double), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(hi.data(), iterations, (size_t)n * sizeof(int), hipMemcpyDeviceToHost));
    HIP_OK(hipMemcpy(hc.data(), converged, (size_t)n * sizeof(int), hipMemcpyDeviceToHost));
    double max_error = 0.0;
    int failures = 0;
    for (int i = 0; i < n; i++) {
      Output truth = cpu(host[i]);
      double error = std::max({std::fabs(hv[i] - truth.v),
                               std::fabs(hm[i] - truth.mu),
                               std::fabs(hg[i] - truth.g)});
      max_error = std::max(max_error, error);
      failures += error > 1e-10 || hi[i] != truth.iterations ||
                  hc[i] != truth.converged;
    }
    std::printf("correctness variant=%s n=%d max_abs_error=%.17g failures=%d status=%s\n",
                variant, n, max_error, failures, failures ? "FAIL" : "PASS");
    if (failures) std::exit(3);
  };
  check(pre60, "pre60");
  check(bound22, "bound22");
  if (smoke) {
    std::printf("RSF_C3_LOOP_BOUND_NATIVE_V3_PASS mode=smoke\n");
    return 0;
  }
  auto before = measure(pre60, stream, warmup, repeat);
  auto bounded = measure(bound22, stream, warmup, repeat);
  auto restore = measure(pre60, stream, warmup, repeat);
  emit("pre60", before);
  emit("bound22", bounded);
  emit("restore60", restore);
  double ratio = percentile(restore, .5) / percentile(before, .5);
  std::printf("reversibility restore_over_pre=%.9f status=%s\n",
              ratio, ratio >= .9 && ratio <= 1.1 ? "PASS" : "FAIL");
  if (ratio < .9 || ratio > 1.1) return 4;
  std::printf("RSF_C3_LOOP_BOUND_NATIVE_V3_PASS\n");
  return 0;
}
