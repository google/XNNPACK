#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/raddexpminusmax.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>
#include <xnnpack/vscaleexpminusmax.h>

#include <benchmark/benchmark.h>


static void ThreePassSoftargmaxWithRecomputing(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function rmax,
  xnn_f32_raddexpminusmax_ukernel_function raddexpminusmax,
  xnn_f32_vscaleexpminusmax_ukernel_function vscaleexpminusmax)
{
  const size_t n = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::roundUp(n, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::divideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float> x(n);
  std::vector<float> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    benchmark::utils::prefetchToL1(x.data(), x.size() * sizeof(float));
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    const auto start = std::chrono::high_resolution_clock::now();
    float x_max = nanf("");
    rmax(n * sizeof(float), x.data(), &x_max);
    float y_sum = nanf("");
    raddexpminusmax(n * sizeof(float), x.data(), &y_sum, x_max);
    vscaleexpminusmax(n * sizeof(float), x.data(), y.data() + packed_n * buffer_index, x_max, 1.0f / y_sum);
    const auto end = std::chrono::high_resolution_clock::now();

    const auto elapsed_seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }

  state.SetItemsProcessed(uint64_t(state.iterations()) * n);
  state.SetBytesProcessed(uint64_t(state.iterations()) * 2 * sizeof(float) * n);
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  for (int32_t n = 1000; n <= 10000000; n *= 10) {
    b->Arg(n);
    b->Arg(3 * n);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(ThreePassSoftargmaxWithRecomputing, avx512f_p5_scalef_unroll128,
    xnn_f32_rmax_ukernel__avx512f, xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_unroll128, xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_unroll128)
      ->Apply(CharacteristicArguments)->UseManualTime();

  BENCHMARK_CAPTURE(ThreePassSoftargmaxWithRecomputing, avx2_p5_unroll64,
    xnn_f32_rmax_ukernel__avx, xnn_f32_raddexpminusmax_ukernel__avx2_p5_unroll64, xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_unroll64)
      ->Apply(CharacteristicArguments)->UseManualTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
