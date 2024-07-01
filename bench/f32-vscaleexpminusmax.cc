#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/reduce.h"
#include "xnnpack/raddexpminusmax.h"
#include "xnnpack/vscaleexpminusmax.h"


static void f32_vscaleexpminusmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_fn rmax,
  xnn_f32_raddexpminusmax_ukernel_fn raddexpminusmax,
  xnn_f32_vscaleexpminusmax_ukernel_fn vscaleexpminusmax,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_elements * sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> x(elements);
  std::vector<float, AlignedAllocator<float, 64>> y(packed_elements * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float x_max = nanf("");
    rmax(elements * sizeof(float), x.data(), &x_max, /*params=*/nullptr);
    float y_sum = nanf("");
    raddexpminusmax(elements * sizeof(float), x.data(), &y_sum, x_max);
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    vscaleexpminusmax(elements * sizeof(float), x.data(), y.data() + packed_elements * buffer_index, x_max, 1.0f / y_sum);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgName("N");
  for (int32_t n = 10000; n <= 100000000; n *= 10) {
    b->Arg(n);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u16,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u32,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u32,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u48,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u48,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u64,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u64,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u80,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u80,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u96,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u96,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u112,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u112,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u128,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u128,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u144,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u144,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u160,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u160,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u176,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u176,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_u192,
    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_u128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_u192,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u8,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u8,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u16,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u16,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u24,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u32,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u40,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u48,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u48,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u56,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u56,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u64,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u64,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u72,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u72,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u80,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u80,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u88,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u88,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_u96,
    xnn_f32_rmax_ukernel__avx_u32_acc4,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_u80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_u96,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
