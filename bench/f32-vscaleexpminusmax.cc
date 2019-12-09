#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/rmax.h>
#include <xnnpack/raddexpminusmax.h>
#include <xnnpack/vscaleexpminusmax.h>

#include <benchmark/benchmark.h>


static void f32_vscaleexpminusmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function rmax,
  xnn_f32_raddexpminusmax_ukernel_function raddexpminusmax,
  xnn_f32_vscaleexpminusmax_ukernel_function vscaleexpminusmax,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t n = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::RoundUp(n, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), rng);

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> x(n);
  std::vector<float, AlignedAllocator<float, 64>> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float x_max = nanf("");
    rmax(n * sizeof(float), x.data(), &x_max);
    float y_sum = nanf("");
    raddexpminusmax(n * sizeof(float), x.data(), &y_sum, x_max);
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    vscaleexpminusmax(n * sizeof(float), x.data(), y.data() + packed_n * buffer_index, x_max, 1.0f / y_sum);
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * 2 * sizeof(float) * n, benchmark::Counter::kIsRate);
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgName("N");
  for (int32_t n = 10000; n <= 100000000; n *= 10) {
    b->Arg(n);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x16,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x32,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x32,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x48,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x48,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x64,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x64,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x80,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x80,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x96,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x96,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x112,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x112,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x128,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x128,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x144,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x144,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x160,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x160,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x176,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x176,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx512f_p5_scalef_x192,
    xnn_f32_rmax_ukernel__avx512f,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx512f_p5_scalef_x192,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x8,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x8,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x16,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x16,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x24,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x32,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x40,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x48,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x48,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x56,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x56,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x64,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x64,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x72,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x72,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x80,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x80,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x88,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x88,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleexpminusmax, avx2_p5_x96,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleexpminusmax_ukernel__avx2_p5_x96,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
