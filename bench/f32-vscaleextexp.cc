#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/raddextexp.h>
#include <xnnpack/vscaleextexp.h>


static void f32_vscaleextexp(
  benchmark::State& state,
  xnn_f32_raddextexp_ukernel_fn raddextexp,
  xnn_f32_vscaleextexp_ukernel_fn vscaleextexp,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_n = benchmark::utils::RoundUp(elements, cache_line_size_max / sizeof(float));

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1000.0f, 1000.0f), std::ref(rng));

  const size_t num_buffers = 1 +
    benchmark::utils::DivideRoundUp<size_t>(benchmark::utils::GetMaxCacheSize(), packed_n * sizeof(float));
  std::vector<float, AlignedAllocator<float, 64>> x(elements);
  std::vector<float, AlignedAllocator<float, 64>> y(packed_n * num_buffers);

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float scale[2];
    raddextexp(elements * sizeof(float), x.data(), scale);
    const float ext_mantissa = 1.0f / scale[0];
    const float ext_exponent = -scale[1];
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    vscaleextexp(elements * sizeof(float), x.data(), y.data() + packed_n * buffer_index, ext_mantissa, ext_exponent);
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
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x16,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x16,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x32,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x32,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x48,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x48,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x64,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x64,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x80,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x80,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x96,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x96,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x112,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x112,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x128,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x128,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x144,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x144,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x160,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x160,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x176,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x176,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx512f_p5_scalef_x192,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_x128_acc2,
    xnn_f32_vscaleextexp_ukernel__avx512f_p5_scalef_x192,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x8,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x8,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x16,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x16,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x24,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x24,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x32,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x32,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x40,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x40,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x48,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x48,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x56,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x56,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x64,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x64,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x72,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x72,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x80,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x80,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x88,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x88,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_vscaleextexp, avx2_p5_x96,
    xnn_f32_raddextexp_ukernel__avx2_p5_x80_acc2,
    xnn_f32_vscaleextexp_ukernel__avx2_p5_x96,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
