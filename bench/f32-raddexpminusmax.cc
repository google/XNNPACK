// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"
#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/raddexpminusmax.h>
#include <xnnpack/rmax.h>


static void f32_raddexpminusmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function rmax,
  xnn_f32_raddexpminusmax_ukernel_function raddexpminusmax,
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

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float x_max = nanf("");
    rmax(n * sizeof(float), x.data(), &x_max);
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    float y_sum = nanf("");
    raddexpminusmax(n * sizeof(float), x.data(), &y_sum, x_max);
  }

  state.counters["Freq"] = benchmark::utils::GetCurrentCpuFrequency();
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * n, benchmark::Counter::kIsRate);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * sizeof(float) * n, benchmark::Counter::kIsRate);
}

static void CharacteristicArguments(benchmark::internal::Benchmark* b) {
  b->ArgName("N");
  for (int32_t n = 10000; n <= 100000000; n *= 10) {
    b->Arg(n);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x64,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x64_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x64_acc4,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x64_acc4,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x72,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x72,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x72_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x72_acc3,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x80,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x80_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x80_acc5,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x80_acc5,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x96,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x96_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x96_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96_acc3,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx2_p5_x96_acc6,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx2_p5_x96_acc6,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x128,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x128_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x128_acc4,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x144,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x144,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x144_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x160,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x160,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x160_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x160_acc5,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x192,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x192_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x192_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddexpminusmax, avx512f_p5_scalef_x192_acc6,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
