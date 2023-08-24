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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/raddextexp.h>


static void f32_raddextexp(
  benchmark::State& state,
  xnn_f32_raddextexp_ukernel_fn raddextexp,
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

  std::generate(x.begin(), x.end(), std::ref(f32rng));

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }

    float y_sum[2] = { nanf(""), nanf("") };
    raddextexp(elements * sizeof(float), x.data(), y_sum);
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
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u128,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u128_acc2,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u128_acc4,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u128_acc4,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u144,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u144_acc3,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u144_acc3,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u160,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u160_acc2,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u160_acc5,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u160_acc5,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u192,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u192_acc2,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u192_acc3,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc3,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx512f_p5_scalef_u192_acc6,
    xnn_f32_raddextexp_ukernel__avx512f_p5_scalef_u192_acc6,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u64,
    xnn_f32_raddextexp_ukernel__avx2_p5_u64,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u64_acc2,
    xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u64_acc4,
    xnn_f32_raddextexp_ukernel__avx2_p5_u64_acc4,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u72,
    xnn_f32_raddextexp_ukernel__avx2_p5_u72,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u72_acc3,
    xnn_f32_raddextexp_ukernel__avx2_p5_u72_acc3,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u80,
    xnn_f32_raddextexp_ukernel__avx2_p5_u80,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u80_acc2,
    xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u80_acc5,
    xnn_f32_raddextexp_ukernel__avx2_p5_u80_acc5,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u96,
    xnn_f32_raddextexp_ukernel__avx2_p5_u96,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u96_acc2,
    xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u96_acc3,
    xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc3,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddextexp, avx2_p5_u96_acc6,
    xnn_f32_raddextexp_ukernel__avx2_p5_u96_acc6,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
