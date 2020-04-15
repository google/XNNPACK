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
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>


static void f32_raddstoreexpminusmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_function rmax,
  xnn_f32_raddstoreexpminusmax_ukernel_function raddstoreexpminusmax,
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
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    float y_sum = nanf("");
    raddstoreexpminusmax(n * sizeof(float), x.data(), y.data() + buffer_index * packed_n, &y_sum, x_max);
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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x4,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x8,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x8_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x12,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x12_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x12_acc3,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc3,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x16,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x16_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x16_acc4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc4,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x20,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x20_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_p5_x20_acc5,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc5,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x4,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x8,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x8_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x12,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x12_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x12_acc3,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc3,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x16,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x16_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x16_acc4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc4,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x20,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x20_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc2,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_lut64_p2_x20_acc5,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc5,
    benchmark::utils::CheckNEON)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x4,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x8,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x8_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x12,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x12_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x12_acc3,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc3,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x16,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x16_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x16_acc4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc4,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x20,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x20_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_p5_x20_acc5,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc5,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x4,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x8,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x8_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x12,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x12_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x12_acc3,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc3,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x16,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x16_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x16_acc4,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc4,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x20,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x20_acc2,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc2,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_lut64_p2_x20_acc5,
    xnn_f32_rmax_ukernel__neon,
    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc5,
    benchmark::utils::CheckNEONFMA)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x128,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x128_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x128_acc4,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x144,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x144_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x160,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x160_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x160_acc5,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x192,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x192_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x192_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_p5_scalef_x192_acc6,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6,
    benchmark::utils::CheckAVX512F)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x64,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x64_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x64_acc4,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc4,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x72,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x72_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72_acc3,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x80,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x80_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x80_acc5,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc5,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x96,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x96_acc2,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc2,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x96_acc3,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc3,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_p5_x96_acc6,
    xnn_f32_rmax_ukernel__avx,
    xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc6,
    benchmark::utils::CheckAVX2)->Apply(CharacteristicArguments)->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x4,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x4)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x8,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x8_acc2,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x12,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x12_acc2,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x12_acc3,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc3)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x16,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x16_acc2,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x16_acc4,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc4)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x20,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x20_acc2,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_p5_x20_acc5,
    xnn_f32_rmax_ukernel__sse,
    xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc5)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x4,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x4)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x8,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x8_acc2,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x12,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x12_acc2,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x12_acc3,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc3)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x16,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x16_acc2,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x16_acc4,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc4)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x20,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x20_acc2,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc2)->Apply(CharacteristicArguments)->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, psimd_p5_x20_acc5,
    xnn_f32_rmax_ukernel__psimd,
    xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc5)->Apply(CharacteristicArguments)->UseRealTime();
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC

BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_lut64_p2_x1,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x1)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_lut64_p2_x2,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_lut64_p2_x2_acc2,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2_acc2)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_lut64_p2_x4,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_lut64_p2_x4_acc2,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc2)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_lut64_p2_x4_acc4,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc4)->Apply(CharacteristicArguments)->UseRealTime();

BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_p5_x1,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x1)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_p5_x2,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_p5_x2_acc2,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2_acc2)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_p5_x4,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_p5_x4_acc2,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc2)->Apply(CharacteristicArguments)->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_p5_x4_acc4,
  xnn_f32_rmax_ukernel__scalar,
  xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc4)->Apply(CharacteristicArguments)->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
