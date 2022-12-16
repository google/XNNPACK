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
#include <xnnpack/microparams-init.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/rmax.h>


static void f32_raddstoreexpminusmax(
  benchmark::State& state,
  xnn_f32_rmax_ukernel_fn rmax,
  xnn_f32_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax,
  xnn_init_f32_expminus_params_fn init_params,
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

  xnn_f32_expminus_params params;
  init_params(&params);

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    float x_max = nanf("");
    rmax(elements * sizeof(float), x.data(), &x_max);
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    float y_sum = nanf("");
    raddstoreexpminusmax(elements * sizeof(float), x.data(), &x_max, y.data() + buffer_index * packed_elements, &y_sum, &params);
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

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x4,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x8,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x8,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x8_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x8_acc2,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x12,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x12,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x12_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x12_acc2,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x12_acc3,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x12_acc3,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x16,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x16,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x16_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x16_acc2,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x16_acc4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x16_acc4,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x20,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x20,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x20_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x20_acc2,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_p5_x20_acc5,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_x20_acc5,
                    xnn_init_f32_expminus_neon_rr2_p5_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x4,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x8,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x8,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x8_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x8_acc2,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x12,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x12,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x12_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x12_acc2,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x12_acc3,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x12_acc3,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x16,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x16,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x16_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x16_acc2,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x16_acc4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x16_acc4,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x20,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x20,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x20_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x20_acc2,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neon_rr2_lut64_p2_x20_acc5,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_x20_acc5,
                    xnn_init_f32_expminus_neon_rr2_lut64_p2_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x4,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x8,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x8,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x8_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x8_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x12,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x12,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x12_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x12_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x12_acc3,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x12_acc3,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x16,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x16,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x16_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x16_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x16_acc4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x16_acc4,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x20,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x20,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x20_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x20_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_p5_x20_acc5,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_x20_acc5,
                    xnn_init_f32_expminus_neonfma_rr1_p5_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x4,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x8,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x8,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x8_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x8_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x12,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x12,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x12_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x12_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x12_acc3,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x12_acc3,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x16,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x16_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x16_acc4,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x16_acc4,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x20,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x20,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x20_acc2,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x20_acc2,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, neonfma_rr1_lut64_p2_x20_acc5,
                    xnn_f32_rmax_ukernel__neon,
                    xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_x20_acc5,
                    xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x128,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x128,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x128_acc2,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x128_acc2,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x128_acc4,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x128_acc4,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x144,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x144,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x144_acc3,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x144_acc3,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x160,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x160,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x160_acc2,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x160_acc2,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x160_acc5,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x160_acc5,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x192,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x192_acc2,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192_acc2,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x192_acc3,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192_acc3,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx512f_rr1_p5_scalef_x192_acc6,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_x192_acc6,
                    xnn_init_f32_expminus_avx512_rr1_p5_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x64,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x64,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x64_acc2,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x64_acc2,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x64_acc4,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x64_acc4,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x72,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x72,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x72_acc3,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x72_acc3,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x80,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x80,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x80_acc2,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x80_acc2,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x80_acc5,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x80_acc5,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x96,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x96_acc2,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96_acc2,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x96_acc3,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96_acc3,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, avx2_rr1_p5_x96_acc6,
                    xnn_f32_rmax_ukernel__avx,
                    xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_x96_acc6,
                    xnn_init_f32_expminus_avx2_rr1_p5_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x4,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x4,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x8,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x8,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x8_acc2,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x8_acc2,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x12,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x12,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x12_acc2,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x12_acc2,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x12_acc3,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x12_acc3,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x16,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x16,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x16_acc2,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x16_acc2,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x16_acc4,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x16_acc4,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x20,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x20_acc2,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20_acc2,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, sse2_rr2_p5_x20_acc5,
                    xnn_f32_rmax_ukernel__sse,
                    xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_x20_acc5,
                    xnn_init_f32_expminus_sse2_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x4,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x4,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x8,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x8,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x8_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x8_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x12,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x12,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x12_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x12_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x12_acc3,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x12_acc3,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x16,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x16,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x16_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x16_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x16_acc4,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x16_acc4,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x20,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x20_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmrelaxedsimd_rr2_p5_x20_acc5,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_x20_acc5,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x4,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x4,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x8,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x8,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x8_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x8_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x12,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x12,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x12_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x12_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x12_acc3,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x12_acc3,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x16,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x16_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x16_acc4,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x16_acc4,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x20,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x20,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x20_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x20_acc2,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, wasmsimd_rr2_p5_x20_acc5,
                    xnn_f32_rmax_ukernel__wasmsimd_arm,
                    xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_x20_acc5,
                    xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_x1,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x1,
                  xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_x2,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x2,
                  xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_x2_acc2,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x2_acc2,
                  xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_x4,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x4,
                  xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_x4_acc2,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x4_acc2,
                  xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_lut64_p2_x4_acc4,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_x4_acc4,
                  xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_x1,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x1,
                  xnn_init_f32_expminus_scalar_rr2_p5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_x2,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x2,
                  xnn_init_f32_expminus_scalar_rr2_p5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_x2_acc2,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x2_acc2,
                  xnn_init_f32_expminus_scalar_rr2_p5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_x4,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4,
                  xnn_init_f32_expminus_scalar_rr2_p5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_x4_acc2,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc2,
                  xnn_init_f32_expminus_scalar_rr2_p5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_raddstoreexpminusmax, scalar_rr2_p5_x4_acc4,
                  xnn_f32_rmax_ukernel__scalar,
                  xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_x4_acc4,
                  xnn_init_f32_expminus_scalar_rr2_p5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
