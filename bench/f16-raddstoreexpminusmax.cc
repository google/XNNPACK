// Copyright 2022-2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <random>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/raddstoreexpminusmax.h"
#include "src/xnnpack/reduce.h"
#include "test/replicable_random_device.h"
#include <benchmark/benchmark.h>

static void f16_raddstoreexpminusmax(
    benchmark::State& state, xnn_f16_rmax_ukernel_fn rmax,
    xnn_f16_raddstoreexpminusmax_ukernel_fn raddstoreexpminusmax,
    xnn_init_f16_expminus_params_fn init_params,
    uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t elements = state.range(0);
  const size_t cache_line_size_max = 128;
  const size_t packed_elements = benchmark::utils::RoundUp(
      elements, cache_line_size_max / sizeof(xnn_float16));

  xnnpack::ReplicableRandomDevice rng;
  auto f32rng = std::bind(
      std::uniform_real_distribution<float>(-100.0f, 100.0f), std::ref(rng));

  const size_t num_buffers = 1 + benchmark::utils::DivideRoundUp<size_t>(
                                     benchmark::utils::GetMaxCacheSize(),
                                     packed_elements * sizeof(xnn_float16));
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> x(elements);
  xnnpack::Buffer<xnn_float16, XNN_ALLOCATION_ALIGNMENT> y(packed_elements *
                                                           num_buffers);

  std::generate(x.begin(), x.end(), f32rng);

  benchmark::utils::DisableDenormals();

  size_t buffer_index = 0;
  for (auto _ : state) {
    state.PauseTiming();
    xnn_float16 x_max;
    rmax(elements * sizeof(xnn_float16), x.data(), &x_max, /*params=*/nullptr);
    if (++buffer_index == num_buffers) {
      buffer_index = 0;
    }
    state.ResumeTiming();

    xnn_float16 y_sum;
    raddstoreexpminusmax(elements * sizeof(xnn_float16), x.data(), &x_max,
                         y.data() + buffer_index * packed_elements, &y_sum,
                         nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * elements * sizeof(xnn_float16);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
BENCHMARK_CAPTURE(
    f16_raddstoreexpminusmax, neonfp16arith_rr2_p2_u16,
    xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4,
    xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u16, nullptr,
    xnn_arch_arm_neon_fp16_arith)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(
    f16_raddstoreexpminusmax, neonfp16arith_rr2_p2_u16_acc2,
    xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4,
    xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u16_acc2,
    nullptr, xnn_arch_arm_neon_fp16_arith)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(
    f16_raddstoreexpminusmax, neonfp16arith_rr2_p2_u32,
    xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4,
    xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32, nullptr,
    xnn_arch_arm_neon_fp16_arith)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(
    f16_raddstoreexpminusmax, neonfp16arith_rr2_p2_u32_acc2,
    xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4,
    xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2,
    nullptr, xnn_arch_arm_neon_fp16_arith)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(
    f16_raddstoreexpminusmax, neonfp16arith_rr2_p2_u32_acc4,
    xnn_f16_rmax_ukernel__neonfp16arith_u32_acc4,
    xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc4,
    nullptr, xnn_arch_arm_neon_fp16_arith)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ENABLE_AVX2 && XNN_ENABLE_F16C && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u16,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u16_acc2,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16_acc2,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u32,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u32_acc2,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc2,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u32_acc4,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc4,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u40,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u40_acc2,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc2,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u40_acc5,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc5,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u48,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u48_acc2,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc2,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u48_acc3,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc3,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u64,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u64_acc2,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc2,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
BENCHMARK_CAPTURE(f16_raddstoreexpminusmax, avx2_rr1_p2_u64_acc4,
                  xnn_f16_rmax_ukernel__f16c_u32,
                  xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc4,
                  nullptr, xnn_arch_x86_avx2)
    ->Apply(
        benchmark::utils::UnaryElementwiseParameters<xnn_float16, xnn_float16>)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
