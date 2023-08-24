// Copyright 2022 Google LLC
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
#include <xnnpack/vlrelu.h>


static void qs8_vlrelu(
  benchmark::State& state,
  xnn_qs8_vlrelu_ukernel_fn lrelu,
  xnn_init_qs8_lrelu_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i8rng = std::bind(
    std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()),
    std::ref(rng));

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(int8_t));
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(i8rng));
  std::fill(y.begin(), y.end(), INT8_C(0xAA));

  xnn_qs8_lrelu_params params;
  init_params(&params, 0.75f /* positive scale */, 1.25f /* negative scale */, -1 /* input zero point */, 1 /* output zero point */);
  for (auto _ : state) {
    lrelu(num_elements * sizeof(int8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(int8_t) + sizeof(int8_t));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_vlrelu, neon_u8,
                    xnn_qs8_vlrelu_ukernel__neon_u8,
                    xnn_init_qs8_lrelu_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, neon_u16,
                    xnn_qs8_vlrelu_ukernel__neon_u16,
                    xnn_init_qs8_lrelu_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, neon_u32,
                    xnn_qs8_vlrelu_ukernel__neon_u32,
                    xnn_init_qs8_lrelu_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM
  BENCHMARK_CAPTURE(qs8_vlrelu, armsimd32_u4,
                    xnn_qs8_vlrelu_ukernel__armsimd32_u4,
                    xnn_init_qs8_lrelu_armsimd32_params,
                    benchmark::utils::CheckARMV6)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, armsimd32_u8,
                    xnn_qs8_vlrelu_ukernel__armsimd32_u8,
                    xnn_init_qs8_lrelu_armsimd32_params,
                    benchmark::utils::CheckARMV6)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_vlrelu, avx2_u16,
                    xnn_qs8_vlrelu_ukernel__avx2_u16,
                    xnn_init_qs8_lrelu_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, avx2_u32,
                    xnn_qs8_vlrelu_ukernel__avx2_u32,
                    xnn_init_qs8_lrelu_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, avx2_u64,
                    xnn_qs8_vlrelu_ukernel__avx2_u64,
                    xnn_init_qs8_lrelu_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vlrelu, avx_u8,
                    xnn_qs8_vlrelu_ukernel__avx_u8,
                    xnn_init_qs8_lrelu_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, avx_u16,
                    xnn_qs8_vlrelu_ukernel__avx_u16,
                    xnn_init_qs8_lrelu_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, avx_u32,
                    xnn_qs8_vlrelu_ukernel__avx_u32,
                    xnn_init_qs8_lrelu_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vlrelu, sse41_u8,
                    xnn_qs8_vlrelu_ukernel__sse41_u8,
                    xnn_init_qs8_lrelu_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, sse41_u16,
                    xnn_qs8_vlrelu_ukernel__sse41_u16,
                    xnn_init_qs8_lrelu_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, sse41_u32,
                    xnn_qs8_vlrelu_ukernel__sse41_u32,
                    xnn_init_qs8_lrelu_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vlrelu, ssse3_u16,
                    xnn_qs8_vlrelu_ukernel__ssse3_u16,
                    xnn_init_qs8_lrelu_sse2_params,
                    benchmark::utils::CheckSSSE3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, ssse3_u32,
                    xnn_qs8_vlrelu_ukernel__ssse3_u32,
                    xnn_init_qs8_lrelu_sse2_params,
                    benchmark::utils::CheckSSSE3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vlrelu, sse2_u16,
                    xnn_qs8_vlrelu_ukernel__sse2_u16,
                    xnn_init_qs8_lrelu_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, sse2_u32,
                    xnn_qs8_vlrelu_ukernel__sse2_u32,
                    xnn_init_qs8_lrelu_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmrelaxedsimd_arm_u16,
                    xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u16,
                    xnn_init_qs8_lrelu_wasmsimd_arm_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmrelaxedsimd_arm_u32,
                    xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_arm_u32,
                    xnn_init_qs8_lrelu_wasmsimd_arm_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vlrelu, wasmrelaxedsimd_x86_u8,
                    xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u8,
                    xnn_init_qs8_lrelu_wasmsimd_x86_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmrelaxedsimd_x86_u16,
                    xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u16,
                    xnn_init_qs8_lrelu_wasmsimd_x86_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmrelaxedsimd_x86_u32,
                    xnn_qs8_vlrelu_ukernel__wasmrelaxedsimd_x86_u32,
                    xnn_init_qs8_lrelu_wasmsimd_x86_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmsimd_arm_u16,
                    xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u16,
                    xnn_init_qs8_lrelu_wasmsimd_arm_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmsimd_arm_u32,
                    xnn_qs8_vlrelu_ukernel__wasmsimd_arm_u32,
                    xnn_init_qs8_lrelu_wasmsimd_arm_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vlrelu, wasmsimd_x86_u8,
                    xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u8,
                    xnn_init_qs8_lrelu_wasmsimd_x86_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmsimd_x86_u16,
                    xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u16,
                    xnn_init_qs8_lrelu_wasmsimd_x86_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vlrelu, wasmsimd_x86_u32,
                    xnn_qs8_vlrelu_ukernel__wasmsimd_x86_u32,
                    xnn_init_qs8_lrelu_wasmsimd_x86_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(qs8_vlrelu, scalar_andxor_u1,
                  xnn_qs8_vlrelu_ukernel__scalar_andxor_u1,
                  xnn_init_qs8_lrelu_scalar_andxor_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qs8_vlrelu, scalar_andxor_u2,
                  xnn_qs8_vlrelu_ukernel__scalar_andxor_u2,
                  xnn_init_qs8_lrelu_scalar_andxor_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qs8_vlrelu, scalar_andxor_u4,
                  xnn_qs8_vlrelu_ukernel__scalar_andxor_u4,
                  xnn_init_qs8_lrelu_scalar_andxor_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(qs8_vlrelu, scalar_select_u1,
                  xnn_qs8_vlrelu_ukernel__scalar_select_u1,
                  xnn_init_qs8_lrelu_scalar_select_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qs8_vlrelu, scalar_select_u2,
                  xnn_qs8_vlrelu_ukernel__scalar_select_u2,
                  xnn_init_qs8_lrelu_scalar_select_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qs8_vlrelu, scalar_select_u4,
                  xnn_qs8_vlrelu_ukernel__scalar_select_u4,
                  xnn_init_qs8_lrelu_scalar_select_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
