// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"


static void qs8_vadd(
  benchmark::State& state,
  xnn_qs8_vadd_minmax_ukernel_fn vadd,
  xnn_init_qs8_add_minmax_params_fn init_params,
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

  std::vector<int8_t, AlignedAllocator<int8_t, 64>> a(num_elements);
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> b(num_elements);
  std::vector<int8_t, AlignedAllocator<int8_t, 64>> sum(num_elements);
  std::generate(a.begin(), a.end(), std::ref(i8rng));
  std::generate(b.begin(), b.end(), std::ref(i8rng));

  union xnn_qs8_add_minmax_params params;
  init_params(&params,
    1 /* a zero point */, 1 /* b zero point */, 1 /* output zero point */,
    0.5f /* a-output scale */, 0.75f /* b-output scale */,
    std::numeric_limits<int8_t>::min() + 1, std::numeric_limits<int8_t>::max() - 1);
  for (auto _ : state) {
    vadd(num_elements * sizeof(int8_t), a.data(), b.data(), sum.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t num_elements_per_iteration = num_elements;
  state.counters["num_elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * num_elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 3 * num_elements * sizeof(int8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qs8_vadd, neon_ld64_u8,
                    xnn_qs8_vadd_minmax_ukernel__neon_ld64_u8,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, neon_ld64_u16,
                    xnn_qs8_vadd_minmax_ukernel__neon_ld64_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, neon_ld64_u24,
                    xnn_qs8_vadd_minmax_ukernel__neon_ld64_u24,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, neon_ld64_u32,
                    xnn_qs8_vadd_minmax_ukernel__neon_ld64_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, neon_ld128_u16,
                    xnn_qs8_vadd_minmax_ukernel__neon_ld128_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, neon_ld128_u32,
                    xnn_qs8_vadd_minmax_ukernel__neon_ld128_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qs8_vadd, avx512skx_mul32_ld128_u16,
                    xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx512skx_mul32_ld128_u32,
                    xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, avx2_mul32_ld64_u8,
                    xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u8,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx2_mul32_ld64_u16,
                    xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx2_mul32_ld64_u24,
                    xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u24,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx2_mul32_ld64_u32,
                    xnn_qs8_vadd_minmax_ukernel__avx2_mul32_ld64_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, avx_mul16_ld64_u8,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u8,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx_mul16_ld64_u16,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx_mul16_ld64_u24,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u24,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx_mul16_ld64_u32,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul16_ld64_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, avx_mul32_ld32_u8,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u8,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx_mul32_ld32_u16,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx_mul32_ld32_u24,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u24,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, avx_mul32_ld32_u32,
                    xnn_qs8_vadd_minmax_ukernel__avx_mul32_ld32_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul16_ld64_u8,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u8,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul16_ld64_u16,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul16_ld64_u24,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u24,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul16_ld64_u32,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul16_ld64_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul32_ld32_u8,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u8,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul32_ld32_u16,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u16,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul32_ld32_u24,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u24,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse41_mul32_ld32_u32,
                    xnn_qs8_vadd_minmax_ukernel__sse41_mul32_ld32_u32,
                    xnn_init_qs8_add_minmax_scalar_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qs8_vadd, sse2_mul16_ld64_u8,
                    xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u8,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse2_mul16_ld64_u16,
                    xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u16,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse2_mul16_ld64_u24,
                    xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u24,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, sse2_mul16_ld64_u32,
                    xnn_qs8_vadd_minmax_ukernel__sse2_mul16_ld64_u32,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qs8_vadd, wasmsimd_u8,
                    xnn_qs8_vadd_minmax_ukernel__wasmsimd_u8,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, wasmsimd_u16,
                    xnn_qs8_vadd_minmax_ukernel__wasmsimd_u16,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, wasmsimd_u24,
                    xnn_qs8_vadd_minmax_ukernel__wasmsimd_u24,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qs8_vadd, wasmsimd_u32,
                    xnn_qs8_vadd_minmax_ukernel__wasmsimd_u32,
                    xnn_init_qs8_add_minmax_scalar_params)
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(qs8_vadd, scalar_u1,
                  xnn_qs8_vadd_minmax_ukernel__scalar_u1,
                  xnn_init_qs8_add_minmax_scalar_params)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qs8_vadd, scalar_u2,
                  xnn_qs8_vadd_minmax_ukernel__scalar_u2,
                  xnn_init_qs8_add_minmax_scalar_params)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qs8_vadd, scalar_u4,
                  xnn_qs8_vadd_minmax_ukernel__scalar_u4,
                  xnn_init_qs8_add_minmax_scalar_params)
  ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
