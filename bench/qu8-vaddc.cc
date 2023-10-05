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

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vbinary.h>


static void qu8_vaddc(
  benchmark::State& state,
  xnn_qu8_vadd_minmax_ukernel_fn vaddc,
  xnn_init_qu8_add_minmax_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> a(num_elements);
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> sum(num_elements);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  const uint8_t b = u8rng();

  union xnn_qu8_add_minmax_params params;
  init_params(&params,
    127 /* a zero point */, 127 /* b zero point */, 127 /* output zero point */,
    0.5f /* a-output scale */, 0.75f /* b-output scale */,
    std::numeric_limits<uint8_t>::min() + 1, std::numeric_limits<uint8_t>::max() - 1);
  for (auto _ : state) {
    vaddc(num_elements * sizeof(uint8_t), a.data(), &b, sum.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t num_elements_per_iteration = num_elements;
  state.counters["num_elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * num_elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(int8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(qu8_vaddc, neon_ld64_u8,
                    xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u8,
                    xnn_init_qu8_add_minmax_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, neon_ld64_u16,
                    xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u16,
                    xnn_init_qu8_add_minmax_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, neon_ld64_u32,
                    xnn_qu8_vaddc_minmax_ukernel__neon_ld64_u32,
                    xnn_init_qu8_add_minmax_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, neon_ld128_u16,
                    xnn_qu8_vaddc_minmax_ukernel__neon_ld128_u16,
                    xnn_init_qu8_add_minmax_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(qu8_vaddc, avx512skx_mul32_ld128_u16,
                    xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u16,
                    xnn_init_qu8_add_minmax_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, avx512skx_mul32_ld128_u32,
                    xnn_qu8_vaddc_minmax_ukernel__avx512skx_mul32_ld128_u32,
                    xnn_init_qu8_add_minmax_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, avx2_mul32_ld64_u8,
                    xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_u8,
                    xnn_init_qu8_add_minmax_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, avx2_mul32_ld64_u16,
                    xnn_qu8_vaddc_minmax_ukernel__avx2_mul32_ld64_u16,
                    xnn_init_qu8_add_minmax_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, xop_mul32_ld32_u8,
                    xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_u8,
                    xnn_init_qu8_add_minmax_sse4_params,
                    benchmark::utils::CheckXOP)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, xop_mul32_ld32_u16,
                    xnn_qu8_vaddc_minmax_ukernel__xop_mul32_ld32_u16,
                    xnn_init_qu8_add_minmax_sse4_params,
                    benchmark::utils::CheckXOP)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, avx_mul16_ld64_u8,
                    xnn_qu8_vaddc_minmax_ukernel__avx_mul16_ld64_u8,
                    xnn_init_qu8_add_minmax_sse2_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, avx_mul16_ld64_u16,
                    xnn_qu8_vaddc_minmax_ukernel__avx_mul16_ld64_u16,
                    xnn_init_qu8_add_minmax_sse2_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, avx_mul32_ld32_u8,
                    xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_u8,
                    xnn_init_qu8_add_minmax_sse4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, avx_mul32_ld32_u16,
                    xnn_qu8_vaddc_minmax_ukernel__avx_mul32_ld32_u16,
                    xnn_init_qu8_add_minmax_sse4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, sse41_mul16_ld64_u8,
                    xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u8,
                    xnn_init_qu8_add_minmax_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, sse41_mul16_ld64_u16,
                    xnn_qu8_vaddc_minmax_ukernel__sse41_mul16_ld64_u16,
                    xnn_init_qu8_add_minmax_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, sse41_mul32_ld32_u8,
                    xnn_qu8_vaddc_minmax_ukernel__sse41_mul32_ld32_u8,
                    xnn_init_qu8_add_minmax_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, sse41_mul32_ld32_u16,
                    xnn_qu8_vaddc_minmax_ukernel__sse41_mul32_ld32_u16,
                    xnn_init_qu8_add_minmax_sse4_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(qu8_vaddc, sse2_mul16_ld64_u8,
                    xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u8,
                    xnn_init_qu8_add_minmax_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, sse2_mul16_ld64_u16,
                    xnn_qu8_vaddc_minmax_ukernel__sse2_mul16_ld64_u16,
                    xnn_init_qu8_add_minmax_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(qu8_vaddc, wasmsimd_u8,
                    xnn_qu8_vaddc_minmax_ukernel__wasmsimd_u8,
                    xnn_init_qu8_add_minmax_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(qu8_vaddc, wasmsimd_u16,
                    xnn_qu8_vaddc_minmax_ukernel__wasmsimd_u16,
                    xnn_init_qu8_add_minmax_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(qu8_vaddc, scalar_u1,
                  xnn_qu8_vaddc_minmax_ukernel__scalar_u1,
                  xnn_init_qu8_add_minmax_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qu8_vaddc, scalar_u2,
                  xnn_qu8_vaddc_minmax_ukernel__scalar_u2,
                  xnn_init_qu8_add_minmax_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(qu8_vaddc, scalar_u4,
                  xnn_qu8_vaddc_minmax_ukernel__scalar_u4,
                  xnn_init_qu8_add_minmax_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
