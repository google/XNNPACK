// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/lut.h"
#include "xnnpack/microfnptr.h"


static void x8_lut(
  benchmark::State& state,
  xnn_x8_lut_ukernel_fn lut,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> input(num_elements);
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> output(num_elements);
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> table(256);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));
  std::generate(input.begin(), input.end(), std::ref(u8rng));
  std::generate(table.begin(), table.end(), std::ref(u8rng));
  std::fill(output.begin(), output.end(), UINT8_C(0xAA));

  for (auto _ : state) {
    lut(num_elements * sizeof(uint8_t), input.data(), output.data(), table.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(uint8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(x8_lut, aarch64_neon_tbx128x4_u16,
                    xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, aarch64_neon_tbx128x4_u32,
                    xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, aarch64_neon_tbx128x4_u48,
                    xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u48)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, aarch64_neon_tbx128x4_u64,
                    xnn_x8_lut_ukernel__aarch64_neon_tbx128x4_u64)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u64,
                    xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u64,
                    benchmark::utils::CheckAVX512VBMI)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u128,
                    xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u128,
                    benchmark::utils::CheckAVX512VBMI)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u192,
                    xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u192,
                    benchmark::utils::CheckAVX512VBMI)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u256,
                    xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u256,
                    benchmark::utils::CheckAVX512VBMI)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u64,
                    xnn_x8_lut_ukernel__avx512skx_vpshufb_u64,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u128,
                    xnn_x8_lut_ukernel__avx512skx_vpshufb_u128,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u192,
                    xnn_x8_lut_ukernel__avx512skx_vpshufb_u192,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u256,
                    xnn_x8_lut_ukernel__avx512skx_vpshufb_u256,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(x8_lut, avx2_u32,
                    xnn_x8_lut_ukernel__avx2_u32,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx2_u64,
                    xnn_x8_lut_ukernel__avx2_u64,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx2_u96,
                    xnn_x8_lut_ukernel__avx2_u96,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx2_u128,
                    xnn_x8_lut_ukernel__avx2_u128,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(x8_lut, avx_u16,
                    xnn_x8_lut_ukernel__avx_u16,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx_u32,
                    xnn_x8_lut_ukernel__avx_u32,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx_u48,
                    xnn_x8_lut_ukernel__avx_u48,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, avx_u64,
                    xnn_x8_lut_ukernel__avx_u64,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(x8_lut, ssse3_u16,
                    xnn_x8_lut_ukernel__ssse3_u16,
                    benchmark::utils::CheckSSSE3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, ssse3_u32,
                    xnn_x8_lut_ukernel__ssse3_u32,
                    benchmark::utils::CheckSSSE3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u16,
                    xnn_x8_lut_ukernel__wasmpshufb_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u32,
                    xnn_x8_lut_ukernel__wasmpshufb_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u48,
                    xnn_x8_lut_ukernel__wasmpshufb_u48)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u64,
                    xnn_x8_lut_ukernel__wasmpshufb_u64)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(x8_lut, wasmsimd_u16,
                    xnn_x8_lut_ukernel__wasmsimd_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, wasmsimd_u32,
                    xnn_x8_lut_ukernel__wasmsimd_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, wasmsimd_u48,
                    xnn_x8_lut_ukernel__wasmsimd_u48)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(x8_lut, wasmsimd_u64,
                    xnn_x8_lut_ukernel__wasmsimd_u64)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(x8_lut, scalar_u1,
                  xnn_x8_lut_ukernel__scalar_u1)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u2,
                  xnn_x8_lut_ukernel__scalar_u2)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u4,
                  xnn_x8_lut_ukernel__scalar_u4)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u8,
                  xnn_x8_lut_ukernel__scalar_u8)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u16,
                  xnn_x8_lut_ukernel__scalar_u16)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
