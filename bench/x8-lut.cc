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

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/lut.h"
#include "src/xnnpack/microfnptr.h"
#include <benchmark/benchmark.h>

static void x8_lut(benchmark::State& state, xnn_x8_lut_ukernel_fn lut,
                   uint64_t arch_flags = 0) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> input(num_elements);
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> output(num_elements);
  xnnpack::Buffer<uint8_t, XNN_ALLOCATION_ALIGNMENT> table(256);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);
  xnnpack::fill_uniform_random_bits(table.data(), table.size(), rng);

  for (auto _ : state) {
    lut(num_elements * sizeof(uint8_t), input.data(), output.data(),
        table.data());
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(uint8_t);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
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

#if XNN_ENABLE_AVX512VBMI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u64,
                  xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u64,
                  xnn_arch_x86_avx512vbmi)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u128,
                  xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u128,
                  xnn_arch_x86_avx512vbmi)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u192,
                  xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u192,
                  xnn_arch_x86_avx512vbmi)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx512vbmi_vpermx2b_u256,
                  xnn_x8_lut_ukernel__avx512vbmi_vpermx2b_u256,
                  xnn_arch_x86_avx512vbmi)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512VBMI && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u64,
                  xnn_x8_lut_ukernel__avx512skx_vpshufb_u64,
                  xnn_arch_x86_avx512skx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u128,
                  xnn_x8_lut_ukernel__avx512skx_vpshufb_u128,
                  xnn_arch_x86_avx512skx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u192,
                  xnn_x8_lut_ukernel__avx512skx_vpshufb_u192,
                  xnn_arch_x86_avx512skx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx512skx_vpshufb_u256,
                  xnn_x8_lut_ukernel__avx512skx_vpshufb_u256,
                  xnn_arch_x86_avx512skx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ENABLE_AVX512SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
BENCHMARK_CAPTURE(x8_lut, avx2_u32, xnn_x8_lut_ukernel__avx2_u32,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx2_u64, xnn_x8_lut_ukernel__avx2_u64,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx2_u96, xnn_x8_lut_ukernel__avx2_u96,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx2_u128, xnn_x8_lut_ukernel__avx2_u128,
                  xnn_arch_x86_avx2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

BENCHMARK_CAPTURE(x8_lut, avx_u16, xnn_x8_lut_ukernel__avx_u16,
                  xnn_arch_x86_avx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx_u32, xnn_x8_lut_ukernel__avx_u32,
                  xnn_arch_x86_avx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx_u48, xnn_x8_lut_ukernel__avx_u48,
                  xnn_arch_x86_avx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, avx_u64, xnn_x8_lut_ukernel__avx_u64,
                  xnn_arch_x86_avx)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

BENCHMARK_CAPTURE(x8_lut, ssse3_u16, xnn_x8_lut_ukernel__ssse3_u16,
                  xnn_arch_x86_ssse3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, ssse3_u32, xnn_x8_lut_ukernel__ssse3_u32,
                  xnn_arch_x86_ssse3)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u16, xnn_x8_lut_ukernel__wasmpshufb_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u32, xnn_x8_lut_ukernel__wasmpshufb_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u48, xnn_x8_lut_ukernel__wasmpshufb_u48)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, wasmpshufb_u64, xnn_x8_lut_ukernel__wasmpshufb_u64)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
BENCHMARK_CAPTURE(x8_lut, wasmsimd_u16, xnn_x8_lut_ukernel__wasmsimd_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, wasmsimd_u32, xnn_x8_lut_ukernel__wasmsimd_u32)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, wasmsimd_u48, xnn_x8_lut_ukernel__wasmsimd_u48)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, wasmsimd_u64, xnn_x8_lut_ukernel__wasmsimd_u64)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(x8_lut, scalar_u1, xnn_x8_lut_ukernel__scalar_u1)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u2, xnn_x8_lut_ukernel__scalar_u2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u4, xnn_x8_lut_ukernel__scalar_u4)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u8, xnn_x8_lut_ukernel__scalar_u8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();
BENCHMARK_CAPTURE(x8_lut, scalar_u16, xnn_x8_lut_ukernel__scalar_u16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint8_t, uint8_t>)
    ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
