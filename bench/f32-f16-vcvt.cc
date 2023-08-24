// Copyright 2021 Google LLC
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
#include <xnnpack/vcvt.h>


static void f32_f16_vcvt(
  benchmark::State& state,
  xnn_f32_f16_vcvt_ukernel_fn cvt,
  xnn_init_f32_f16_cvt_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(float));
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), UINT16_C(0x7E00));

  xnn_f32_f16_cvt_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint16_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint16_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_f16_vcvt, neonfp16_u8,
                    xnn_f32_f16_vcvt_ukernel__neonfp16_u8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, neonfp16_u16,
                    xnn_f32_f16_vcvt_ukernel__neonfp16_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u8,
                    xnn_f32_f16_vcvt_ukernel__neon_u8,
                    xnn_init_f32_f16_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u16,
                    xnn_f32_f16_vcvt_ukernel__neon_u16,
                    xnn_init_f32_f16_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u24,
                    xnn_f32_f16_vcvt_ukernel__neon_u24,
                    xnn_init_f32_f16_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, neon_u32,
                    xnn_f32_f16_vcvt_ukernel__neon_u32,
                    xnn_init_f32_f16_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx512skx_u16,
                    xnn_f32_f16_vcvt_ukernel__avx512skx_u16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx512skx_u32,
                    xnn_f32_f16_vcvt_ukernel__avx512skx_u32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_f16_vcvt, f16c_u8,
                    xnn_f32_f16_vcvt_ukernel__f16c_u8,
                    xnn_init_f32_f16_cvt_f16c_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, f16c_u16,
                    xnn_f32_f16_vcvt_ukernel__f16c_u16,
                    xnn_init_f32_f16_cvt_f16c_params,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u8,
                    xnn_f32_f16_vcvt_ukernel__avx_u8,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u16,
                    xnn_f32_f16_vcvt_ukernel__avx_u16,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u24,
                    xnn_f32_f16_vcvt_ukernel__avx_u24,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, avx_u32,
                    xnn_f32_f16_vcvt_ukernel__avx_u32,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u8,
                    xnn_f32_f16_vcvt_ukernel__sse41_u8,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u16,
                    xnn_f32_f16_vcvt_ukernel__sse41_u16,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u24,
                    xnn_f32_f16_vcvt_ukernel__sse41_u24,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse41_u32,
                    xnn_f32_f16_vcvt_ukernel__sse41_u32,
                    xnn_init_f32_f16_cvt_sse2_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u8,
                    xnn_f32_f16_vcvt_ukernel__sse2_u8,
                    xnn_init_f32_f16_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u16,
                    xnn_f32_f16_vcvt_ukernel__sse2_u16,
                    xnn_init_f32_f16_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u24,
                    xnn_f32_f16_vcvt_ukernel__sse2_u24,
                    xnn_init_f32_f16_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, sse2_u32,
                    xnn_f32_f16_vcvt_ukernel__sse2_u32,
                    xnn_init_f32_f16_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u8,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u8,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u16,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u16,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u24,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u24,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmrelaxedsimd_u32,
                    xnn_f32_f16_vcvt_ukernel__wasmrelaxedsimd_u32,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u8,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u8,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u16,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u16,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u24,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u24,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_f16_vcvt, wasmsimd_u32,
                    xnn_f32_f16_vcvt_ukernel__wasmsimd_u32,
                    xnn_init_f32_f16_cvt_wasmsimd_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u1,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u1,
                  xnn_init_f32_f16_cvt_scalar_bitcast_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u2,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u2,
                  xnn_init_f32_f16_cvt_scalar_bitcast_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u3,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u3,
                  xnn_init_f32_f16_cvt_scalar_bitcast_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_bitcast_u4,
                  xnn_f32_f16_vcvt_ukernel__scalar_bitcast_u4,
                  xnn_init_f32_f16_cvt_scalar_bitcast_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u1,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u1,
                  xnn_init_f32_f16_cvt_scalar_fabsf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u2,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u2,
                  xnn_init_f32_f16_cvt_scalar_fabsf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u3,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u3,
                  xnn_init_f32_f16_cvt_scalar_fabsf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_f16_vcvt, scalar_fabsf_u4,
                  xnn_f32_f16_vcvt_ukernel__scalar_fabsf_u4,
                  xnn_init_f32_f16_cvt_scalar_fabsf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
