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
#include <fp16/fp16.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vcvt.h>


static void f16_f32_vcvt(
  benchmark::State& state,
  xnn_f16_f32_vcvt_ukernel_fn cvt,
  xnn_init_f16_f32_cvt_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));
  auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> x(num_elements + XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f16rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  xnn_f16_f32_cvt_params params;
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
  BENCHMARK_CAPTURE(f16_f32_vcvt, neonfp16_x8,
                    xnn_f16_f32_vcvt_ukernel__neonfp16_x8,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neonfp16_x16,
                    xnn_f16_f32_vcvt_ukernel__neonfp16_x16,
                    nullptr /* init params */,
                    benchmark::utils::CheckNEONFP16)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_x8,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_x8,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_x16,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_x16,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_x24,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_x24,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int16_x32,
                    xnn_f16_f32_vcvt_ukernel__neon_int16_x32,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_x8,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_x8,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_x16,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_x16,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_x24,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_x24,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, neon_int32_x32,
                    xnn_f16_f32_vcvt_ukernel__neon_int32_x32,
                    xnn_init_f16_f32_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx512skx_x16,
                    xnn_f16_f32_vcvt_ukernel__avx512skx_x16,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx512skx_x32,
                    xnn_f16_f32_vcvt_ukernel__avx512skx_x32,
                    nullptr /* init params */,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, f16c_x8,
                    xnn_f16_f32_vcvt_ukernel__f16c_x8,
                    nullptr /* init params */,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, f16c_x16,
                    xnn_f16_f32_vcvt_ukernel__f16c_x16,
                    nullptr /* init params */,
                    benchmark::utils::CheckF16C)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_x8,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_x8,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_x16,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_x16,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_x24,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_x24,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int16_x32,
                    xnn_f16_f32_vcvt_ukernel__avx_int16_x32,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_x8,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_x8,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_x16,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_x16,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_x24,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_x24,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, avx_int32_x32,
                    xnn_f16_f32_vcvt_ukernel__avx_int32_x32,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_x8,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_x8,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_x16,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_x16,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_x24,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_x24,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int16_x32,
                    xnn_f16_f32_vcvt_ukernel__sse41_int16_x32,
                    xnn_init_f16_f32_cvt_sse_int16_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_x8,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_x8,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_x16,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_x16,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_x24,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_x24,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse41_int32_x32,
                    xnn_f16_f32_vcvt_ukernel__sse41_int32_x32,
                    xnn_init_f16_f32_cvt_sse_int32_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_x8,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_x8,
                    xnn_init_f16_f32_cvt_sse_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_x16,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_x16,
                    xnn_init_f16_f32_cvt_sse_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_x24,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_x24,
                    xnn_init_f16_f32_cvt_sse_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int16_x32,
                    xnn_f16_f32_vcvt_ukernel__sse2_int16_x32,
                    xnn_init_f16_f32_cvt_sse_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_x8,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_x8,
                    xnn_init_f16_f32_cvt_sse_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_x16,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_x16,
                    xnn_init_f16_f32_cvt_sse_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_x24,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_x24,
                    xnn_init_f16_f32_cvt_sse_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, sse2_int32_x32,
                    xnn_f16_f32_vcvt_ukernel__sse2_int32_x32,
                    xnn_init_f16_f32_cvt_sse_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_x8,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x8,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_x16,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x16,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_x24,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x24,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int16_x32,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int16_x32,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_x8,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x8,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_x16,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x16,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_x24,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x24,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmrelaxedsimd_int32_x32,
                    xnn_f16_f32_vcvt_ukernel__wasmrelaxedsimd_int32_x32,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_x8,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x8,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_x16,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x16,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_x24,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x24,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int16_x32,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int16_x32,
                    xnn_init_f16_f32_cvt_wasmsimd_int16_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_x8,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x8,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_x16,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x16,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_x24,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x24,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f16_f32_vcvt, wasmsimd_int32_x32,
                    xnn_f16_f32_vcvt_ukernel__wasmsimd_int32_x32,
                    xnn_init_f16_f32_cvt_wasmsimd_int32_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_x1,
                  xnn_f16_f32_vcvt_ukernel__scalar_x1,
                  xnn_init_f16_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_x2,
                  xnn_f16_f32_vcvt_ukernel__scalar_x2,
                  xnn_init_f16_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_x3,
                  xnn_f16_f32_vcvt_ukernel__scalar_x3,
                  xnn_init_f16_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f16_f32_vcvt, scalar_x4,
                  xnn_f16_f32_vcvt_ukernel__scalar_x4,
                  xnn_init_f16_f32_cvt_scalar_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<uint16_t, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
