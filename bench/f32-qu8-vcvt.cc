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


static void f32_qu8_vcvt(
  benchmark::State& state,
  xnn_f32_qu8_vcvt_ukernel_fn cvt,
  xnn_init_f32_qu8_cvt_params_fn init_params,
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
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), UINT8_C(0xA5));

  xnn_f32_qu8_cvt_params params;
  init_params(&params,
    25.0f /* scale */,
    127 /* output zero point */,
    std::numeric_limits<uint8_t>::min() + 1 /* output min */,
    std::numeric_limits<uint8_t>::max() - 1 /* output max */);
  for (auto _ : state) {
    cvt(num_elements * sizeof(uint8_t), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = num_elements * (sizeof(uint8_t) + sizeof(float));
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x8,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x8,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x16,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x16,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x24,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x24,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neonv8_x32,
                    xnn_f32_qu8_vcvt_ukernel__neonv8_x32,
                    xnn_init_f32_qu8_cvt_neonv8_params,
                    benchmark::utils::CheckNEONV8)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x8,
                    xnn_f32_qu8_vcvt_ukernel__neon_x8,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x16,
                    xnn_f32_qu8_vcvt_ukernel__neon_x16,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x24,
                    xnn_f32_qu8_vcvt_ukernel__neon_x24,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, neon_x32,
                    xnn_f32_qu8_vcvt_ukernel__neon_x32,
                    xnn_init_f32_qu8_cvt_neon_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_x32,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_x32,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_x64,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_x64,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_x96,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_x96,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx512skx_x128,
                    xnn_f32_qu8_vcvt_ukernel__avx512skx_x128,
                    xnn_init_f32_qu8_cvt_avx512_params,
                    benchmark::utils::CheckAVX512SKX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, int8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_x16,
                    xnn_f32_qu8_vcvt_ukernel__avx2_x16,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_x32,
                    xnn_f32_qu8_vcvt_ukernel__avx2_x32,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_x48,
                    xnn_f32_qu8_vcvt_ukernel__avx2_x48,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx2_x64,
                    xnn_f32_qu8_vcvt_ukernel__avx2_x64,
                    xnn_init_f32_qu8_cvt_avx2_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x8,
                    xnn_f32_qu8_vcvt_ukernel__avx_x8,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x16,
                    xnn_f32_qu8_vcvt_ukernel__avx_x16,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x24,
                    xnn_f32_qu8_vcvt_ukernel__avx_x24,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, avx_x32,
                    xnn_f32_qu8_vcvt_ukernel__avx_x32,
                    xnn_init_f32_qu8_cvt_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x8,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x8,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x16,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x16,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x24,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x24,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, sse2_x32,
                    xnn_f32_qu8_vcvt_ukernel__sse2_x32,
                    xnn_init_f32_qu8_cvt_sse2_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x8,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x8,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x16,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x16,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x24,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x24,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_cvt_x32,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_cvt_x32,
                    xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x8,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x8,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x16,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x16,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x24,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x24,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasmsimd_magic_x32,
                    xnn_f32_qu8_vcvt_ukernel__wasmsimd_magic_x32,
                    xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_x1,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x1,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_x2,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x2,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_x3,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x3,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_qu8_vcvt, wasm_fmagic_x4,
                    xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x4,
                    xnn_init_f32_qu8_cvt_scalar_fmagic_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_x1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_x1,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_x2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_x2,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_x3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_x3,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_fmagic_x4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_x4,
                  xnn_init_f32_qu8_cvt_scalar_fmagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_x1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x1,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_x2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x2,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_x3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x3,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_imagic_x4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_imagic_x4,
                  xnn_init_f32_qu8_cvt_scalar_imagic_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_x1,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_x1,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_x2,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_x2,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_x3,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_x3,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_qu8_vcvt, scalar_lrintf_x4,
                  xnn_f32_qu8_vcvt_ukernel__scalar_lrintf_x4,
                  xnn_init_f32_qu8_cvt_scalar_lrintf_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, uint8_t>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
