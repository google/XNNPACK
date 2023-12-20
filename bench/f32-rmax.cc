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
#include <xnnpack/reduce.h>


static void f32_rmax(
  benchmark::State& state,
  xnn_f32_reduce_ukernel_fn rmax,
  xnn_init_f32_default_params_fn init_params = nullptr,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> input(elements);
  std::generate(input.begin(), input.end(), std::ref(f32rng));

  xnn_f32_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }

  float output = std::nanf("");
  for (auto _ : state) {
    rmax(elements * sizeof(float), input.data(), &output, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rmax, avx512f_u16,
                    xnn_f32_rmax_ukernel__avx512f_u16,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx512f_u32_acc2,
                    xnn_f32_rmax_ukernel__avx512f_u32_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx512f_u48_acc3,
                    xnn_f32_rmax_ukernel__avx512f_u48_acc3,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx512f_u64_acc2,
                    xnn_f32_rmax_ukernel__avx512f_u64_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx512f_u64_acc4,
                    xnn_f32_rmax_ukernel__avx512f_u64_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, avx_u8,
                    xnn_f32_rmax_ukernel__avx_u8,
                    xnn_init_f32_default_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx_u16_acc2,
                    xnn_f32_rmax_ukernel__avx_u16_acc2,
                    xnn_init_f32_default_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx_u24_acc3,
                    xnn_f32_rmax_ukernel__avx_u24_acc3,
                    xnn_init_f32_default_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx_u32_acc2,
                    xnn_f32_rmax_ukernel__avx_u32_acc2,
                    xnn_init_f32_default_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, avx_u32_acc4,
                    xnn_f32_rmax_ukernel__avx_u32_acc4,
                    xnn_init_f32_default_avx_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, sse_u4,
                    xnn_f32_rmax_ukernel__sse_u4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, sse_u8_acc2,
                    xnn_f32_rmax_ukernel__sse_u8_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, sse_u12_acc3,
                    xnn_f32_rmax_ukernel__sse_u12_acc3)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, sse_u16_acc2,
                    xnn_f32_rmax_ukernel__sse_u16_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, sse_u16_acc4,
                    xnn_f32_rmax_ukernel__sse_u16_acc4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rmax, neon_u4,
                    xnn_f32_rmax_ukernel__neon_u4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, neon_u8_acc2,
                    xnn_f32_rmax_ukernel__neon_u8_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, neon_u12_acc3,
                    xnn_f32_rmax_ukernel__neon_u12_acc3,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, neon_u16_acc2,
                    xnn_f32_rmax_ukernel__neon_u16_acc2,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, neon_u16_acc4,
                    xnn_f32_rmax_ukernel__neon_u16_acc4,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_minmax_u4,
                    xnn_f32_rmax_ukernel__wasmsimd_minmax_u4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_minmax_u8_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_minmax_u8_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_minmax_u12_acc3,
                    xnn_f32_rmax_ukernel__wasmsimd_minmax_u12_acc3)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_minmax_u16_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_minmax_u16_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_minmax_u16_acc4,
                    xnn_f32_rmax_ukernel__wasmsimd_minmax_u16_acc4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_pminmax_u4,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_pminmax_u8_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u8_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_pminmax_u12_acc3,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u12_acc3)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_pminmax_u16_acc2,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasmsimd_pminmax_u16_acc4,
                    xnn_f32_rmax_ukernel__wasmsimd_pminmax_u16_acc4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rmax, wasm_u1,
                    xnn_f32_rmax_ukernel__wasm_u1)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasm_u2_acc2,
                    xnn_f32_rmax_ukernel__wasm_u2_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasm_u3_acc3,
                    xnn_f32_rmax_ukernel__wasm_u3_acc3)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasm_u4_acc2,
                    xnn_f32_rmax_ukernel__wasm_u4_acc2)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, wasm_u4_acc4,
                    xnn_f32_rmax_ukernel__wasm_u4_acc4)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_rmax, rvv_u1v,
                    xnn_f32_rmax_ukernel__rvv_u1v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, rvv_u2v,
                    xnn_f32_rmax_ukernel__rvv_u2v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, rvv_u4v,
                    xnn_f32_rmax_ukernel__rvv_u4v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_rmax, rvv_u8v,
                    xnn_f32_rmax_ukernel__rvv_u8v,
                    /*init_params=*/nullptr,
                    benchmark::utils::CheckRVV)
    ->Apply(benchmark::utils::ReductionParameters<float>)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

BENCHMARK_CAPTURE(f32_rmax, scalar_u1,
                  xnn_f32_rmax_ukernel__scalar_u1)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rmax, scalar_u2_acc2,
                  xnn_f32_rmax_ukernel__scalar_u2_acc2)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rmax, scalar_u3_acc3,
                  xnn_f32_rmax_ukernel__scalar_u3_acc3)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rmax, scalar_u4_acc2,
                  xnn_f32_rmax_ukernel__scalar_u4_acc2)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_rmax, scalar_u4_acc4,
                  xnn_f32_rmax_ukernel__scalar_u4_acc4)
  ->Apply(benchmark::utils::ReductionParameters<float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
