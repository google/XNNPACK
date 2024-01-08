// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/transpose.h>


void transpose(
    benchmark::State& state,
    xnn_x64_transposec_ukernel_fn transpose,
    xnn_init_x64_transpose_params_fn init_params = nullptr,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t height = state.range(0);
  const size_t width = state.range(1);
  const size_t tile_hbytes = height * sizeof(uint64_t);
  const size_t tile_wbytes = width * sizeof(uint64_t);

  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> x(
      height * width + XNN_EXTRA_BYTES / sizeof(uint64_t));
  std::vector<uint64_t, AlignedAllocator<uint64_t, 64>> y(
      height * width + XNN_EXTRA_BYTES / sizeof(uint64_t));
  std::iota(x.begin(), x.end(), 0);
  std::fill(y.begin(), y.end(), 0);

  xnn_x64_transpose_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    transpose(x.data(), y.data(), tile_wbytes, tile_hbytes, width,
              height, &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

static void BenchmarkKernelSize(benchmark::internal::Benchmark* b)
{
  b->ArgNames({"height", "width"});
  b->Args({32, 32});
  b->Args({64, 64});
  b->Args({117, 117});
  b->Args({128, 128});
  b->Args({256, 256});
  b->Args({512, 512});
  b->Args({1024, 1024});
}

BENCHMARK_CAPTURE(transpose, 1x2_scalar_int, xnn_x64_transposec_ukernel__1x2_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x1_scalar_int, xnn_x64_transposec_ukernel__2x1_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x2_scalar_int, xnn_x64_transposec_ukernel__2x2_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x1_scalar_int, xnn_x64_transposec_ukernel__4x1_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x2_scalar_int, xnn_x64_transposec_ukernel__4x2_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 1x2_scalar_float, xnn_x64_transposec_ukernel__1x2_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x1_scalar_float, xnn_x64_transposec_ukernel__2x1_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x2_scalar_float, xnn_x64_transposec_ukernel__2x2_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x1_scalar_float, xnn_x64_transposec_ukernel__4x1_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x2_scalar_float, xnn_x64_transposec_ukernel__4x2_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(transpose, 2x2_multi_mov_sse2, xnn_x64_transposec_ukernel__2x2_multi_mov_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_multi_sse2, xnn_x64_transposec_ukernel__2x2_multi_multi_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_switch_sse2, xnn_x64_transposec_ukernel__2x2_multi_switch_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_reuse_mov_sse2, xnn_x64_transposec_ukernel__2x2_reuse_mov_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_reuse_multi_sse2, xnn_x64_transposec_ukernel__2x2_reuse_multi_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_reuse_switch_sse2, xnn_x64_transposec_ukernel__2x2_reuse_switch_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    4x4_multi_mov_avx,
                    xnn_x64_transposec_ukernel__4x4_multi_mov_avx,
                    xnn_init_x64_transpose_avx_params, benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    4x4_multi_multi_avx,
                    xnn_x64_transposec_ukernel__4x4_multi_multi_avx,
                    xnn_init_x64_transpose_avx_params, benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    4x4_multi_switch_avx,
                    xnn_x64_transposec_ukernel__4x4_multi_switch_avx,
                    xnn_init_x64_transpose_avx_params, benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    4x4_reuse_mov_avx,
                    xnn_x64_transposec_ukernel__4x4_reuse_mov_avx,
                    xnn_init_x64_transpose_avx_params, benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    4x4_reuse_multi_avx,
                    xnn_x64_transposec_ukernel__4x4_reuse_multi_avx,
                    xnn_init_x64_transpose_avx_params, benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    4x4_reuse_switch_avx,
                    xnn_x64_transposec_ukernel__4x4_reuse_switch_avx,
                    xnn_init_x64_transpose_avx_params, benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
