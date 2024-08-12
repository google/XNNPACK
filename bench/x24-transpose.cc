// Copyright 2022 Google LLC
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

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/transpose.h"


void transpose(
    benchmark::State& state,
    xnn_x24_transposec_ukernel_fn transpose,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t height = state.range(0);
  const size_t width = state.range(1);
  const size_t tile_hbytes = height * 3;
  const size_t tile_wbytes = width * 3;

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> x(
      height * width * 3 + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> y(
      height * width * 3 + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::iota(x.begin(), x.end(), 0);
  std::fill(y.begin(), y.end(), 0);

  for (auto _ : state) {
    transpose(x.data(), y.data(), tile_wbytes, tile_hbytes, width,
              height);
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
  b->Args({1024, 1024});
}

BENCHMARK_CAPTURE(transpose, 1x2_scalar, xnn_x24_transposec_ukernel__1x2_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 1x4_scalar, xnn_x24_transposec_ukernel__1x4_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x1_scalar, xnn_x24_transposec_ukernel__2x1_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x2_scalar, xnn_x24_transposec_ukernel__2x2_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x4_scalar, xnn_x24_transposec_ukernel__2x4_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x1_scalar, xnn_x24_transposec_ukernel__4x1_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x2_scalar, xnn_x24_transposec_ukernel__4x2_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x4_scalar, xnn_x24_transposec_ukernel__4x4_scalar)
    ->Apply(BenchmarkKernelSize)->UseRealTime();

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(transpose, 2x2_neon_tbl64, xnn_x24_transposec_ukernel__2x2_neon_tbl64)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(transpose, 4x4_neon_tbl128, xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(transpose, 4x4_ssse3, xnn_x24_transposec_ukernel__4x4_ssse3,
                    benchmark::utils::CheckSSSE3)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
