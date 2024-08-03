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

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/transpose.h"


void transpose(
    benchmark::State& state,
    xnn_x32_transposec_ukernel_fn transpose,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  const size_t height = state.range(0);
  const size_t width = state.range(1);
  const size_t tile_hbytes = height * sizeof(uint32_t);
  const size_t tile_wbytes = width * sizeof(uint32_t);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> x(
      height * width + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> y(
      height * width + XNN_EXTRA_BYTES / sizeof(uint32_t));
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
  b->Args({128, 128});
  b->Args({256, 256});
  b->Args({512, 512});
  b->Args({1024, 1024});
}

BENCHMARK_CAPTURE(transpose, 1x2_scalar_int, xnn_x32_transposec_ukernel__1x2_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 1x4_scalar_int, xnn_x32_transposec_ukernel__1x4_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x1_scalar_int, xnn_x32_transposec_ukernel__2x1_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x2_scalar_int, xnn_x32_transposec_ukernel__2x2_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x4_scalar_int, xnn_x32_transposec_ukernel__2x4_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x1_scalar_int, xnn_x32_transposec_ukernel__4x1_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x2_scalar_int, xnn_x32_transposec_ukernel__4x2_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x4_scalar_int, xnn_x32_transposec_ukernel__4x4_scalar_int)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 1x2_scalar_float, xnn_x32_transposec_ukernel__1x2_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 1x4_scalar_float, xnn_x32_transposec_ukernel__1x4_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x1_scalar_float, xnn_x32_transposec_ukernel__2x1_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x2_scalar_float, xnn_x32_transposec_ukernel__2x2_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 2x4_scalar_float, xnn_x32_transposec_ukernel__2x4_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x1_scalar_float, xnn_x32_transposec_ukernel__4x1_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x2_scalar_float, xnn_x32_transposec_ukernel__4x2_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose, 4x4_scalar_float, xnn_x32_transposec_ukernel__4x4_scalar_float)
    ->Apply(BenchmarkKernelSize)->UseRealTime();

#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

void transpose_rvv(
    benchmark::State& state,
    xnn_x32_transposec_ukernel_fn transpose_uk,
    size_t tile_height,
    benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  // Test case not supported by VLEN is skipped
  if (xnn_init_hardware_config()->vlenb < tile_height * sizeof(uint32_t)) {
    return;
  }

  transpose(state, transpose_uk, isa_check);
}

BENCHMARK_CAPTURE(transpose_rvv, 4x4_rvv, xnn_x32_transposec_ukernel__4x4_rvv, 4)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose_rvv, 8x8_rvv, xnn_x32_transposec_ukernel__8x8_rvv, 8)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose_rvv, 16x8_rvv, xnn_x32_transposec_ukernel__16x8_rvv, 16)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
BENCHMARK_CAPTURE(transpose_rvv, 32x8_rvv, xnn_x32_transposec_ukernel__32x8_rvv, 32)
    ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV

#if XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(transpose, 4x4_neon_tbl128, xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(transpose, 2x2_multi_dec_neon, xnn_x32_transposec_ukernel__2x2_multi_dec_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_mov_neon, xnn_x32_transposec_ukernel__2x2_multi_mov_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_multi_neon, xnn_x32_transposec_ukernel__2x2_multi_multi_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_switch_neon, xnn_x32_transposec_ukernel__2x2_multi_switch_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_dec_neon, xnn_x32_transposec_ukernel__2x2_reuse_dec_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_mov_neon, xnn_x32_transposec_ukernel__2x2_reuse_mov_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_multi_neon, xnn_x32_transposec_ukernel__2x2_reuse_multi_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 2x2_multi_switch_neon, xnn_x32_transposec_ukernel__2x2_reuse_switch_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_dec_zip_neon, xnn_x32_transposec_ukernel__4x4_multi_dec_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_mov_zip_neon, xnn_x32_transposec_ukernel__4x4_multi_mov_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_multi_zip_neon, xnn_x32_transposec_ukernel__4x4_multi_multi_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_switch_zip_neon, xnn_x32_transposec_ukernel__4x4_multi_switch_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_dec_zip_neon, xnn_x32_transposec_ukernel__4x4_reuse_dec_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_mov_zip_neon, xnn_x32_transposec_ukernel__4x4_reuse_mov_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_multi_zip_neon, xnn_x32_transposec_ukernel__4x4_reuse_multi_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_switch_zip_neon, xnn_x32_transposec_ukernel__4x4_reuse_switch_zip_neon)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(transpose, 4x4_multi_mov_wasmsimd, xnn_x32_transposec_ukernel__4x4_multi_mov_wasmsimd)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_multi_wasmsimd, xnn_x32_transposec_ukernel__4x4_multi_multi_wasmsimd)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_switch_wasmsimd, xnn_x32_transposec_ukernel__4x4_multi_switch_wasmsimd)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_mov_wasmsimd, xnn_x32_transposec_ukernel__4x4_reuse_mov_wasmsimd)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_multi_wasmsimd, xnn_x32_transposec_ukernel__4x4_reuse_multi_wasmsimd)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_switch_wasmsimd, xnn_x32_transposec_ukernel__4x4_reuse_switch_wasmsimd)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(transpose, 4x4_sse, xnn_x32_transposec_ukernel__4x4_sse)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_mov_sse2, xnn_x32_transposec_ukernel__4x4_multi_mov_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_multi_sse2, xnn_x32_transposec_ukernel__4x4_multi_multi_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_multi_switch_sse2, xnn_x32_transposec_ukernel__4x4_multi_switch_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_mov_sse2, xnn_x32_transposec_ukernel__4x4_reuse_mov_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_multi_sse2, xnn_x32_transposec_ukernel__4x4_reuse_multi_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose, 4x4_reuse_switch_sse2, xnn_x32_transposec_ukernel__4x4_reuse_switch_sse2)
      ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    8x8_multi_mov_avx,
                    xnn_x32_transposec_ukernel__8x8_multi_mov_avx,
                    benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    8x8_multi_switch_avx,
                    xnn_x32_transposec_ukernel__8x8_multi_switch_avx,
                    benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    8x8_reuse_mov_avx,
                    xnn_x32_transposec_ukernel__8x8_reuse_mov_avx,
                    benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    8x8_reuse_multi_avx,
                    xnn_x32_transposec_ukernel__8x8_reuse_multi_avx,
                    benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
  BENCHMARK_CAPTURE(transpose,
                    8x8_reuse_switch_avx,
                    xnn_x32_transposec_ukernel__8x8_reuse_switch_avx,
                    benchmark::utils::CheckAVX2)
       ->Apply(BenchmarkKernelSize)->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
