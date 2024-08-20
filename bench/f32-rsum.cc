// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-rsum.yaml
//   Generator: tools/generate-rdsum-benchmark.py

#include "bench/rsum-benchmark.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum, neon_u4,
                    xnn_f32_rsum_ukernel__neon_u4,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum, neon_u8_acc2,
                    xnn_f32_rsum_ukernel__neon_u8_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum, neon_u12_acc3,
                    xnn_f32_rsum_ukernel__neon_u12_acc3,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum, neon_u16_acc2,
                    xnn_f32_rsum_ukernel__neon_u16_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_rsum, neon_u16_acc4,
                    xnn_f32_rsum_ukernel__neon_u16_acc4,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckNEON)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, sse_u4,
                    xnn_f32_rsum_ukernel__sse_u4,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, sse_u8_acc2,
                    xnn_f32_rsum_ukernel__sse_u8_acc2,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, sse_u12_acc3,
                    xnn_f32_rsum_ukernel__sse_u12_acc3,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, sse_u16_acc2,
                    xnn_f32_rsum_ukernel__sse_u16_acc2,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, sse_u16_acc4,
                    xnn_f32_rsum_ukernel__sse_u16_acc4,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx_u8,
                    xnn_f32_rsum_ukernel__avx_u8,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx_u16_acc2,
                    xnn_f32_rsum_ukernel__avx_u16_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx_u24_acc3,
                    xnn_f32_rsum_ukernel__avx_u24_acc3,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx_u32_acc2,
                    xnn_f32_rsum_ukernel__avx_u32_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx_u32_acc4,
                    xnn_f32_rsum_ukernel__avx_u32_acc4,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u16,
                    xnn_f32_rsum_ukernel__avx512f_u16,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u32_acc2,
                    xnn_f32_rsum_ukernel__avx512f_u32_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u48_acc3,
                    xnn_f32_rsum_ukernel__avx512f_u48_acc3,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u64_acc2,
                    xnn_f32_rsum_ukernel__avx512f_u64_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_rsum, avx512f_u64_acc4,
                    xnn_f32_rsum_ukernel__avx512f_u64_acc4,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(f32_rsum, hvx_u32,
                    xnn_f32_rsum_ukernel__hvx_u32,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckHVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(f32_rsum, hvx_u64_acc2,
                    xnn_f32_rsum_ukernel__hvx_u64_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckHVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(f32_rsum, hvx_u96_acc3,
                    xnn_f32_rsum_ukernel__hvx_u96_acc3,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckHVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(f32_rsum, hvx_u128_acc2,
                    xnn_f32_rsum_ukernel__hvx_u128_acc2,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckHVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  BENCHMARK_CAPTURE(f32_rsum, hvx_u128_acc4,
                    xnn_f32_rsum_ukernel__hvx_u128_acc4,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckHVX)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u4,
                    xnn_f32_rsum_ukernel__wasmsimd_u4,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u8_acc2,
                    xnn_f32_rsum_ukernel__wasmsimd_u8_acc2,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u12_acc3,
                    xnn_f32_rsum_ukernel__wasmsimd_u12_acc3,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u16_acc2,
                    xnn_f32_rsum_ukernel__wasmsimd_u16_acc2,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_rsum, wasmsimd_u16_acc4,
                    xnn_f32_rsum_ukernel__wasmsimd_u16_acc4,
                    xnn_init_f32_scaleminmax_scalar_params)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  BENCHMARK_CAPTURE(f32_rsum, rvv_u1v,
                    xnn_f32_rsum_ukernel__rvv_u1v,
                    xnn_init_f32_scaleminmax_scalar_params,
                    benchmark::utils::CheckRVV)
    ->Apply(BenchmarkRSUM)
    ->UseRealTime();
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


BENCHMARK_CAPTURE(f32_rsum, scalar_u1,
                  xnn_f32_rsum_ukernel__scalar_u1,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u2_acc2,
                  xnn_f32_rsum_ukernel__scalar_u2_acc2,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u3_acc3,
                  xnn_f32_rsum_ukernel__scalar_u3_acc3,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u4_acc2,
                  xnn_f32_rsum_ukernel__scalar_u4_acc2,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_rsum, scalar_u4_acc4,
                  xnn_f32_rsum_ukernel__scalar_u4_acc4,
                  xnn_init_f32_scaleminmax_scalar_params)
  ->Apply(BenchmarkRSUM)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
