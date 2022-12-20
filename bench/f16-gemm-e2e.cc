// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>
#include "bench/end2end.h"
#include "bench/utils.h"
#include "models/models.h"

#include <xnnpack.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>


static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f16_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_f16_igemm_minmax_ukernel_fn igemm_minmax,
  xnn_f16_gemm_minmax_ukernel_fn gemm1_minmax,
  xnn_f16_igemm_minmax_ukernel_fn igemm1_minmax,
  xnn_init_f16_minmax_params_fn init_params,
  uint8_t mr, uint8_t nr, uint8_t log2_kr = 0, uint8_t log2_sr = 0,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check && !isa_check(state)) {
    return;
  }
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  // Override microkernels chosen in xnn_initialize
  // Note: do not directly assign to xnn_params.f32.gemm because it breaks older gcc.
  std::memset(&xnn_params.f16.gemm, 0, sizeof(xnn_params.f16.gemm));
  std::memset(&xnn_params.f16.gemm2, 0, sizeof(xnn_params.f16.gemm2));
  xnn_params.f16.gemm.minmax.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm_minmax));
  xnn_params.f16.gemm.minmax.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm_minmax));
  xnn_params.f16.gemm.minmax.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1_minmax));
  xnn_params.f16.gemm.minmax.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1_minmax));
  xnn_params.f16.gemm.init.f16 = init_params;
  xnn_params.f16.gemm.mr = mr;
  xnn_params.f16.gemm.nr = nr;
  xnn_params.f16.gemm.log2_kr = log2_kr;
  xnn_params.f16.gemm.log2_sr = log2_sr;

  #if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
    // If JIT is enabled, we want to make sure that we are still benchmarking
    // non-JIT microkernels, so nullify the pointers to generators.
    for (size_t i = 0; i < XNN_MAX_MR; i++) {
      xnn_params.f16.gemm.generator.gemm[i] = xnn_init_hmp_gemm_codegen(nullptr);
      xnn_params.f16.gemm.generator.igemm[i]  = xnn_init_hmp_igemm_codegen(nullptr);
    }
  #endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

  auto execution_plan = model_factory(nullptr);
  if (execution_plan.empty()) {
    state.SkipWithError("failed to create a model");
    return;
  }

  for (auto _ : state) {
    for (const std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)>& op : execution_plan) {
      xnn_status status = xnn_run_operator(op.get(), nullptr);
      if (status != xnn_status_success) {
        state.SkipWithError("failed to run a model");
        return;
      }
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }
}

#if XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 & XNN_ENABLE_ASSEMBLY
  static void f16_gemm_4x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_8x8__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      8 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_4x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
      xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_4x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_ld32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
      xnn_f16_igemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75,
      xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_FP16_END2END(f16_gemm_4x8__asm_aarch64_neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_6x8__asm_aarch64_neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_8x8__asm_aarch64_neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_4x16__asm_aarch64_neonfp16arith_ld32);
  BENCHMARK_FP16_END2END(f16_gemm_4x16__asm_aarch64_neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_6x16__asm_aarch64_neonfp16arith_ld32);
  BENCHMARK_FP16_END2END(f16_gemm_6x16__asm_aarch64_neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55);
  BENCHMARK_FP16_END2END(f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a55r0);
  BENCHMARK_FP16_END2END(f16_gemm_6x16__asm_aarch64_neonfp16arith_cortex_a75);
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && XNN_ARCH_ARM64 & XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void f16_gemm_4x8__neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_4x8__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x8__neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_6x8__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_8x8__neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_8x8__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x8__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      8 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_4x16__neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_4x16__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_6x16__neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_6x16__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_gemm_8x16__neonfp16arith_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_8x16__neonfp16arith_ld64,
      xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_f16_igemm_minmax_ukernel_1x16__neonfp16arith_ld64,
      xnn_init_f16_minmax_fp16arith_params,
      8 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_FP16_END2END(f16_gemm_4x8__neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_6x8__neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_8x8__neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_4x16__neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_6x16__neonfp16arith_ld64);
  BENCHMARK_FP16_END2END(f16_gemm_8x16__neonfp16arith_ld64);
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
