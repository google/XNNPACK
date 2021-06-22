// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>

#include <benchmark/benchmark.h>

#include "bench/end2end.h"
#include "bench/utils.h"
#include "models/models.h"
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f32_gemm_minmax_ukernel_function gemm,
  xnn_f32_igemm_minmax_ukernel_function igemm,
  xnn_f32_gemm_minmax_ukernel_function gemm1,
  xnn_f32_igemm_minmax_ukernel_function igemm1,
  xnn_init_f32_minmax_params_fn init_params,
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
  xnn_params.f32.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function(gemm));
  xnn_params.f32.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function(igemm));
  xnn_params.f32.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function(gemm1));
  xnn_params.f32.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function(igemm1));
  xnn_params.f32.gemm.init.f32 = init_params;
  xnn_params.f32.gemm.mr = mr;
  xnn_params.f32.gemm.nr = nr;
  xnn_params.f32.gemm.log2_kr = log2_kr;
  xnn_params.f32.gemm.log2_sr = log2_sr;

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


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x12__aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x12__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x12__aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x12__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x12__aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 12 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a55,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_5x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a55,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a53,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a73(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a73,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__neonfma_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_ld64)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_cortex_a53)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_cortex_a55)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_5x8__aarch64_neonfma_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__aarch64_neonfma_prfm_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_cortex_a55);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_cortex_a73);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_prfm_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_4x12__aarch64_neonfma_cortex_a53);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__neonfma_lane_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_6x8__neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neonfma_lane_ld128);
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x8__aarch32_neon_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a7(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a7,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__aarch32_neon_pld_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_pld_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_pld_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch32_neon_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch32_neon_cortex_a7);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch32_neon_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch32_neon_cortex_a55);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch32_neon_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch32_neon_pld_cortex_a75);
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_dup_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8__neon_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__neon_dup_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neonfma_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_4x8__neonfma_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__neonfma_dup_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_6x8__neonfma_dup_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_6x8__neonfma_dup_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__neonfma_dup_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neonfma_dup_ld64,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_4x8s4__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neon,
      xnn_f32_igemm_minmax_ukernel_4x8s4__neon,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neon,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8s4__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma,
      xnn_f32_igemm_minmax_ukernel_4x8s4__neonfma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_6x8s4__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neon,
      xnn_f32_igemm_minmax_ukernel_6x8s4__neon,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neon,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x8s4__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma,
      xnn_f32_igemm_minmax_ukernel_6x8s4__neonfma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  static void f32_gemm_8x8s4__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_8x8s4__neon,
      xnn_f32_igemm_minmax_ukernel_8x8s4__neon,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neon,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_8x8s4__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma,
      xnn_f32_igemm_minmax_ukernel_8x8s4__neonfma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x8__neon_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__neon_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neon_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neon_lane_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__neon_dup_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__neon_dup_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neon_dup_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neon_dup_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__neonfma_dup_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__neonfma_dup_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neonfma_dup_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__neonfma_dup_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__neon);
  BENCHMARK_FP32_END2END(f32_gemm_6x8s4__neon);
  BENCHMARK_FP32_END2END(f32_gemm_8x8s4__neon);

  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__neonfma);
  BENCHMARK_FP32_END2END(f32_gemm_6x8s4__neonfma);
  BENCHMARK_FP32_END2END(f32_gemm_8x8s4__neonfma);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_4x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_5x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_5x16__avx512f_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_6x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_7x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_7x16__avx512f_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      7 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }
  static void f32_gemm_8x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_8x16__avx512f_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_init_f32_minmax_scalar_params,
      8 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void f32_gemm_4x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x8__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_5x8__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      5 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_6x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_6x8__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_7x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_7x8__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      7 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_8x8__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_8x8__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      8 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_3x16__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x16__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      5 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_3x16s4__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_3x16s4__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_4x16s4__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }
  static void f32_gemm_5x16s4__fma3_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_5x16s4__fma3_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast,
      xnn_init_f32_minmax_avx_params,
      5 /* mr */, 16 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */,
      benchmark::utils::CheckFMA3);
  }

  static void f32_gemm_4x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x8__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_5x8__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      5 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_6x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_6x8__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_7x8__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_7x8__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x8__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      7 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_3x16__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_3x16__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_4x16__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x16__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void f32_gemm_5x16__avx_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_5x16__avx_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx_broadcast,
      xnn_init_f32_minmax_avx_params,
      5 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void f32_gemm_3x8__sse2_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_3x8__sse2_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__sse2_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_4x8__sse2_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__sse2_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_5x8__sse2_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_3x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_3x8__sse_load1,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_4x8__sse_load1,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_5x8__sse_load1,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_3x8__sse_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_4x8__sse_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_5x8__sse_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_3x8s4__sse,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_4x8s4__sse,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_5x8s4__sse,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x16__avx512f_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_5x16__avx512f_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_6x16__avx512f_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_7x16__avx512f_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_8x16__avx512f_broadcast);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_7x8__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_8x8__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_3x16__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_4x16__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_5x16__fma3_broadcast);

  BENCHMARK_FP32_END2END(f32_gemm_3x16s4__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_4x16s4__fma3_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_5x16s4__fma3_broadcast);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__avx_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__avx_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__avx_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_7x8__avx_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_3x16__avx_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_4x16__avx_broadcast);
  BENCHMARK_FP32_END2END(f32_gemm_5x16__avx_broadcast);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__sse2_dup);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__sse2_dup);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__sse2_dup);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__sse_load1);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__sse_load1);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__sse_load1);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__sse_dup);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__sse_dup);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__sse_dup);

  BENCHMARK_FP32_END2END(f32_gemm_3x8s4__sse);
  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__sse);
  BENCHMARK_FP32_END2END(f32_gemm_5x8s4__sse);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  static void f32_gemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_6x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_3x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_6x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmsimd_arm_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmsimd_arm_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmsimd_arm_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmsimd_arm_loadsplat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmsimd_x86_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmsimd_x86_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmsimd_x86_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmsimd_x86_loadsplat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmsimd_arm_splat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmsimd_arm_splat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmsimd_arm_splat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmsimd_arm_splat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmsimd_x86_splat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmsimd_x86_splat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmsimd_x86_splat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmsimd_x86_splat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8s4__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_gemm_5x8s4__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_gemm_6x8s4__wasmsimd_arm);

  BENCHMARK_FP32_END2END(f32_gemm_3x8s4__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_gemm_5x8s4__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_gemm_6x8s4__wasmsimd_x86);
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASM
  static void f32_gemm_2x4__wasm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_2x4__wasm,
      xnn_f32_igemm_minmax_ukernel_2x4__wasm,
      xnn_f32_gemm_minmax_ukernel_1x4__wasm,
      xnn_f32_igemm_minmax_ukernel_1x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      2 /* mr */, 4 /* nr */);
  }

  static void f32_gemm_4x4__wasm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x4__wasm,
      xnn_f32_igemm_minmax_ukernel_4x4__wasm,
      xnn_f32_gemm_minmax_ukernel_1x4__wasm,
      xnn_f32_igemm_minmax_ukernel_1x4__wasm,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 4 /* nr */);
  }

  BENCHMARK_FP32_END2END(f32_gemm_2x4__wasm);
  BENCHMARK_FP32_END2END(f32_gemm_4x4__wasm);
#endif  // XNN_ARCH_WASM


static void f32_gemm_2x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_f32_gemm_minmax_ukernel_2x4__scalar,
    xnn_f32_igemm_minmax_ukernel_2x4__scalar,
    xnn_f32_gemm_minmax_ukernel_1x4__scalar,
    xnn_f32_igemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    2 /* mr */, 4 /* nr */);
}

static void f32_gemm_4x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_f32_gemm_minmax_ukernel_4x4__scalar,
    xnn_f32_igemm_minmax_ukernel_4x4__scalar,
    xnn_f32_gemm_minmax_ukernel_1x4__scalar,
    xnn_f32_igemm_minmax_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    4 /* mr */, 4 /* nr */);
}

BENCHMARK_FP32_END2END(f32_gemm_2x4__scalar);
BENCHMARK_FP32_END2END(f32_gemm_4x4__scalar);


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
