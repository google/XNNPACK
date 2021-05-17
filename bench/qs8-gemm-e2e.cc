// Copyright 2021 Google LLC
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


// define XNN_ENABLE_FULL_BENCHMARKS=1 to enable all microkernel benchmarks.

static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_qs8_gemm_ukernel_function gemm,
  xnn_qs8_igemm_ukernel_function igemm,
  xnn_qs8_gemm_ukernel_function gemm1,
  xnn_qs8_igemm_ukernel_function igemm1,
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
  // Note: do not directly assign to xnn_params.qs8.gemm because it breaks older gcc.
  xnn_params.qs8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function(gemm));
  xnn_params.qs8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function(igemm));
  xnn_params.qs8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function(gemm1));
  xnn_params.qs8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function(igemm1));
  xnn_params.qs8.gemm.mr = mr;
  xnn_params.qs8.gemm.nr = nr;
  xnn_params.qs8.gemm.log2_kr = log2_kr;
  xnn_params.qs8.gemm.log2_sr = log2_sr;

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
#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld32,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld32,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      1 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      1 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm,
      1 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      1 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      1 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53,
      1 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_cortex_a55,
      xnn_qs8_igemm_minmax_ukernel_4x16c4__aarch64_neondot_cortex_a55,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      4 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld32,
      xnn_qs8_igemm_minmax_ukernel_4x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld32,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      4 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      4 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_cortex_a53,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      2 /* mr */, 8  /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld32)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64)
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_cortex_a55)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld32)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64)
#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__aarch64_neon_mlal_padal_cortex_a53)
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mlal_padal_prfm)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__aarch64_neon_mull_padal)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c16__aarch64_neon_mlal_padal)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane,
      1 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      1 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_2x8__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane,
      2 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_2x16__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      2 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_3x8__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane,
      3 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_3x16__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_4x8__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane,
      4 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_4x16__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_6x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_6x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_6x8__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane,
      6 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_6x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_6x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_6x16__neon_mlal_lane,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      1 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      1 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      2 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      2 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      3 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      4 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mlal_lane_prfm,
      6 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mlal_lane_prfm,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      1 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      1 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_2x8__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      2 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_2x16__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      2 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_3x8__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      3 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_3x16__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_4x8__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8__neon_mull_addw_dup,
      4 /* mr */, 8  /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16__neon_mull_addw_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_4x16__neon_mull_addw_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16__neon_mull_addw_dup,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      1 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      1 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      2 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_2x16c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      2 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_3x8c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      3 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_3x16c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      3 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_4x8c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup,
      4 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16c2__neon_mlal_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_4x16c2__neon_mlal_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup,
      4 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      1 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      1 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_2x8c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      2 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_2x16c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      2 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_3x8c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      3 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_3x16c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      3 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_4x8c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x8c2__neon_mull_padal_dup,
      4 /* mr */, 8  /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16c2__neon_mull_padal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_4x16c2__neon_mull_padal_dup,
      xnn_qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      xnn_qs8_igemm_minmax_ukernel_1x16c2__neon_mull_padal_dup,
      4 /* mr */, 16 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot,
      1 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_1x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      1 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_4x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_4x8c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot,
      4 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_4x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_4x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      4 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_6x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_6x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_6x8c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot,
      6 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_6x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_6x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_6x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      6 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_8x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_8x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_8x8c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot,
      8 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qs8_gemm_minmax_ukernel_8x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_8x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_8x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      8 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mull_padal,
      1 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mull_padal,
      1 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mull_padal,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_2x16c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mull_padal,
      2 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_3x8c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mull_padal,
      3 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_3x16c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mull_padal,
      3 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_4x8c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mull_padal,
      4 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16c8__neon_mull_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_4x16c8__neon_mull_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mull_padal,
      4 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      1 /* mr */, 8  /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      1 /* mr */, 16 /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_2x8c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      2 /* mr */, 8  /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_2x16c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      2 /* mr */, 16 /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_3x8c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      4 /* mr */, 8  /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_3x16c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      4 /* mr */, 16 /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_4x8c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c16__neon_mlal_padal,
      4 /* mr */, 8  /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16c16__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_4x16c16__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c16__neon_mlal_padal,
      4 /* mr */, 16 /* nr */, 4 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      1 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      1 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      2 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_2x16c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_2x16c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      2 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x8c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_3x8c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      3 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_3x16c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_3x16c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      3 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x8c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_4x8c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__neon_mlal_padal,
      4 /* mr */, 8  /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qs8_gemm_minmax_ukernel_4x16c8__neon_mlal_padal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_4x16c8__neon_mlal_padal,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__neon_mlal_padal,
      4 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c4__neondot);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_8x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_8x16c4__neondot);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c8__neon_mlal_padal);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16c8__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8c8__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16c8__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c8__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c8__neon_mlal_padal);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__neon_mull_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c8__neon_mull_padal);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__neon_mull_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16c8__neon_mull_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8c8__neon_mull_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16c8__neon_mull_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c8__neon_mull_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c8__neon_mull_padal);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c16__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c16__neon_mlal_padal);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c16__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16c16__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8c16__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16c16__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c16__neon_mlal_padal);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c16__neon_mlal_padal);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c2__neon_mlal_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c2__neon_mlal_padal_dup);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c2__neon_mlal_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16c2__neon_mlal_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8c2__neon_mlal_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16c2__neon_mlal_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c2__neon_mlal_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c2__neon_mlal_padal_dup);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c2__neon_mull_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c2__neon_mull_padal_dup);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c2__neon_mull_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16c2__neon_mull_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8c2__neon_mull_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16c2__neon_mull_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c2__neon_mull_padal_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c2__neon_mull_padal_dup);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x16__neon_mlal_lane);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8__neon_mull_addw_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16__neon_mull_addw_dup);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8__neon_mull_addw_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16__neon_mull_addw_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8__neon_mull_addw_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16__neon_mull_addw_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8__neon_mull_addw_dup);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16__neon_mull_addw_dup);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__avx512skx,
      1 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void qs8_gemm_minmax_ukernel_2x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_2x16c8__avx512skx,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__avx512skx,
      2 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_3x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_3x16c8__avx512skx,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__avx512skx,
      3 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void qs8_gemm_minmax_ukernel_4x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_4x16c8__avx512skx,
      xnn_qs8_gemm_minmax_ukernel_1x16c8__avx512skx,
      xnn_qs8_igemm_minmax_ukernel_1x16c8__avx512skx,
      4 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__avx2,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__avx2,
      1 /* mr */, 8 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX2);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x8c8__avx2,
      xnn_qs8_igemm_minmax_ukernel_2x8c8__avx2,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__avx2,
      2 /* mr */, 8 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX2);
  }

  static void qs8_gemm_minmax_ukernel_3x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x8c8__avx2,
      xnn_qs8_igemm_minmax_ukernel_3x8c8__avx2,
      xnn_qs8_gemm_minmax_ukernel_1x8c8__avx2,
      xnn_qs8_igemm_minmax_ukernel_1x8c8__avx2,
      3 /* mr */, 8 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX2);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld64,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_1x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld128,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld64,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_2x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld128,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld64,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld128,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld64,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__xop_ld128,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld64,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_1x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld128,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld64,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_2x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld128,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__xop_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld64,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__xop_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__xop_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__xop_ld128,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld64,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_1x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld128,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld64,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_2x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld128,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld64,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld128,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld64,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__avx_ld128,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld64,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_1x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld128,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld64,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_2x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld128,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__avx_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld64,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__avx_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__avx_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__avx_ld128,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld64,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld128,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld64,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_2x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld128,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld64,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld128,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld64,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse41_ld128,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld64,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_1x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld128,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld64,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_2x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld128,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__sse41_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld64,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__sse41_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse41_ld128,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld64,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld128,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c2__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld64,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_2x4c2__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld128,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld64,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld128,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld64,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__ssse3_ld128,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld64,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld128,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld64,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld128,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__ssse3_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__ssse3_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld64,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__ssse3_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__ssse3_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__ssse3_ld128,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSSE3);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld64,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld128,
      1 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld64,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_2x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c2__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld128,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld64,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_3x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c2__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld128,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld64,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_4x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_4x4c2__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c2__sse2_ld128,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld64,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_1x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld128,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld64,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_2x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld128,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__sse2_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld64,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__sse2_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__sse2_ld128,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }


#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16c8__avx512skx);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c8__avx512skx);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c8__avx2);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8c8__avx2);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8c8__avx2);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__xop_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__xop_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__xop_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__xop_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__avx_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__avx_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__avx_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__avx_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__sse41_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__sse41_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__sse41_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__sse41_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__ssse3_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__ssse3_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__ssse3_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__ssse3_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__ssse3_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__ssse3_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__ssse3_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__ssse3_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c2__sse2_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c2__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c2__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4c2__sse2_ld128);

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__sse2_ld128);
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__sse2_ld128);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
#if XNN_ENABLE_FULL_BENCHMARKS
  static void qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

  static void qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      1 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
#endif  // XNN_ENABLE_FULL_BENCHMARKS

  static void qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld64,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__wasmsimd_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

  static void qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld128,
      xnn_qs8_igemm_minmax_ukernel_2x4c8__wasmsimd_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld64,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__wasmsimd_ld64,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld64,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

  static void qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld128,
      xnn_qs8_igemm_minmax_ukernel_3x4c8__wasmsimd_ld128,
      xnn_qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      xnn_qs8_igemm_minmax_ukernel_1x4c8__wasmsimd_ld128,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

#if XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128)
#endif  // XNN_ENABLE_FULL_BENCHMARKS
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld64)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld128)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld64)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld128)
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ENABLE_FULL_BENCHMARKS
static void qs8_gemm_minmax_ukernel_1x2__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_1x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x2__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x2__scalar,
    1 /* mr */, 2 /* nr */);
}

static void qs8_gemm_minmax_ukernel_1x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_1x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x4__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x4__scalar,
    1 /* mr */, 4 /* nr */);
}
#endif  // XNN_ENABLE_FULL_BENCHMARKS

static void qs8_gemm_minmax_ukernel_2x2__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_2x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_2x2__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x2__scalar,
    2 /* mr */, 2 /* nr */);
}

static void qs8_gemm_minmax_ukernel_3x2__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_3x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_3x2__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x2__scalar,
    3 /* mr */, 2 /* nr */);
}

static void qs8_gemm_minmax_ukernel_4x2__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_4x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_4x2__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x2__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x2__scalar,
    4 /* mr */, 2 /* nr */);
}

static void qs8_gemm_minmax_ukernel_2x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_2x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_2x4__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x4__scalar,
    2 /* mr */, 4 /* nr */);
}

static void qs8_gemm_minmax_ukernel_3x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_3x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_3x4__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x4__scalar,
    3 /* mr */, 4 /* nr */);
}

static void qs8_gemm_minmax_ukernel_4x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_4x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_4x4__scalar,
    xnn_qs8_gemm_minmax_ukernel_1x4__scalar,
    xnn_qs8_igemm_minmax_ukernel_1x4__scalar,
    4 /* mr */, 4 /* nr */);
}

#if XNN_ENABLE_FULL_BENCHMARKS
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x2__scalar)
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4__scalar)
#endif  // XNN_ENABLE_FULL_BENCHMARKS
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x2__scalar)
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x2__scalar)
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x2__scalar)
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4__scalar)
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4__scalar)
BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x4__scalar)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
