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
#include <xnnpack/params-init.h>


static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_qu8_gemm_minmax_ukernel_function gemm,
  xnn_qu8_igemm_minmax_ukernel_function igemm,
  xnn_qu8_gemm_minmax_ukernel_function gemm1,
  xnn_qu8_igemm_minmax_ukernel_function igemm1,
  xnn_init_qu8_conv_minmax_params_fn init_params,
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
  // Note: do not directly assign to xnn_params.qu8.gemm because it breaks older gcc.
  xnn_params.qu8.gemm.minmax.gemm = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function(gemm));
  xnn_params.qu8.gemm.minmax.igemm = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function(igemm));
  xnn_params.qu8.gemm.minmax.gemm1 = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function(gemm1));
  xnn_params.qu8.gemm.minmax.igemm1 = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function(igemm1));
  xnn_params.qu8.gemm.init.qu8 = init_params;
  xnn_params.qu8.gemm.mr = mr;
  xnn_params.qu8.gemm.nr = nr;
  xnn_params.qu8.gemm.log2_kr = log2_kr;
  xnn_params.qu8.gemm.log2_sr = log2_sr;

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

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qu8_gemm_4x8__aarch32_neon_mlal_lane_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a53,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qu8_gemm_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qu8_gemm_4x8__aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_cortex_a7,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qu8_gemm_4x8__aarch32_neon_mlal_lane_prfm_cortex_a7(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_cortex_a7,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qu8_gemm_4x8__aarch32_neon_mlal_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qu8_gemm_4x8__aarch32_neon_mlal_lane_prfm_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__aarch32_neon_mlal_lane_prfm_cortex_a53)
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__aarch32_neon_mlal_lane_cortex_a53)
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__aarch32_neon_mlal_lane_prfm_cortex_a7)
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__aarch32_neon_mlal_lane_cortex_a7)
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__aarch32_neon_mlal_lane_prfm_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__aarch32_neon_mlal_lane_ld64)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qu8_gemm_4x16c4__aarch64_neondot_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_cortex_a55,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_4x16c4__aarch64_neondot_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__aarch64_neondot_ld128,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_4x8c4__aarch64_neondot_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_ld128,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_ld128,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_4x8c4__aarch64_neondot_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__aarch64_neondot_cortex_a55,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  static void qu8_gemm_4x16__aarch64_neon_mlal_lane_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a75,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a75,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_4x16__aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_cortex_a53,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void qu8_gemm_4x16__aarch64_neon_mlal_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_ld64,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_4x16__aarch64_neon_mlal_lane_prfm_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__aarch64_neon_mlal_lane_prfm_ld64,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  BENCHMARK_QU8_END2END(qu8_gemm_4x8c4__aarch64_neondot_cortex_a55);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16c4__aarch64_neondot_cortex_a55);
  BENCHMARK_QU8_END2END(qu8_gemm_4x8c4__aarch64_neondot_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16c4__aarch64_neondot_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__aarch64_neon_mlal_lane_cortex_a75);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__aarch64_neon_mlal_lane_prfm_cortex_a75);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__aarch64_neon_mlal_lane_cortex_a53);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__aarch64_neon_mlal_lane_prfm_cortex_a53);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__aarch64_neon_mlal_lane_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__aarch64_neon_mlal_lane_prfm_ld64);
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_gemm_2x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      2 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_3x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      3 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_4x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_6x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      6 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_2x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      2 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_3x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      3 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_4x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_6x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      6 /* mr */, 16 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void qu8_gemm_1x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      1 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_2x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_2x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      2 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_3x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_3x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      3 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_4x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_5x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_5x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_5x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      5 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_6x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_6x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      6 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_8x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_8x8c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x8c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      8 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_1x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      1 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_2x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_2x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      2 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_3x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_3x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      3 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_4x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_4x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      4 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_5x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_5x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      5 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_6x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_6x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_6x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      6 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_8x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_8x16c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x16c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      8 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_2x32c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_2x32c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x32c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x32c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      2 /* mr */, 32 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  static void qu8_gemm_3x32c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_3x32c4__neondot,
      xnn_qu8_gemm_minmax_rndnu_ukernel_1x32c4__neondot,
      xnn_qu8_igemm_minmax_rndnu_ukernel_1x32c4__neondot,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      3 /* mr */, 32 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }
  BENCHMARK_QU8_END2END(qu8_gemm_1x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_2x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_3x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_4x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_5x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_6x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_8x8c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_1x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_2x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_3x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_5x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_6x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_8x16c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_2x32c4__neondot);
  BENCHMARK_QU8_END2END(qu8_gemm_3x32c4__neondot);

  BENCHMARK_QU8_END2END(qu8_gemm_2x8__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_3x8__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_4x8__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_6x8__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_2x16__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_3x16__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16__neon_mlal_lane);
  BENCHMARK_QU8_END2END(qu8_gemm_6x16__neon_mlal_lane);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_gemm_2x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x16c8__avx512skx,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      2 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void qu8_gemm_3x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x16c8__avx512skx,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      3 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void qu8_gemm_4x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x16c8__avx512skx,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      4 /* mr */, 16 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX512F);
  }

  static void qu8_gemm_2x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      2 /* mr */, 8 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX2);
  }
  static void qu8_gemm_3x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      3 /* mr */, 8 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX2);
  }

  static void qu8_gemm_2x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__xop_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_2x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__xop_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_3x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__xop_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_3x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__xop_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_4x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__xop_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_4x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__xop_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qu8_gemm_2x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_3x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__xop_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qu8_gemm_2x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__xop_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }
  static void qu8_gemm_3x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__xop_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckXOP);
  }

  static void qu8_gemm_2x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_2x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_3x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_3x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_4x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_4x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }


  static void qu8_gemm_2x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_2x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_3x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }
  static void qu8_gemm_3x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckAVX);
  }

  static void qu8_gemm_2x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_2x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_3x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_3x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_4x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_4x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qu8_gemm_2x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_2x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_3x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }
  static void qu8_gemm_3x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckSSE41);
  }

  static void qu8_gemm_2x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_2x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_3x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_3x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_4x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_4x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 0 /* log2_sr */);
  }

  static void qu8_gemm_2x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_2x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_3x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }
  static void qu8_gemm_3x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */, 0 /* log2_sr */);
  }


  BENCHMARK_QU8_END2END(qu8_gemm_2x16c8__avx512skx);
  BENCHMARK_QU8_END2END(qu8_gemm_3x16c8__avx512skx);
  BENCHMARK_QU8_END2END(qu8_gemm_4x16c8__avx512skx);

  BENCHMARK_QU8_END2END(qu8_gemm_2x8c8__avx2);
  BENCHMARK_QU8_END2END(qu8_gemm_3x8c8__avx2);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__xop_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__xop_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__xop_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__xop_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__xop_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__xop_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__xop_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__xop_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__xop_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__xop_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__avx_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__avx_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__avx_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__avx_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__avx_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__avx_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__avx_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__avx_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__avx_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__avx_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__sse41_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__sse41_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__sse41_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__sse41_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__sse41_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__sse41_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__sse41_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__sse41_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__sse41_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__sse41_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__sse2_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__sse2_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__sse2_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__sse2_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__sse2_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__sse2_ld128);

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__sse2_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__sse2_ld128);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__sse2_ld64);
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__sse2_ld128);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */);
  }
  static void qu8_gemm_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */);
  }
  static void qu8_gemm_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */);
  }
  static void qu8_gemm_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */);
  }
  static void qu8_gemm_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */);
  }
  static void qu8_gemm_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */);
  }

  static void qu8_gemm_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 2 /* log2_sr */);
  }
  static void qu8_gemm_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 1 /* log2_kr */, 2 /* log2_sr */);
  }
  static void qu8_gemm_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 2 /* log2_sr */);
  }
  static void qu8_gemm_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 1 /* log2_kr */, 2 /* log2_sr */);
  }
  static void qu8_gemm_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 2 /* log2_sr */);
  }
  static void qu8_gemm_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      4 /* mr */, 4 /* nr */, 1 /* log2_kr */, 2 /* log2_sr */);
  }

  static void qu8_gemm_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_3x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      4 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      4 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

  static void qu8_gemm_2x4c8__wasmsimd_mul32_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_mul32_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_mul32_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_2x4c8__wasmsimd_mul32_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_mul32_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_mul32_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      2 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_3x4c8__wasmsimd_mul32_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul32_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul32_ld64,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld64,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld64,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }
  static void qu8_gemm_3x4c8__wasmsimd_mul32_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul32_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_mul32_ld128,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld128,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_mul32_ld128,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      3 /* mr */, 4 /* nr */, 3 /* log2_kr */);
  }

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2__wasmsimd_dot16x2_ld128)

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c2s4__wasmsimd_dot16x2_ld128)

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__wasmsimd_dot16x2_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__wasmsimd_dot16x2_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4c8__wasmsimd_dot16x2_ld128)

  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__wasmsimd_mul32_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_2x4c8__wasmsimd_mul32_ld128)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__wasmsimd_mul32_ld64)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4c8__wasmsimd_mul32_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qu8_gemm_2x2__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x2__wasm_fmagic,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      2 /* mr */, 2 /* nr */);
  }
  static void qu8_gemm_3x2__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x2__wasm_fmagic,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      3 /* mr */, 2 /* nr */);
  }
  static void qu8_gemm_4x2__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x2__wasm_fmagic,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      4 /* mr */, 2 /* nr */);
  }
  static void qu8_gemm_2x4__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_2x4__wasm_fmagic,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      2 /* mr */, 4 /* nr */);
  }
  static void qu8_gemm_3x4__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_3x4__wasm_fmagic,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      3 /* mr */, 4 /* nr */);
  }
  static void qu8_gemm_4x4__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic,
      xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_qu8_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
      4 /* mr */, 4 /* nr */);
  }

  BENCHMARK_QU8_END2END(qu8_gemm_2x2__wasm_fmagic)
  BENCHMARK_QU8_END2END(qu8_gemm_3x2__wasm_fmagic)
  BENCHMARK_QU8_END2END(qu8_gemm_4x2__wasm_fmagic)
  BENCHMARK_QU8_END2END(qu8_gemm_2x4__wasm_fmagic)
  BENCHMARK_QU8_END2END(qu8_gemm_3x4__wasm_fmagic)
  BENCHMARK_QU8_END2END(qu8_gemm_4x4__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


static void qu8_gemm_2x2__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    2 /* mr */, 2 /* nr */);
}
static void qu8_gemm_3x2__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    3 /* mr */, 2 /* nr */);
}
static void qu8_gemm_4x2__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    4 /* mr */, 2 /* nr */);
}
static void qu8_gemm_2x4__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    2 /* mr */, 4 /* nr */);
}
static void qu8_gemm_3x4__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    3 /* mr */, 4 /* nr */);
}
static void qu8_gemm_4x4__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params,
    4 /* mr */, 4 /* nr */);
}

static void qu8_gemm_2x2__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    2 /* mr */, 2 /* nr */);
}
static void qu8_gemm_3x2__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    3 /* mr */, 2 /* nr */);
}
static void qu8_gemm_4x2__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    4 /* mr */, 2 /* nr */);
}
static void qu8_gemm_2x4__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    2 /* mr */, 4 /* nr */);
}
static void qu8_gemm_3x4__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    3 /* mr */, 4 /* nr */);
}
static void qu8_gemm_4x4__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params,
    4 /* mr */, 4 /* nr */);
}

static void qu8_gemm_2x2__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    2 /* mr */, 2 /* nr */);
}
static void qu8_gemm_3x2__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    3 /* mr */, 2 /* nr */);
}
static void qu8_gemm_4x2__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    4 /* mr */, 2 /* nr */);
}
static void qu8_gemm_2x4__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    2 /* mr */, 4 /* nr */);
}
static void qu8_gemm_3x4__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    3 /* mr */, 4 /* nr */);
}
static void qu8_gemm_4x4__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_qu8_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params,
    4 /* mr */, 4 /* nr */);
}

BENCHMARK_QU8_END2END(qu8_gemm_2x2__scalar_fmagic)
BENCHMARK_QU8_END2END(qu8_gemm_3x2__scalar_fmagic)
BENCHMARK_QU8_END2END(qu8_gemm_4x2__scalar_fmagic)
BENCHMARK_QU8_END2END(qu8_gemm_2x4__scalar_fmagic)
BENCHMARK_QU8_END2END(qu8_gemm_3x4__scalar_fmagic)
BENCHMARK_QU8_END2END(qu8_gemm_4x4__scalar_fmagic)

BENCHMARK_QU8_END2END(qu8_gemm_2x2__scalar_imagic)
BENCHMARK_QU8_END2END(qu8_gemm_3x2__scalar_imagic)
BENCHMARK_QU8_END2END(qu8_gemm_4x2__scalar_imagic)
BENCHMARK_QU8_END2END(qu8_gemm_2x4__scalar_imagic)
BENCHMARK_QU8_END2END(qu8_gemm_3x4__scalar_imagic)
BENCHMARK_QU8_END2END(qu8_gemm_4x4__scalar_imagic)

BENCHMARK_QU8_END2END(qu8_gemm_2x2__scalar_lrintf)
BENCHMARK_QU8_END2END(qu8_gemm_3x2__scalar_lrintf)
BENCHMARK_QU8_END2END(qu8_gemm_4x2__scalar_lrintf)
BENCHMARK_QU8_END2END(qu8_gemm_2x4__scalar_lrintf)
BENCHMARK_QU8_END2END(qu8_gemm_3x4__scalar_lrintf)
BENCHMARK_QU8_END2END(qu8_gemm_4x4__scalar_lrintf)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
