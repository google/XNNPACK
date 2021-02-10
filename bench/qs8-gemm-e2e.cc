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
  static void qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_4x16c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot_ld64,
      xnn_qs8_igemm_minmax_ukernel_1x16c4__neondot,
      4 /* mr */, 16 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot_ld64)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
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

  static void qs8_gemm_minmax_ukernel_12x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_gemm_minmax_ukernel_12x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_12x8c4__neondot,
      xnn_qs8_gemm_minmax_ukernel_1x8c4__neondot,
      xnn_qs8_igemm_minmax_ukernel_1x8c4__neondot,
      12 /* mr */, 8  /* nr */, 2 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_4x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_6x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_8x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_8x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_12x8c4__neondot);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_WASMSIMD
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

  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld64)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_1x4c8__wasmsimd_ld128)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld64)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_2x4c8__wasmsimd_ld128)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld64)
  BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_3x4c8__wasmsimd_ld128)
#endif  // XNN_ARCH_WASMSIMD

#if SCALAR_IGEMM
static void qs8_gemm_minmax_ukernel_8x8c4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_gemm_minmax_ukernel_8x8c4__scalar,
    xnn_qs8_igemm_minmax_ukernel_8x8c4__scalar,
    xnn_qs8_gemm_minmax_ukernel_8x8c4__scalar,
    xnn_qs8_igemm_minmax_ukernel_8x8c4__scalar,
    8 /* mr */, 8 /* nr */, 2 /* log2_kr */, 0 /* log2_sr */);

BENCHMARK_QS8_END2END(qs8_gemm_minmax_ukernel_8x8c4__scalar);
#endif  // SCALAR_IGEMM

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
