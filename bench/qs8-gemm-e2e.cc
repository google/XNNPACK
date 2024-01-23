// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/end2end.h"
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/config.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/models.h>
#include <xnnpack/pack.h>


static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm,
  xnn_qs8_qc8w_igemm_minmax_ukernel_fn igemm,
  xnn_qs8_qc8w_gemm_minmax_ukernel_fn gemm1,
  xnn_qs8_qc8w_igemm_minmax_ukernel_fn igemm1,
  xnn_init_qs8_qc8w_conv_minmax_params_fn init_params,
  uint8_t mr, uint8_t nr, uint8_t log2_kr = 0, uint8_t log2_sr = 0,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  assert(gemm_config != nullptr);

  // Override microkernels chosen in xnn_initialize
  std::memset(gemm_config, 0, sizeof(struct xnn_gemm_config));
  gemm_config->minmax.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm));
  gemm_config->minmax.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm));
  gemm_config->minmax.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1));
  gemm_config->minmax.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1));
  gemm_config->init.qs8_qc8w = init_params;
  gemm_config->mr = mr;
  gemm_config->nr = nr;
  gemm_config->log2_kr = log2_kr;
  gemm_config->log2_sr = log2_sr;
  gemm_config->pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_qs8_gemm_goi_w;
  gemm_config->pack_igemm_goki = (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w;
  gemm_config->pack_igemm_kgo = (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w;
  gemm_config->pack_deconv_goki = (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w;

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


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_4x8c4__asm_aarch32_neondot_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_4x8c4__asm_aarch32_neondot_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8c4__asm_aarch32_neondot_cortex_a55)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8c4__asm_aarch32_neondot_ld64)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a7(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__asm_aarch32_neon_mlal_lane_ld64)
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY


#if XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_ld32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_cortex_a55)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_ld32)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c4__asm_aarch64_neondot_ld128)
#endif  // XNN_ENABLE_ARM_DOTPROD && XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY


#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16__asm_aarch64_neon_mlal_lane_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal_cortex_a53)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal_prfm)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__asm_aarch64_neon_mlal)
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_2x8c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_4x8c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_6x8c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_8x8c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/8, 8 /*nr=*/, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_2x16c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_4x16c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_6x16c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }
  static void qs8_qc8w_gemm_8x16c8__neoni8mm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__neoni8mm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__neoni8mm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/8, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckNEONI8MM);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x8c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x8c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x16c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16c8__neoni8mm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x16c8__neoni8mm);
#endif  // XNN_ENABLE_ARM_I8MM && XNN_ARCH_ARM64

#if XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void qs8_qc8w_gemm_4x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_6x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_8x8c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/8, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_4x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_6x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }
  static void qs8_qc8w_gemm_8x16c4__neondot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c4__neondot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__neondot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/8, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEONDOT);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x8c4__neondot);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16c4__neondot);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x16c4__neondot);
#endif  // XNN_ENABLE_ARM_DOTPROD && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qs8_qc8w_gemm_2x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_3x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/3, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_3x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_6x8__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_6x16__neon_mlal_lane(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neon_mlal_lane,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_3x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/3, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_3x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_4x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_6x8__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, 8 /*nr=*/, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_6x16__neon_mlal_lane_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c2__neon_mlal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c2__neon_mlal_ld1r(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c2__neon_mlal_ld2r(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c2__neon_mlal_ld4r(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c2s4__neon_mlal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c4__neon_mlal_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c4__neon_mlal_ld1r(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c4__neon_mlal_ld2r(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void qs8_qc8w_gemm_2x8c4s2__neon_mlal(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal,
      xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params,
      /*mr=*/2, 8 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/1,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c4__neon_mlal_dup);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c4__neon_mlal_ld1r);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c4__neon_mlal_ld2r);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c4s2__neon_mlal);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c2__neon_mlal_dup);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c2__neon_mlal_ld1r);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c2__neon_mlal_ld2r);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c2s4__neon_mlal);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x8__neon_mlal_lane);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16__neon_mlal_lane);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x16__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x16__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x8__neon_mlal_lane_prfm);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16__neon_mlal_lane_prfm);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM
  static void qs8_qc8w_gemm_1x1c4__armsimd32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      /*mr=*/1, 1 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckARMV6);
  }
  static void qs8_qc8w_gemm_2x1c4__armsimd32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x1c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x1c4__armsimd32,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x1c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      /*mr=*/2, 1 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckARMV6);
  }
  static void qs8_qc8w_gemm_1x2c4__armsimd32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      /*mr=*/1, 2 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckARMV6);
  }
  static void qs8_qc8w_gemm_2x2c4__armsimd32(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2c4__armsimd32,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2c4__armsimd32,
      xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params,
      /*mr=*/2, 2 /*nr=*/, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckARMV6);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x1c4__armsimd32);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x1c4__armsimd32);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x2c4__armsimd32);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x2c4__armsimd32);
#endif  // XNN_ARCH_ARM

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qs8_qc8w_gemm_1x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/1, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_2x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/2, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_3x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_4x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_5x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_6x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_7x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/7, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_8x16c4__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c4__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c4__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/8, /*nr=*/16, /*log2_kr=*/2, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_1x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/1, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_2x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/2, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_3x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_4x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_5x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_6x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_7x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/7, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_8x16c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params,
      /*mr=*/8, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_1x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/1, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_2x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/2, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_3x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_4x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_5x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_6x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_7x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/7, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_8x8c8__avx512vnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avx512vnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx512vnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/8, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512VNNI);
  }
  static void qs8_qc8w_gemm_1x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/1, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_2x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/2, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_3x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_4x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_5x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_6x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_7x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/7, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_8x8c8__avxvnni(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x8c8__avxvnni,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avxvnni,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params,
      /*mr=*/8, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVXVNNI);
  }
  static void qs8_qc8w_gemm_1x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/1, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_2x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/2, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_3x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_4x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_5x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_5x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_6x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_6x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_7x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_7x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/7, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_8x16c8__avx512skx(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_8x16c8__avx512skx,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x16c8__avx512skx,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params,
      /*mr=*/8, /*nr=*/16, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX512F);
  }
  static void qs8_qc8w_gemm_2x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__avx2,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
      /*mr=*/2, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_qc8w_gemm_3x8c8__avx2(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x8c8__avx2,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c8__avx2,
      xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params,
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX2);
  }
  static void qs8_qc8w_gemm_2x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_2x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_3x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_3x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_4x4c2__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_4x4c2__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_2x4c2s4__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_2x4c2s4__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_3x4c2s4__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_3x4c2s4__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_4x4c2s4__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_4x4c2s4__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_2x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_3x4c8__xop_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__xop_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__xop_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_2x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_3x4c8__xop_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__xop_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__xop_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckXOP);
  }
  static void qs8_qc8w_gemm_2x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_2x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_3x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_3x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_4x4c2__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_4x4c2__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_2x4c2s4__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_2x4c2s4__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_3x4c2s4__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_3x4c2s4__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_4x4c2s4__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_4x4c2s4__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_2x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_2x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_3x4c8__avx_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__avx_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_3x4c8__avx_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__avx_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__avx_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckAVX);
  }
  static void qs8_qc8w_gemm_2x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_2x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_3x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_3x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_4x4c2__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_4x4c2__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_2x4c2s4__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_2x4c2s4__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_3x4c2s4__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_3x4c2s4__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_4x4c2s4__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_4x4c2s4__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_2x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_2x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_3x4c8__sse41_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_3x4c8__sse41_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse41_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse41_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0,
      benchmark::utils::CheckSSE41);
  }
  static void qs8_qc8w_gemm_2x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_2x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_3x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_3x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_4x4c2__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_4x4c2__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_2x4c2s4__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_2x4c2s4__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_3x4c2s4__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_3x4c2s4__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_4x4c2s4__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_4x4c2s4__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_2x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_2x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_3x4c8__sse2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0);
  }
  static void qs8_qc8w_gemm_3x4c8__sse2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__sse2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__sse2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3, /*log2_sr=*/0);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_5x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_7x16c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x16c8__avx512vnni);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_5x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_7x8c8__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x8c8__avx512vnni);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_5x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_7x8c8__avxvnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x8c8__avxvnni);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_5x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_7x16c4__avx512vnni);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x16c4__avx512vnni);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_1x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_5x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_6x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_7x16c8__avx512skx);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_8x16c8__avx512skx);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x8c8__avx2);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x8c8__avx2);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__xop_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__xop_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__xop_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__xop_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__xop_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__avx_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__avx_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__avx_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__avx_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__avx_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__sse41_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__sse41_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__sse41_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__sse41_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__sse41_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__sse2_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__sse2_ld128);

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__sse2_ld128);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__sse2_ld64);
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__sse2_ld128);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_qc8w_gemm_2x4c16__wasmsdot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c16__wasmsdot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/4, /*log2_sr=*/0,
      benchmark::utils::CheckWAsmSDOT);
  }
  static void qs8_qc8w_gemm_3x4c16__wasmsdot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c16__wasmsdot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/4, /*log2_sr=*/0,
      benchmark::utils::CheckWAsmSDOT);
  }
  static void qs8_qc8w_gemm_4x4c16__wasmsdot(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c16__wasmsdot,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c16__wasmsdot,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/4, /*log2_sr=*/0,
      benchmark::utils::CheckWAsmSDOT);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c16__wasmsdot)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c16__wasmsdot)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c16__wasmsdot)
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_qc8w_gemm_2x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1);
  }
  static void qs8_qc8w_gemm_2x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1);
  }
  static void qs8_qc8w_gemm_3x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1);
  }
  static void qs8_qc8w_gemm_3x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1);
  }
  static void qs8_qc8w_gemm_4x4c2__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1);
  }
  static void qs8_qc8w_gemm_4x4c2__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1);
  }
  static void qs8_qc8w_gemm_2x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_2x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_3x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_3x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_4x4c2s4__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_4x4c2s4__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/1, /*log2_sr=*/2);
  }
  static void qs8_qc8w_gemm_2x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3);
  }
  static void qs8_qc8w_gemm_2x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/2, /*nr=*/4, /*log2_kr=*/3);
  }
  static void qs8_qc8w_gemm_3x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3);
  }
  static void qs8_qc8w_gemm_3x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/3, /*nr=*/4, /*log2_kr=*/3);
  }
  static void qs8_qc8w_gemm_4x4c8__wasmsimd_dot16x2_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/3);
  }
  static void qs8_qc8w_gemm_4x4c8__wasmsimd_dot16x2_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128,
      xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params,
      /*mr=*/4, /*nr=*/4, /*log2_kr=*/3);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2__wasmsimd_dot16x2_ld128)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2__wasmsimd_dot16x2_ld128)

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c2s4__wasmsimd_dot16x2_ld128)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c2s4__wasmsimd_dot16x2_ld128)

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4c8__wasmsimd_dot16x2_ld128)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4c8__wasmsimd_dot16x2_ld128)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c8__wasmsimd_dot16x2_ld64)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4c8__wasmsimd_dot16x2_ld128)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void qs8_qc8w_gemm_2x2__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__wasm_fmagic,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
      /*mr=*/2, /*nr=*/2);
  }
  static void qs8_qc8w_gemm_3x2__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__wasm_fmagic,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
      /*mr=*/3, /*nr=*/2);
  }
  static void qs8_qc8w_gemm_4x2__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__wasm_fmagic,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__wasm_fmagic,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
      /*mr=*/4, /*nr=*/2);
  }
  static void qs8_qc8w_gemm_2x4__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__wasm_fmagic,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
      /*mr=*/2, /*nr=*/4);
  }
  static void qs8_qc8w_gemm_3x4__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__wasm_fmagic,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
      /*mr=*/3, /*nr=*/4);
  }
  static void qs8_qc8w_gemm_4x4__wasm_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__wasm_fmagic,
      xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__wasm_fmagic,
      xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
      /*mr=*/4, /*nr=*/4);
  }

  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x2__wasm_fmagic)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x2__wasm_fmagic)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x2__wasm_fmagic)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4__wasm_fmagic)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4__wasm_fmagic)
  BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4__wasm_fmagic)
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


static void qs8_qc8w_gemm_2x2__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_fmagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
    /*mr=*/2, /*nr=*/2);
}
static void qs8_qc8w_gemm_3x2__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__scalar_fmagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
    /*mr=*/3, /*nr=*/2);
}
static void qs8_qc8w_gemm_4x2__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_fmagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
    /*mr=*/4, /*nr=*/2);
}
static void qs8_qc8w_gemm_2x4__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_fmagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
    /*mr=*/2, /*nr=*/4);
}
static void qs8_qc8w_gemm_3x4__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_fmagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
    /*mr=*/3, /*nr=*/4);
}
static void qs8_qc8w_gemm_4x4__scalar_fmagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_fmagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_fmagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params,
    /*mr=*/4, /*nr=*/4);
}

static void qs8_qc8w_gemm_2x2__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_imagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
    /*mr=*/2, /*nr=*/2);
}
static void qs8_qc8w_gemm_3x2__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__scalar_imagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
    /*mr=*/3, /*nr=*/2);
}
static void qs8_qc8w_gemm_4x2__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_imagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
    /*mr=*/4, /*nr=*/2);
}
static void qs8_qc8w_gemm_2x4__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_imagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
    /*mr=*/2, /*nr=*/4);
}
static void qs8_qc8w_gemm_3x4__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_imagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
    /*mr=*/3, /*nr=*/4);
}
static void qs8_qc8w_gemm_4x4__scalar_imagic(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_imagic,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_imagic,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params,
    /*mr=*/4, /*nr=*/4);
}

static void qs8_qc8w_gemm_2x2__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x2__scalar_lrintf,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
    /*mr=*/2, /*nr=*/2);
}
static void qs8_qc8w_gemm_3x2__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x2__scalar_lrintf,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
    /*mr=*/3, /*nr=*/2);
}
static void qs8_qc8w_gemm_4x2__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_lrintf,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x2__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
    /*mr=*/4, /*nr=*/2);
}
static void qs8_qc8w_gemm_2x4__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x4__scalar_lrintf,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
    /*mr=*/2, /*nr=*/4);
}
static void qs8_qc8w_gemm_3x4__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_3x4__scalar_lrintf,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
    /*mr=*/3, /*nr=*/4);
}
static void qs8_qc8w_gemm_4x4__scalar_lrintf(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x4__scalar_lrintf,
    xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x4__scalar_lrintf,
    xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params,
    /*mr=*/4, /*nr=*/4);
}

BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x2__scalar_fmagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x2__scalar_fmagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x2__scalar_fmagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4__scalar_fmagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4__scalar_fmagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4__scalar_fmagic)

BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x2__scalar_imagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x2__scalar_imagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x2__scalar_imagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4__scalar_imagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4__scalar_imagic)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4__scalar_imagic)

BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x2__scalar_lrintf)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x2__scalar_lrintf)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x2__scalar_lrintf)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_2x4__scalar_lrintf)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_3x4__scalar_lrintf)
BENCHMARK_QS8_END2END(qs8_qc8w_gemm_4x4__scalar_lrintf)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
