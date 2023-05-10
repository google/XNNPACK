// Copyright 2023 Google LLC
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

#include "bench/end2end.h"
#include "bench/utils.h"
#include <benchmark/benchmark.h>

#include <xnnpack.h>
#include <xnnpack/config.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/models.h>


static void DWConvEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f16_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
  xnn_init_f16_minmax_params_fn init_params,
  uint8_t channel_tile, uint8_t primary_tile,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  struct xnn_dwconv_config* dwconv_config = xnn_init_f16_dwconv_config();
  if (dwconv_config == nullptr) {
    state.SkipWithError("hardware does not support F16 DWCONV");
    return;
  }

  // Save dwconv_config so that we can modify it for the benchmark and later restore it.
  struct xnn_dwconv_config saved_dwconv_params[XNN_MAX_F16_DWCONV_UKERNELS];
  memcpy(saved_dwconv_params, dwconv_config, sizeof(saved_dwconv_params));

  // Override microkernels chosen in xnn_initialize
  for (size_t i = 0; i < XNN_MAX_F16_DWCONV_UKERNELS; i++) {
    // Replace only the microkernel with the matching kernel size.
    if (dwconv_config[i].primary_tile == primary_tile) {
      std::memset(&dwconv_config[i], 0, sizeof(dwconv_config[i]));

      // Note: do not directly assign to dwconv_config[i] because it breaks older gcc.
      dwconv_config[i].minmax.unipass = xnn_dwconv_unipass_ukernel_fn(dwconv_minmax);
      dwconv_config[i].channel_tile = channel_tile;
      dwconv_config[i].channel_subtile = channel_tile;
      dwconv_config[i].channel_round = 1;
      dwconv_config[i].primary_tile = primary_tile;
      dwconv_config[i].init.f16 = init_params;
      break;
    }
  }

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

  // Restore dwconv_config to original state as defined in init.c.
  memcpy(dwconv_config, saved_dwconv_params, sizeof(saved_dwconv_params));
}

static void DWConvEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f16_dwconv_minmax_multipass_ukernel_fn dwconv_minmax,
  xnn_init_f16_minmax_params_fn init_params,
  uint8_t channel_tile, uint8_t channel_subtile, uint8_t channel_round,
  uint8_t primary_tile, uint8_t middle_tile, uint8_t last_tile,
  uint8_t primary_tile_to_replace,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  struct xnn_dwconv_config* dwconv_config = xnn_init_f16_dwconv_config();
  if (dwconv_config == nullptr) {
    state.SkipWithError("failed to initialize f16 DWCONV config");
    return;
  }

  // Save dwconv_convig so that we can modify it for the benchmark and later restore it.
  struct xnn_dwconv_config saved_dwconv_params[XNN_MAX_F16_DWCONV_UKERNELS];
  memcpy(saved_dwconv_params, dwconv_config, sizeof(saved_dwconv_params));

  bool found = false;
  for (size_t i = 0; i < XNN_MAX_F16_DWCONV_UKERNELS; i++) {
    if (dwconv_config[i].primary_tile == primary_tile_to_replace) {
      found = true;
    } else if (dwconv_config[i].last_tile != 0) {
      // Found a multipass microkernel, replace it.
      found = true;
    }
  }

  if (!found) {
    state.SkipWithError("can't replace with multipass");
    return;
  }

  // Override microkernels chosen in xnn_initialize
  for (size_t i = 0; i < XNN_MAX_F16_DWCONV_UKERNELS; i++) {
    // Replace only the microkernel with the matching kernel size.
    if (dwconv_config[i].primary_tile == primary_tile_to_replace ||
        dwconv_config[i].last_tile != 0) {
      // Replace either when the primary_tile_to_replace matches, or replace the
      // first multipass dwconv microkernel we find.
      // TODO(zhin): support specifying target multipass dwconv to replace.
      std::memset(&dwconv_config[i], 0, sizeof(dwconv_config[i]));

      // Note: do not directly assign to dwconv_config[i] because it breaks older gcc.
      dwconv_config[i].minmax.multipass = xnn_dwconv_multipass_ukernel_fn(dwconv_minmax);
      dwconv_config[i].channel_tile = channel_tile;
      dwconv_config[i].channel_subtile = channel_subtile;
      dwconv_config[i].channel_round = channel_round;
      dwconv_config[i].primary_tile = primary_tile;
      dwconv_config[i].middle_tile = middle_tile;
      dwconv_config[i].last_tile = last_tile;
      dwconv_config[i].init.f16 = init_params;
      break;
    }
  }

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

  memcpy(dwconv_config, saved_dwconv_params, sizeof(saved_dwconv_params));
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  static void f16_dwconv_4p8c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_4p8c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*primary_tile=*/4, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_4p8c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_4p8c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*primary_tile=*/4, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_4p16c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*primary_tile=*/4, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_4p16c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_4p16c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*primary_tile=*/4, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_4p32c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_4p32c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*primary_tile=*/4, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_4p32c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_4p32c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*primary_tile=*/4, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_9p8c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*primary_tile=*/9, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_9p8c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_9p8c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*primary_tile=*/9, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_9p16c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*primary_tile=*/9, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_9p16c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_9p16c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*primary_tile=*/9, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_9p32c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_9p32c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*primary_tile=*/9, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_9p32c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_9p32c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*primary_tile=*/9, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_25p8c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_25p8c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p8c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_25p16c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_25p16c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_25p32c__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p32c__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_25p32c__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p32c__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_5f5m5l8c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l8c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l16c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l16c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l32c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_5f5m5l32c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_6f6m7l8c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l8c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l16c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l16c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l32c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_6f6m7l32c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  static void f16_dwconv_8f8m9l8c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l8c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l16c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l16c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l32c8s4r__neonfp16arith(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }
  static void f16_dwconv_8f8m9l32c8s4r__neonfp16arith_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__neonfp16arith_acc2, xnn_init_f16_minmax_fp16arith_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckNEONFP16ARITH);
  }

  BENCHMARK_FP16_END2END(f16_dwconv_4p8c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_4p8c__neonfp16arith_acc2);
  BENCHMARK_FP16_END2END(f16_dwconv_4p16c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_4p16c__neonfp16arith_acc2);
  BENCHMARK_FP16_END2END(f16_dwconv_4p32c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_4p32c__neonfp16arith_acc2);

  BENCHMARK_FP16_END2END(f16_dwconv_9p8c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_9p8c__neonfp16arith_acc2);
  BENCHMARK_FP16_END2END(f16_dwconv_9p16c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_9p16c__neonfp16arith_acc2);
  BENCHMARK_FP16_END2END(f16_dwconv_9p32c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_9p32c__neonfp16arith_acc2);

  BENCHMARK_FP16_END2END(f16_dwconv_25p8c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_25p8c__neonfp16arith_acc2);
  BENCHMARK_FP16_END2END(f16_dwconv_25p16c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_25p16c__neonfp16arith_acc2);
  BENCHMARK_FP16_END2END(f16_dwconv_25p32c__neonfp16arith);
  BENCHMARK_FP16_END2END(f16_dwconv_25p32c__neonfp16arith_acc2);

  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l8c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l8c8s4r__neonfp16arith_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l16c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l16c8s4r__neonfp16arith_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l32c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l32c8s4r__neonfp16arith_acc2)

  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l8c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l8c8s4r__neonfp16arith_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l16c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l16c8s4r__neonfp16arith_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l32c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l32c8s4r__neonfp16arith_acc2)

  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l8c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l8c8s4r__neonfp16arith_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l16c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l16c8s4r__neonfp16arith_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l32c8s4r__neonfp16arith)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l32c8s4r__neonfp16arith_acc2)

#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f16_dwconv_25p8c__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p8c__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p8c__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p8c__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p16c__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p16c__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p16c__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p16c__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p32c__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p32c__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_25p32c__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_25p32c__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*primary_tile=*/25, /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  static void f16_dwconv_5f5m5l8c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l8c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l16c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l16c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l16c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l32c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_5f5m5l32c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_5f5m5l32c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/5, /*middle_tile=*/5, /*last_tile=*/5,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  static void f16_dwconv_6f6m7l8c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l8c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l8c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l16c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l16c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l16c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l32c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_6f6m7l32c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_6f6m7l32c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/6, /*middle_tile=*/6, /*last_tile=*/7,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  static void f16_dwconv_8f8m9l8c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l8c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/8, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l16c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l16c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l16c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/16, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l32c8s4r__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }
  static void f16_dwconv_8f8m9l32c8s4r__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(
      state, model,
      xnn_f16_dwconv_minmax_ukernel_8f8m9l32c8s4r__fma3_acc2, xnn_init_f16_minmax_avx_params,
      /*channel_tile=*/32, /*channel_subtile=*/8, /*channel_round=*/4,
      /*primary_tile=*/8, /*middle_tile=*/8, /*last_tile=*/9,
      /*primary_tile_to_replace=*/25,
      /*isa_check=*/benchmark::utils::CheckFMA3);
  }

  BENCHMARK_FP16_END2END(f16_dwconv_25p8c__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_25p8c__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_25p16c__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_25p16c__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_25p32c__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_25p32c__fma3_acc2)

  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l8c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l8c8s4r__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l16c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l16c8s4r__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l32c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_5f5m5l32c8s4r__fma3_acc2)

  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l8c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l8c8s4r__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l16c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l16c8s4r__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l32c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_6f6m7l32c8s4r__fma3_acc2)

  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l8c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l8c8s4r__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l16c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l16c8s4r__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l32c8s4r__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_8f8m9l32c8s4r__fma3_acc2)

#endif // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
