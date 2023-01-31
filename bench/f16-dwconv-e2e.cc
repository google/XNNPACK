// Copyright 2023 Google LLC
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
#include <xnnpack/dwconv.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>


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

  // Save xnn_params.f16.dwconv so that we can modify it for the benchmark and later restore it.
  struct dwconv_parameters saved_dwconv_params[XNN_MAX_F16_DWCONV_UKERNELS];
  static_assert(sizeof(saved_dwconv_params) == sizeof(xnn_params.f16.dwconv), "size of dwconv params must match");
  memcpy(saved_dwconv_params, xnn_params.f16.dwconv, sizeof(saved_dwconv_params));

  // Override microkernels chosen in xnn_initialize
  for (size_t i = 0; i < XNN_MAX_F16_DWCONV_UKERNELS; i++) {
    // Replace only the microkernel with the matching kernel size.
    if (xnn_params.f16.dwconv[i].primary_tile == primary_tile) {
      std::memset(&xnn_params.f16.dwconv[i], 0, sizeof(xnn_params.f16.dwconv[i]));

      // Note: do not directly assign to xnn_params.f16.dwconv[i] because it breaks older gcc.
      xnn_params.f16.dwconv[i].minmax.unipass = xnn_dwconv_unipass_ukernel_fn(dwconv_minmax);
      xnn_params.f16.dwconv[i].channel_tile = channel_tile;
      xnn_params.f16.dwconv[i].primary_tile = primary_tile;
      xnn_params.f16.dwconv[i].last_tile = 0;
      xnn_params.f16.dwconv[i].init.f16 = init_params;
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

  // Restore xnn_params.f16.dwconv to original state as defined in init.c.
  memcpy(xnn_params.f16.dwconv, saved_dwconv_params, sizeof(saved_dwconv_params));
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

  BENCHMARK_FP16_END2END(f16_dwconv_25p8c__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_25p8c__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_25p16c__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_25p16c__fma3_acc2)
  BENCHMARK_FP16_END2END(f16_dwconv_25p32c__fma3)
  BENCHMARK_FP16_END2END(f16_dwconv_25p32c__fma3_acc2)

#endif // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
