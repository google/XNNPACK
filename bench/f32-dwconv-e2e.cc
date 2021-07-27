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
#include <xnnpack/dwconv.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


static void DWConvEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f32_dwconv_minmax_unipass_ukernel_function dwconv,
  xnn_init_f32_minmax_params_fn init_params,
  uint8_t channel_tile, uint8_t primary_tile,
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
  for (size_t i = 0; i < XNN_MAX_F32_DWCONV_UKERNELS; i++) {
    // Replace only the microkernel the matching kernel size.
    if (xnn_params.f32.dwconv[i].primary_tile == primary_tile) {
      // Note: do not directly assign to xnn_params.f32.dwconv[i] because it breaks older gcc.
      xnn_params.f32.dwconv[i].minmax.unipass = xnn_dwconv_unipass_ukernel_function(dwconv);
      xnn_params.f32.dwconv[i].channel_tile = channel_tile;
      xnn_params.f32.dwconv[i].primary_tile = primary_tile;
      xnn_params.f32.dwconv[i].incremental_tile = 0;
      xnn_params.f32.dwconv[i].init.f32 = init_params;
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
}

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_dwconv_up4x9__aarch64_neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up4x9__aarch64_neonfma_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__aarch64_neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__aarch64_neonfma_cortex_a55);
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_dwconv_up4x9__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__neon,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_up4x9__neon_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_up8x9__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__neon,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_up8x9__neon_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_up4x9__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_up4x9__neonfma_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_up8x9__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_up8x9__neonfma_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__neon);
  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__neon_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__neon);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__neon_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__neonfma_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__neonfma_acc2);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_dwconv_up4x9__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__sse,
      xnn_init_f32_minmax_sse_params,
      4 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_up4x9__sse_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2,
      xnn_init_f32_minmax_sse_params,
      4 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_up8x9__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__sse,
      xnn_init_f32_minmax_sse_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_up8x9__sse_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2,
      xnn_init_f32_minmax_sse_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up8x9__avx(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__avx,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_up8x9__avx_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_up16x9__avx(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up16x9__avx,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_up16x9__avx_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }

  static void f32_dwconv_up8x9__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__fma3,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_up8x9__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_up16x9__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up16x9__fma3,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_up16x9__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }

  static void f32_dwconv_up16x9__avx512f(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_up16x9__avx512f_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_up32x9__avx512f(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f,
      xnn_init_f32_minmax_scalar_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_up32x9__avx512f_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2,
      xnn_init_f32_minmax_scalar_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_up16x9__avx512f);
  BENCHMARK_FP32_END2END(f32_dwconv_up16x9__avx512f_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up32x9__avx512f);
  BENCHMARK_FP32_END2END(f32_dwconv_up32x9__avx512f_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__fma3);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__fma3_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up16x9__fma3);
  BENCHMARK_FP32_END2END(f32_dwconv_up16x9__fma3_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__avx);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__avx_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up16x9__avx);
  BENCHMARK_FP32_END2END(f32_dwconv_up16x9__avx_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__sse);
  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__sse_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__sse);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__sse_acc2);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD
  static void f32_dwconv_up4x9__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up4x9__wasmsimd_arm_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up8x9__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up8x9__wasmsimd_arm_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up4x9__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up4x9__wasmsimd_x86_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up8x9__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_up8x9__wasmsimd_x86_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__wasmsimd_arm_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__wasmsimd_arm_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_dwconv_up4x9__wasmsimd_x86_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_dwconv_up8x9__wasmsimd_x86_acc2);
#endif  // XNN_ARCH_WASMSIMD

static void f32_dwconv_up1x9__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_up1x9__scalar,
      xnn_init_f32_minmax_scalar_params,
      1 /* channel tile */, 9 /* primary tile */);
}

static void f32_dwconv_up1x9__scalar_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2,
      xnn_init_f32_minmax_scalar_params,
      1 /* channel tile */, 9 /* primary tile */);
}

static void f32_dwconv_up2x9__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_up2x9__scalar,
      xnn_init_f32_minmax_scalar_params,
      2 /* channel tile */, 9 /* primary tile */);
}

static void f32_dwconv_up2x9__scalar_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2,
      xnn_init_f32_minmax_scalar_params,
      2 /* channel tile */, 9 /* primary tile */);
}

BENCHMARK_FP32_END2END(f32_dwconv_up1x9__scalar);
BENCHMARK_FP32_END2END(f32_dwconv_up1x9__scalar_acc2);
BENCHMARK_FP32_END2END(f32_dwconv_up2x9__scalar);
BENCHMARK_FP32_END2END(f32_dwconv_up2x9__scalar_acc2);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
