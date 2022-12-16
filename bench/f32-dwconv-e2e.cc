// Copyright 2019 Google LLC
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
  xnn_f32_dwconv_minmax_unipass_ukernel_fn dwconv_minmax,
  xnn_f32_dwconv_unipass_ukernel_fn dwconv,
  xnn_init_f32_minmax_params_fn init_params,
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

  // Override microkernels chosen in xnn_initialize
  for (size_t i = 0; i < XNN_MAX_F32_DWCONV_UKERNELS; i++) {
    // Replace only the microkernel with the matching kernel size.
    if (xnn_params.f32.dwconv[i].primary_tile == primary_tile) {
      std::memset(&xnn_params.f32.dwconv[i], 0, sizeof(xnn_params.f32.dwconv[i]));

      // Note: do not directly assign to xnn_params.f32.dwconv[i] because it breaks older gcc.
      xnn_params.f32.dwconv[i].minmax.unipass = xnn_dwconv_unipass_ukernel_fn(dwconv_minmax);
      xnn_params.f32.dwconv[i].linear.unipass = xnn_dwconv_unipass_ukernel_fn(dwconv);
      xnn_params.f32.dwconv[i].channel_tile = channel_tile;
      xnn_params.f32.dwconv[i].primary_tile = primary_tile;
      xnn_params.f32.dwconv[i].last_tile = 0;
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
  static void f32_dwconv_9p4c__asm_aarch64_neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p4c__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__asm_aarch64_neonfma_cortex_a55,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__asm_aarch64_neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__asm_aarch64_neonfma_cortex_a55);
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_dwconv_9p4c__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neon,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_9p4c__neon_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neon_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_9p8c__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neon,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_9p8c__neon_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neon_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_9p16c__neon(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neon,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_9p16c__neon_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neon_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  static void f32_dwconv_9p4c__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_9p4c__neonfma_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__neonfma_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_9p8c__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_9p8c__neonfma_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__neonfma_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_9p16c__neonfma(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  static void f32_dwconv_9p16c__neonfma_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__neonfma_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__neonfma_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__neonfma);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__neonfma_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__neon);
  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__neon_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__neon);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__neon_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__neon);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__neon_acc2);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_dwconv_9p4c__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__sse,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_sse_params,
      4 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_9p4c__sse_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__sse_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_sse_params,
      4 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_9p8c__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__sse,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_sse_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void f32_dwconv_9p8c__sse_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__sse_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_sse_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p8c__avx(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__avx,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_9p8c__avx_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__avx_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_9p16c__avx(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__avx,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void f32_dwconv_9p16c__avx_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__avx_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }

  static void f32_dwconv_9p8c__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__fma3,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_9p8c__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__fma3_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_9p16c__fma3(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__fma3,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }
  static void f32_dwconv_9p16c__fma3_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__fma3_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_avx_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckFMA3);
  }

  static void f32_dwconv_9p16c__avx512f(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_9p16c__avx512f_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p16c__avx512f_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_9p32c__avx512f(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p32c__avx512f,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }
  static void f32_dwconv_9p32c__avx512f_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p32c__avx512f_acc2,
      nullptr /* dwconv */,
      xnn_init_f32_minmax_scalar_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512F);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__avx512f);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__avx512f_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p32c__avx512f);
  BENCHMARK_FP32_END2END(f32_dwconv_9p32c__avx512f_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__fma3);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__fma3_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__fma3);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__fma3_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__avx);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__avx_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__avx);
  BENCHMARK_FP32_END2END(f32_dwconv_9p16c__avx_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__sse);
  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__sse_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__sse);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__sse_acc2);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_dwconv_9p4c__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_arm,
      xnn_f32_dwconv_ukernel_9p4c__wasmsimd,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p4c__wasmsimd_arm_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_arm_acc2,
      xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p8c__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_arm,
      xnn_f32_dwconv_ukernel_9p8c__wasmsimd,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p8c__wasmsimd_arm_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_arm_acc2,
      xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p4c__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_x86,
      xnn_f32_dwconv_ukernel_9p4c__wasmsimd,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p4c__wasmsimd_x86_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p4c__wasmsimd_x86_acc2,
      xnn_f32_dwconv_ukernel_9p4c__wasmsimd_acc2,
      xnn_init_f32_minmax_scalar_params,
      4 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p8c__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_x86,
      xnn_f32_dwconv_ukernel_9p8c__wasmsimd,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  static void f32_dwconv_9p8c__wasmsimd_x86_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_f32_dwconv_minmax_ukernel_9p8c__wasmsimd_x86_acc2,
      xnn_f32_dwconv_ukernel_9p8c__wasmsimd_acc2,
      xnn_init_f32_minmax_scalar_params,
      8 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__wasmsimd_arm_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__wasmsimd_arm);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__wasmsimd_arm_acc2);

  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_dwconv_9p4c__wasmsimd_x86_acc2);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__wasmsimd_x86);
  BENCHMARK_FP32_END2END(f32_dwconv_9p8c__wasmsimd_x86_acc2);
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

static void f32_dwconv_9p1c__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_9p1c__scalar,
    xnn_f32_dwconv_ukernel_9p1c__scalar,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 9 /* primary tile */);
}

static void f32_dwconv_9p1c__scalar_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_9p1c__scalar_acc2,
    xnn_f32_dwconv_ukernel_9p1c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    1 /* channel tile */, 9 /* primary tile */);
}

static void f32_dwconv_9p2c__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_9p2c__scalar,
    xnn_f32_dwconv_ukernel_9p2c__scalar,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 9 /* primary tile */);
}

static void f32_dwconv_9p2c__scalar_acc2(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_f32_dwconv_minmax_ukernel_9p2c__scalar_acc2,
    xnn_f32_dwconv_ukernel_9p2c__scalar_acc2,
    xnn_init_f32_minmax_scalar_params,
    2 /* channel tile */, 9 /* primary tile */);
}

BENCHMARK_FP32_END2END(f32_dwconv_9p1c__scalar);
BENCHMARK_FP32_END2END(f32_dwconv_9p1c__scalar_acc2);
BENCHMARK_FP32_END2END(f32_dwconv_9p2c__scalar);
BENCHMARK_FP32_END2END(f32_dwconv_9p2c__scalar_acc2);

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
