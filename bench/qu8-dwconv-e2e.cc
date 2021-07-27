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
#include <xnnpack/dwconv.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


static void DWConvEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_qu8_dwconv_minmax_unipass_ukernel_function dwconv,
  xnn_init_qu8_conv_minmax_params_fn init_params,
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
  for (size_t i = 0; i < XNN_MAX_QU8_DWCONV_UKERNELS; i++) {
    // Replace only the microkernel the matching kernel size.
    if (xnn_params.qu8.dwconv[i].primary_tile == primary_tile) {
      // Note: do not directly assign to xnn_params.qu8.dwconv[i] because it breaks older gcc.
      xnn_params.qu8.dwconv[i].minmax.unipass = xnn_dwconv_unipass_ukernel_function(dwconv);
      xnn_params.qu8.dwconv[i].channel_tile = channel_tile;
      xnn_params.qu8.dwconv[i].primary_tile = primary_tile;
      xnn_params.qu8.dwconv[i].incremental_tile = 0;
      xnn_params.qu8.dwconv[i].init.qu8 = init_params;
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


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void qu8_dwconv_up8x9__neon_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }
  static void qu8_dwconv_up16x9__neon_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16,
      xnn_init_qu8_conv_minmax_rndnu_neon_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckNEON);
  }

  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__neon_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__neon_mul16);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void qu8_dwconv_up16x9__avx512skx_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx512skx_mul32,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512SKX);
  }
  static void qu8_dwconv_up32x9__avx512skx_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32,
      xnn_init_qu8_conv_minmax_fp32_avx512_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX512SKX);
  }
  static void qu8_dwconv_up8x9__avx2_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__avx2_mul32,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qu8_dwconv_up16x9__avx2_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qu8_dwconv_up32x9__avx2_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul32,
      xnn_init_qu8_conv_minmax_fp32_avx2_params,
      32 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX2);
  }
  static void qu8_dwconv_up8x9__avx_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul16,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qu8_dwconv_up16x9__avx_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qu8_dwconv_up8x9__avx_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul32,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qu8_dwconv_up16x9__avx_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul32,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckAVX);
  }
  static void qu8_dwconv_up8x9__sse41_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qu8_dwconv_up16x9__sse41_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul16,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qu8_dwconv_up8x9__sse41_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul32,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qu8_dwconv_up16x9__sse41_mul32(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul32,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */, benchmark::utils::CheckSSE41);
  }
  static void qu8_dwconv_up8x9__sse2_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void qu8_dwconv_up16x9__sse2_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__sse2_mul16,
      xnn_init_qu8_conv_minmax_fp32_sse2_params,
      16 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__avx512skx_mul32);
  BENCHMARK_QU8_END2END(qu8_dwconv_up32x9__avx512skx_mul32);

  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__avx2_mul32);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__avx2_mul32);
  BENCHMARK_QU8_END2END(qu8_dwconv_up32x9__avx2_mul32);

  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__avx_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__avx_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__avx_mul32);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__avx_mul32);

  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__sse41_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__sse41_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__sse41_mul32);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__sse41_mul32);

  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__sse2_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__sse2_mul16);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD
  static void qu8_dwconv_up8x9__wasmsimd_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      8 /* channel tile */, 9 /* primary tile */);
  }
  static void qu8_dwconv_up16x9__wasmsimd_mul16(benchmark::State& state, models::ExecutionPlanFactory model) {
    DWConvEnd2EndBenchmark(state, model,
      xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16,
      xnn_init_qu8_conv_minmax_fp32_wasmsimd_params,
      16 /* channel tile */, 9 /* primary tile */);
  }

  BENCHMARK_QU8_END2END(qu8_dwconv_up8x9__wasmsimd_mul16);
  BENCHMARK_QU8_END2END(qu8_dwconv_up16x9__wasmsimd_mul16);
#endif  // XNN_ARCH_WASMSIMD


static void qu8_dwconv_up1x9__scalar_lrint(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__scalar_lrint,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrint_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void qu8_dwconv_up2x9__scalar_lrint(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrint,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrint_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void qu8_dwconv_up4x9__scalar_lrint(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_qu8_dwconv_minmax_fp32_ukernel_up4x9__scalar_lrint,
    xnn_init_qu8_conv_minmax_fp32_scalar_lrint_params,
    4 /* channel tile */, 9 /* primary tile */);
}
static void qu8_dwconv_up1x9__scalar_magic(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__scalar_magic,
    xnn_init_qu8_conv_minmax_fp32_scalar_magic_params,
    1 /* channel tile */, 9 /* primary tile */);
}
static void qu8_dwconv_up2x9__scalar_magic(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_magic,
    xnn_init_qu8_conv_minmax_fp32_scalar_magic_params,
    2 /* channel tile */, 9 /* primary tile */);
}
static void qu8_dwconv_up4x9__scalar_magic(benchmark::State& state, models::ExecutionPlanFactory model) {
  DWConvEnd2EndBenchmark(state, model,
    xnn_qu8_dwconv_minmax_fp32_ukernel_up4x9__scalar_magic,
    xnn_init_qu8_conv_minmax_fp32_scalar_magic_params,
    4 /* channel tile */, 9 /* primary tile */);
}

BENCHMARK_QU8_END2END(qu8_dwconv_up1x9__scalar_lrint);
BENCHMARK_QU8_END2END(qu8_dwconv_up2x9__scalar_lrint);
BENCHMARK_QU8_END2END(qu8_dwconv_up4x9__scalar_lrint);

BENCHMARK_QU8_END2END(qu8_dwconv_up1x9__scalar_magic);
BENCHMARK_QU8_END2END(qu8_dwconv_up2x9__scalar_magic);
BENCHMARK_QU8_END2END(qu8_dwconv_up4x9__scalar_magic);


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
