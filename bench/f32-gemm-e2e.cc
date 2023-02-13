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
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>


static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_f32_gemm_minmax_ukernel_fn gemm_minmax,
  xnn_f32_igemm_minmax_ukernel_fn igemm_minmax,
  xnn_f32_gemm_minmax_ukernel_fn gemm1_minmax,
  xnn_f32_igemm_minmax_ukernel_fn igemm1_minmax,
  xnn_f32_gemm_relu_ukernel_fn gemm_relu,
  xnn_f32_igemm_relu_ukernel_fn igemm_relu,
  xnn_f32_gemm_relu_ukernel_fn gemm1_relu,
  xnn_f32_igemm_relu_ukernel_fn igemm1_relu,
  xnn_f32_gemm_ukernel_fn gemm,
  xnn_f32_igemm_ukernel_fn igemm,
  xnn_f32_gemm_ukernel_fn gemm1,
  xnn_f32_igemm_ukernel_fn igemm1,
  xnn_init_f32_minmax_params_fn init_params,
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

  // Override microkernels chosen in xnn_initialize
  // Note: do not directly assign to xnn_params.f32.gemm because it breaks older gcc.
  std::memset(&xnn_params.f32.gemm, 0, sizeof(xnn_params.f32.gemm));
  std::memset(&xnn_params.f32.gemm2, 0, sizeof(xnn_params.f32.gemm2));
  xnn_params.f32.gemm.minmax.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm_minmax));
  xnn_params.f32.gemm.minmax.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm_minmax));
  xnn_params.f32.gemm.minmax.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1_minmax));
  xnn_params.f32.gemm.minmax.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1_minmax));
  xnn_params.f32.gemm.relu.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm_relu));
  xnn_params.f32.gemm.relu.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm_relu));
  xnn_params.f32.gemm.relu.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1_relu));
  xnn_params.f32.gemm.relu.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1_relu));
  xnn_params.f32.gemm.linear.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm));
  xnn_params.f32.gemm.linear.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm));
  xnn_params.f32.gemm.linear.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1));
  xnn_params.f32.gemm.linear.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1));
  xnn_params.f32.gemm.init.f32 = init_params;
  xnn_params.f32.gemm.mr = mr;
  xnn_params.f32.gemm.nr = nr;
  xnn_params.f32.gemm.log2_kr = log2_kr;
  xnn_params.f32.gemm.log2_sr = log2_sr;

  #if XNN_PLATFORM_JIT && XNN_ENABLE_JIT
    // If JIT is enabled, we want to make sure that we are still benchmarking
    // non-JIT microkernels, so nullify the pointers to generators.
    for (size_t i = 0; i < XNN_MAX_MR; i++) {
      xnn_params.f32.gemm.generator.gemm[i] = xnn_init_hmp_gemm_codegen(nullptr);
      xnn_params.f32.gemm.generator.igemm[i] = xnn_init_hmp_igemm_codegen(nullptr);
    }
  #endif  // XNN_PLATFORM_JIT && XNN_ENABLE_JIT

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

#if XNN_PLATFORM_JIT
static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  xnn_jit_gemm_code_generator_fn gemm_generator,
  xnn_jit_gemm_code_generator_fn gemm1_generator,
  xnn_jit_igemm_code_generator_fn igemm_generator,
  xnn_jit_igemm_code_generator_fn igemm1_generator,
  xnn_init_f32_minmax_params_fn init_params,
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

  // Set the microkernels to dummies to ensure we run JIT kernels.
  for (size_t i = 0; i < XNN_MAX_MR; i++) {
    xnn_params.f32.gemm.minmax.gemm[i] = xnn_init_hmp_gemm_ukernel(nullptr);
    xnn_params.f32.gemm.minmax.igemm[i] = xnn_init_hmp_igemm_ukernel(nullptr);
  }
  xnn_params.f32.gemm.init.f32 = init_params;
  xnn_params.f32.gemm.mr = mr;
  xnn_params.f32.gemm.nr = nr;
  xnn_params.f32.gemm.log2_kr = log2_kr;
  xnn_params.f32.gemm.log2_sr = log2_sr;

  xnn_params.f32.gemm.generator.gemm[mr-1] = xnn_init_hmp_gemm_codegen(gemm_generator);
  xnn_params.f32.gemm.generator.gemm[0] = xnn_init_hmp_gemm_codegen(gemm1_generator);
  xnn_params.f32.gemm.generator.igemm[mr-1] = xnn_init_hmp_igemm_codegen(igemm_generator);
  xnn_params.f32.gemm.generator.igemm[0] = xnn_init_hmp_igemm_codegen(igemm1_generator);

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
#endif  // XNN_PLATFORM_JIT

#if XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 2 /* nr */);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_prfm_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 2 /* nr */);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 2 /* nr */);
  }
  static void f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 12 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_prfm_cortex_a75,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x2__aarch64_neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 2 /* nr */);
  }
  static void f32_gemm_6x2__aarch64_neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 2 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 8 /* nr */);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_4x2__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_4x2__asm_aarch64_neonfma_ld64)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a53)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_prfm_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__asm_aarch64_neonfma_prfm_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_prfm_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53);

  BENCHMARK_FP32_END2END(f32_gemm_4x2__aarch64_neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x2__aarch64_neonfma_lane_ld64);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_lane_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_lane_ld128);
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY
  static void f32_gemm_4x8__asm_aarch32_neon_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a7(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a53(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a53,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a55(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a75(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a75,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_prfm_cortex_a75,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 8 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a7);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a55);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_prfm_cortex_a75);
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_4x2__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      4 /* mr */, 2 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_6x2__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_6x2__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_6x2__neon_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      6 /* mr */, 2 /* nr */, 0 /* log2_kr */, 0 /* log2_sr */,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      8 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */,
      benchmark::utils::CheckNEONFMA);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x2__neon_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x2__neon_lane_ld64);

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

#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
  static void jit_f32_gemm_4x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 4 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 4 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_1x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 1 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_2x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 2 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_3x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 3 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_4x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 4 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_5x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 5 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_6x8__aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 6 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_1x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 1 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_2x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 2 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_3x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 3 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_4x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 4 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_5x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 5 /* mr */, 8 /* nr */);
  }
  static void jit_f32_gemm_6x8_6x8__aarch64_neonfma_prfm_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75,
        xnn_init_f32_minmax_scalar_params, 6 /* mr */, 8 /* nr */);
  }

BENCHMARK_FP32_END2END(jit_f32_gemm_4x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_4x8__aarch64_neonfma_prfm_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_1x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_2x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_3x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_4x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_5x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_6x8__aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_1x8__aarch64_neonfma_prfm_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_2x8__aarch64_neonfma_prfm_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_3x8__aarch64_neonfma_prfm_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_4x8__aarch64_neonfma_prfm_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_5x8__aarch64_neonfma_prfm_cortex_a75);
BENCHMARK_FP32_END2END(jit_f32_gemm_6x8_6x8__aarch64_neonfma_prfm_cortex_a75);

#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  static void f32_gemm_4x16__avx512f_broadcast(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_4x16__avx512f_broadcast,
      xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast,
      xnn_f32_igemm_minmax_ukernel_1x16__avx512f_broadcast,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__sse2_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_4x8__sse2_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__sse2_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_5x8__sse2_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse2_dup,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */);
  }

  static void f32_gemm_3x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_3x8__sse_load1,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_4x8__sse_load1,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__sse_load1(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_5x8__sse_load1,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_load1,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_load1,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_3x8__sse_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_4x8__sse_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__sse_dup(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_5x8__sse_dup,
      xnn_f32_gemm_minmax_ukernel_1x8__sse_dup,
      xnn_f32_igemm_minmax_ukernel_1x8__sse_dup,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_3x8s4__sse,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_4x8s4__sse,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_sse_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__sse(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_5x8s4__sse,
      xnn_f32_gemm_minmax_ukernel_1x8s4__sse,
      xnn_f32_igemm_minmax_ukernel_1x8s4__sse,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
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


#if XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_3x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_fma_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_3x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_3x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_4x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_4x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_5x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_5x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_6x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_6x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_3x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_4x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_5x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmrelaxedsimd_fma_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_6x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_f32_igemm_ukernel_1x8__wasmrelaxedsimd_fma_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8s4__wasmrelaxedsimd(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_3x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_3x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_3x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__wasmrelaxedsimd(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_4x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_4x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_4x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__wasmrelaxedsimd(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_5x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_5x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_5x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_6x8s4__wasmrelaxedsimd(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd,
      xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_6x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_6x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_6x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_3x8s4__wasmrelaxedsimd_fma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_3x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__wasmrelaxedsimd_fma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_4x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__wasmrelaxedsimd_fma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_5x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_6x8s4__wasmrelaxedsimd_fma(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd_fma,
      xnn_f32_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_f32_igemm_ukernel_1x8s4__wasmrelaxedsimd_fma,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmrelaxedsimd_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmrelaxedsimd_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmrelaxedsimd_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmrelaxedsimd_loadsplat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmrelaxedsimd_fma_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmrelaxedsimd_fma_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmrelaxedsimd_fma_loadsplat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmrelaxedsimd_fma_loadsplat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmrelaxedsimd_splat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmrelaxedsimd_splat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmrelaxedsimd_splat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmrelaxedsimd_splat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8__wasmrelaxedsimd_fma_splat);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__wasmrelaxedsimd_fma_splat);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__wasmrelaxedsimd_fma_splat);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__wasmrelaxedsimd_fma_splat);

  BENCHMARK_FP32_END2END(f32_gemm_3x8s4__wasmrelaxedsimd);
  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__wasmrelaxedsimd);
  BENCHMARK_FP32_END2END(f32_gemm_5x8s4__wasmrelaxedsimd);
  BENCHMARK_FP32_END2END(f32_gemm_6x8s4__wasmrelaxedsimd);

  BENCHMARK_FP32_END2END(f32_gemm_3x8s4__wasmrelaxedsimd_fma);
  BENCHMARK_FP32_END2END(f32_gemm_4x8s4__wasmrelaxedsimd_fma);
  BENCHMARK_FP32_END2END(f32_gemm_5x8s4__wasmrelaxedsimd_fma);
  BENCHMARK_FP32_END2END(f32_gemm_6x8s4__wasmrelaxedsimd_fma);
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  static void f32_gemm_3x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_arm_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_3x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_4x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_5x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_x86_loadsplat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_6x8__wasmsimd_loadsplat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_loadsplat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_3x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_3x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_4x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_4x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_5x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_5x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_arm_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_arm_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_arm_splat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_6x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_6x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_3x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_3x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_3x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_3x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_4x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_4x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_4x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_4x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_4x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_5x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_5x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_5x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_5x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_5x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_6x8__wasmsimd_x86_splat(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_6x8__wasmsimd_x86_splat,
      xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_igemm_minmax_ukernel_1x8__wasmsimd_x86_splat,
      xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_6x8__wasmsimd_splat,
      xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_relu_ukernel_1x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_6x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_6x8__wasmsimd_splat,
      xnn_f32_gemm_ukernel_1x8__wasmsimd_splat,
      xnn_f32_igemm_ukernel_1x8__wasmsimd_splat,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */);
  }
  static void f32_gemm_3x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_3x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_3x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_3x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_4x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_4x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_4x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_5x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_5x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_5x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_6x8s4__wasmsimd_arm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_arm,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_arm,
      xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_6x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_6x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_6x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      6 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_3x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_3x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_3x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_3x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_3x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      3 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_4x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_4x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_4x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_4x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_4x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      4 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_5x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_5x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_5x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_5x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_5x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
      5 /* mr */, 8 /* nr */, 0 /* log2(kr) */, 2 /* log2(sr) */);
  }
  static void f32_gemm_6x8s4__wasmsimd_x86(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_6x8s4__wasmsimd_x86,
      xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_igemm_minmax_ukernel_1x8s4__wasmsimd_x86,
      xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_6x8s4__wasmsimd,
      xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_relu_ukernel_1x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_6x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_6x8s4__wasmsimd,
      xnn_f32_gemm_ukernel_1x8s4__wasmsimd,
      xnn_f32_igemm_ukernel_1x8s4__wasmsimd,
      xnn_init_f32_minmax_wasmsimd_params,
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
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASM
  static void f32_gemm_2x4__wasm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_2x4__wasm,
      xnn_f32_igemm_minmax_ukernel_2x4__wasm,
      xnn_f32_gemm_minmax_ukernel_1x4__wasm,
      xnn_f32_igemm_minmax_ukernel_1x4__wasm,
      xnn_f32_gemm_relu_ukernel_2x4__wasm,
      xnn_f32_igemm_relu_ukernel_2x4__wasm,
      xnn_f32_gemm_relu_ukernel_1x4__wasm,
      xnn_f32_igemm_relu_ukernel_1x4__wasm,
      xnn_f32_gemm_ukernel_2x4__scalar,
      xnn_f32_igemm_ukernel_2x4__scalar,
      xnn_f32_gemm_ukernel_1x4__scalar,
      xnn_f32_igemm_ukernel_1x4__scalar,
      xnn_init_f32_minmax_scalar_params,
      2 /* mr */, 4 /* nr */);
  }

  static void f32_gemm_4x4__wasm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x4__wasm,
      xnn_f32_igemm_minmax_ukernel_4x4__wasm,
      xnn_f32_gemm_minmax_ukernel_1x4__wasm,
      xnn_f32_igemm_minmax_ukernel_1x4__wasm,
      xnn_f32_gemm_relu_ukernel_4x4__wasm,
      xnn_f32_igemm_relu_ukernel_4x4__wasm,
      xnn_f32_gemm_relu_ukernel_1x4__wasm,
      xnn_f32_igemm_relu_ukernel_1x4__wasm,
      xnn_f32_gemm_ukernel_4x4__scalar,
      xnn_f32_igemm_ukernel_4x4__scalar,
      xnn_f32_gemm_ukernel_1x4__scalar,
      xnn_f32_igemm_ukernel_1x4__scalar,
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
    xnn_f32_gemm_relu_ukernel_2x4__scalar,
    xnn_f32_igemm_relu_ukernel_2x4__scalar,
    xnn_f32_gemm_relu_ukernel_1x4__scalar,
    xnn_f32_igemm_relu_ukernel_1x4__scalar,
    xnn_f32_gemm_ukernel_2x4__scalar,
    xnn_f32_igemm_ukernel_2x4__scalar,
    xnn_f32_gemm_ukernel_1x4__scalar,
    xnn_f32_igemm_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    2 /* mr */, 4 /* nr */);
}

static void f32_gemm_4x4__scalar(benchmark::State& state, models::ExecutionPlanFactory model) {
  GEMMEnd2EndBenchmark(state, model,
    xnn_f32_gemm_minmax_ukernel_4x4__scalar,
    xnn_f32_igemm_minmax_ukernel_4x4__scalar,
    xnn_f32_gemm_minmax_ukernel_1x4__scalar,
    xnn_f32_igemm_minmax_ukernel_1x4__scalar,
    xnn_f32_gemm_relu_ukernel_4x4__scalar,
    xnn_f32_igemm_relu_ukernel_4x4__scalar,
    xnn_f32_gemm_relu_ukernel_1x4__scalar,
    xnn_f32_igemm_relu_ukernel_1x4__scalar,
    xnn_f32_gemm_ukernel_4x4__scalar,
    xnn_f32_igemm_ukernel_4x4__scalar,
    xnn_f32_gemm_ukernel_1x4__scalar,
    xnn_f32_igemm_ukernel_1x4__scalar,
    xnn_init_f32_minmax_scalar_params,
    4 /* mr */, 4 /* nr */);
}

BENCHMARK_FP32_END2END(f32_gemm_2x4__scalar);
BENCHMARK_FP32_END2END(f32_gemm_4x4__scalar);


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
