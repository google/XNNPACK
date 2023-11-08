// Copyright 2019 Google LLC
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
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/models.h>
#include <xnnpack/pack.h>


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

  struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == nullptr) {
    state.SkipWithError("hardware does not support F32 gemm");
    return;
  }

  struct xnn_gemm_config* gemm_nr2_config = xnn_init_f32_gemm_nr2_config();
  if (gemm_nr2_config == nullptr) {
    state.SkipWithError("hardware does not support F32 gemm");
    return;
  }

  // Override microkernels chosen in xnn_initialize
  std::memset(gemm_config, 0, sizeof(struct xnn_gemm_config));
  std::memset(gemm_nr2_config, 0, sizeof(struct xnn_gemm_config));
  gemm_config->minmax.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm_minmax));
  gemm_config->minmax.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm_minmax));
  gemm_config->minmax.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1_minmax));
  gemm_config->minmax.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1_minmax));
  gemm_config->relu.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm_relu));
  gemm_config->relu.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm_relu));
  gemm_config->relu.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1_relu));
  gemm_config->relu.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1_relu));
  gemm_config->linear.gemm[mr-1] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm));
  gemm_config->linear.igemm[mr-1] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm));
  gemm_config->linear.gemm[0] = xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn(gemm1));
  gemm_config->linear.igemm[0] = xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn(igemm1));
  gemm_config->init.f32 = init_params;
  gemm_config->mr = mr;
  gemm_config->nr = nr;
  gemm_config->log2_kr = log2_kr;
  gemm_config->log2_sr = log2_sr;
  gemm_config->pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_gemm_goi_w;

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
namespace {
struct MrAndGenerators {
  uint8_t mr;
  xnn_jit_gemm_code_generator_fn gemm_generator;
  xnn_jit_igemm_code_generator_fn igemm_generator;
};

uint8_t MaxMr(const std::vector<MrAndGenerators>& generators) {
  return std::max_element(generators.begin(), generators.end(),
                          [](const auto& lhs, const auto rhs) { return lhs.mr < rhs.mr; })->mr;
}
}

static void GEMMEnd2EndBenchmark(
  benchmark::State& state,
  models::ExecutionPlanFactory model_factory,
  const std::vector<MrAndGenerators>& generators,
  xnn_init_f32_minmax_params_fn init_params,
  uint8_t nr, uint8_t log2_kr = 0, uint8_t log2_sr = 0,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }
  if (xnn_initialize(nullptr /* allocator */) != xnn_status_success) {
    state.SkipWithError("failed to initialize XNNPACK");
    return;
  }

  struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == nullptr) {
    state.SkipWithError("hardware does not support F32 gemm");
    return;
  }

  // Override microkernels chosen in xnn_initialize
  std::memset(gemm_config, 0, sizeof(struct xnn_gemm_config));
  gemm_config->init.f32 = init_params;
  gemm_config->mr = MaxMr(generators);
  gemm_config->nr = nr;
  gemm_config->log2_kr = log2_kr;
  gemm_config->log2_sr = log2_sr;
  gemm_config->pack_gemm_goi = (xnn_packw_gemm_goi_ukernel_fn) xnn_pack_f32_gemm_goi_w;

  for (const auto& generator : generators) {
    const size_t index = generator.mr - 1;
    gemm_config->generator.gemm[index] = xnn_init_hmp_gemm_codegen(generator.gemm_generator);
    gemm_config->generator.igemm[index] = xnn_init_hmp_igemm_codegen(generator.igemm_generator);
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
      /*mr=*/4, /*nr=*/2);
  }
  static void f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2);
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
      /*mr=*/4, /*nr=*/2);
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
      /*mr=*/4, /*nr=*/12);
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
      /*mr=*/4, /*nr=*/8);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
  }
  static void f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
  }
  static void f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
  }
  static void f32_gemm_6x8__asm_aarch64_neonfma_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_4x2__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_FP32_END2END(f32_gemm_4x2__asm_aarch64_neonfma_ld64)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_ld64)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a53_prfm)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a55)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75)
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
  BENCHMARK_FP32_END2END(f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_5x8__asm_aarch64_neonfma_cortex_a75_prfm);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a53_prfm);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a55);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a73);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__asm_aarch64_neonfma_cortex_a75_prfm);
  BENCHMARK_FP32_END2END(f32_gemm_4x12__asm_aarch64_neonfma_cortex_a53);
#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM64
  static void f32_gemm_2x16__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16);
  }
  static void f32_gemm_3x16__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16);
  }
  static void f32_gemm_4x16__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16);
  }
  static void f32_gemm_5x16__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16);
  }
  static void f32_gemm_6x16__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16);
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
      /*mr=*/4, /*nr=*/2);
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
      /*mr=*/6, /*nr=*/2);
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
      /*mr=*/4, /*nr=*/8);
  }
  static void f32_gemm_4x8__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
  }
  static void f32_gemm_6x8__aarch64_neonfma_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/8);
  }

  BENCHMARK_FP32_END2END(f32_gemm_2x16__aarch64_neonfma_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_3x16__aarch64_neonfma_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_4x16__aarch64_neonfma_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_5x16__aarch64_neonfma_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x16__aarch64_neonfma_lane_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_4x2__aarch64_neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x2__aarch64_neonfma_lane_ld64);

  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__aarch64_neonfma_lane_ld128);

  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_lane_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_6x8__aarch64_neonfma_lane_ld128);
#endif  // XNN_ARCH_ARM64

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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a53_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }
  static void f32_gemm_4x8__asm_aarch32_neon_cortex_a75_prfm(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm,
      xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm,
      xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }

  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_ld64);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a7);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a53);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a53_prfm);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a55);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a75);
  BENCHMARK_FP32_END2END(f32_gemm_4x8__asm_aarch32_neon_cortex_a75_prfm);
#endif  // XNN_ARCH_ARM && XNN_ENABLE_ASSEMBLY

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  static void f32_gemm_2x16__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_2x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_2x16__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/2, /*nr=*/16);
  }
  static void f32_gemm_3x16__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_3x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_3x16__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/3, /*nr=*/16);
  }
  static void f32_gemm_4x16__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x16__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/16);
  }
  static void f32_gemm_5x16__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_5x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_5x16__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/5, /*nr=*/16);
  }
  static void f32_gemm_6x16__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_6x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_6x16__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x16__neon_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/6, /*nr=*/16);
  }
  static void f32_gemm_4x2__neon_lane_ld64(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64,
      xnn_f32_igemm_minmax_ukernel_4x2__neon_lane_ld64,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/2, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/2, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
      benchmark::utils::CheckNEON);
  }

  static void f32_gemm_4x8__neon_lane_ld128(benchmark::State& state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(state, model,
      xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_4x8__neon_lane_ld128,
      xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld128,
      xnn_f32_igemm_minmax_ukernel_1x8__neon_lane_ld128,
      nullptr /* gemm_relu */, nullptr /* igemm_relu */, nullptr /* gemm1_relu */, nullptr /* igemm1_relu */,
      nullptr /* gemm */, nullptr /* igemm */, nullptr /* gemm1 */, nullptr /* igemm1 */,
      xnn_init_f32_minmax_scalar_params,
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/8, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/8, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2,
      benchmark::utils::CheckNEONFMA);
  }
  BENCHMARK_FP32_END2END(f32_gemm_2x16__neon_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_3x16__neon_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_4x16__neon_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_5x16__neon_lane_ld128);
  BENCHMARK_FP32_END2END(f32_gemm_6x16__neon_lane_ld128);

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
  static void f32_gemm_4x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {4, xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {4, xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_1x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {6, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_2x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {2, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_3x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {3, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_4x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {4, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_5x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {5, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_6x8__jit_aarch64_neonfma_cortex_a75(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75},
         {6, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_1x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {1, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_2x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {2, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_3x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {3, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_4x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {4, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_5x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {5, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8_6x8__jit_aarch64_neonfma_cortex_a75_prfm(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm},
         {6, xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm, xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_aarch64_neonfma_cortex_a75_prfm);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_1x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_2x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_3x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_4x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_5x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_6x8__jit_aarch64_neonfma_cortex_a75);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_1x8__jit_aarch64_neonfma_cortex_a75_prfm);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_2x8__jit_aarch64_neonfma_cortex_a75_prfm);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_3x8__jit_aarch64_neonfma_cortex_a75_prfm);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_4x8__jit_aarch64_neonfma_cortex_a75_prfm);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_5x8__jit_aarch64_neonfma_cortex_a75_prfm);
BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8_6x8__jit_aarch64_neonfma_cortex_a75_prfm);

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
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/7, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/8, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/7, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/8, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/2,
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/7, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/3, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/4, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/5, /*nr=*/16, /*log2_kr=*/0, /*log2_sr=*/0,
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
  }

  static void f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_fma_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_fma_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_fma_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_fma_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8s4__jit_wasmrelaxedsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8s4__jit_wasmrelaxedsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8s4__jit_wasmrelaxedsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8s4__jit_wasmrelaxedsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8s4__jit_wasmrelaxedsimd32_x86_fma(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8s4__jit_wasmrelaxedsimd32_x86_fma(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8s4__jit_wasmrelaxedsimd32_x86_fma(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8s4__jit_wasmrelaxedsimd32_x86_fma(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
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

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_loadsplat);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_fma_loadsplat);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_splat);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmrelaxedsimd32_x86_fma_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmrelaxedsimd32_x86_fma_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmrelaxedsimd32_x86_fma_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmrelaxedsimd32_x86_fma_splat);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8s4__jit_wasmrelaxedsimd32_x86);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8s4__jit_wasmrelaxedsimd32_x86);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8s4__jit_wasmrelaxedsimd32_x86);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8s4__jit_wasmrelaxedsimd32_x86);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8s4__jit_wasmrelaxedsimd32_x86_fma);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8s4__jit_wasmrelaxedsimd32_x86_fma);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8s4__jit_wasmrelaxedsimd32_x86_fma);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8s4__jit_wasmrelaxedsimd32_x86_fma);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8);
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
      /*mr=*/4, /*nr=*/8);
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
      /*mr=*/5, /*nr=*/8);
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
      /*mr=*/6, /*nr=*/8);
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
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/3, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/4, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/5, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
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
      /*mr=*/6, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
  }

  static void f32_gemm_3x8__jit_wasmsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_wasmsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8__jit_wasmsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8__jit_wasmsimd32_x86_loadsplat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8__jit_wasmsimd32_x86_loadsplat_unroll(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2},
         {3, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}
        },
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_4x8__jit_wasmsimd32_x86_loadsplat_unroll(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2},
         {4, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}
        },
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_5x8__jit_wasmsimd32_x86_loadsplat_unroll(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2},
         {5, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1}
        },
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }
  static void f32_gemm_6x8__jit_wasmsimd32_x86_loadsplat_unroll(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{1, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2},
         {5, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1},
         {6, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2, xnn_generate_f32_igemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2}
        },
        xnn_init_f32_minmax_scalar_params, /*nr=*/8);
  }

  static void f32_gemm_3x8s4__jit_wasmsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
  }
  static void f32_gemm_4x8s4__jit_wasmsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
  }
  static void f32_gemm_5x8s4__jit_wasmsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
  }
  static void f32_gemm_6x8s4__jit_wasmsimd32_x86(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, /*nr=*/8, /*log2_kr=*/0, /*log2_sr=*/2);
  }

  static void f32_gemm_3x8__jit_wasmsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{3, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, 8 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */);
  }
  static void f32_gemm_4x8__jit_wasmsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{4, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, 8 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */);
  }
  static void f32_gemm_5x8__jit_wasmsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{5, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, 8 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */);
  }
  static void f32_gemm_6x8__jit_wasmsimd32_x86_splat(
      benchmark::State &state, models::ExecutionPlanFactory model) {
    GEMMEnd2EndBenchmark(
        state, model,
        {{6, xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1, xnn_generate_f32_igemm_ukernel_6x8s4__wasmsimd32_x86_x1}},
        xnn_init_f32_minmax_scalar_params, 8 /* nr */, 0 /* log2_kr */, 2 /* log2_sr */);
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

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmsimd32_x86_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmsimd32_x86_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmsimd32_x86_loadsplat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmsimd32_x86_loadsplat);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmsimd32_x86_loadsplat_unroll);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmsimd32_x86_loadsplat_unroll);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmsimd32_x86_loadsplat_unroll);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmsimd32_x86_loadsplat_unroll);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8s4__jit_wasmsimd32_x86);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8s4__jit_wasmsimd32_x86);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8s4__jit_wasmsimd32_x86);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8s4__jit_wasmsimd32_x86);

  BENCHMARK_FP32_END2END_JIT(f32_gemm_3x8__jit_wasmsimd32_x86_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_4x8__jit_wasmsimd32_x86_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_5x8__jit_wasmsimd32_x86_splat);
  BENCHMARK_FP32_END2END_JIT(f32_gemm_6x8__jit_wasmsimd32_x86_splat);
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
      /*mr=*/2, /*nr=*/4);
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
      /*mr=*/4, /*nr=*/4);
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
    /*mr=*/2, /*nr=*/4);
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
    /*mr=*/4, /*nr=*/4);
}

BENCHMARK_FP32_END2END(f32_gemm_2x4__scalar);
BENCHMARK_FP32_END2END(f32_gemm_4x4__scalar);


#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
