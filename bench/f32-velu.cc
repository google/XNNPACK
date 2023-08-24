// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/vunary.h>


static void f32_velu(
  benchmark::State& state,
  xnn_f32_velu_ukernel_fn elu,
  xnn_init_f32_elu_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-20.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements);
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  union xnn_f32_elu_params params;
  init_params(&params, 1.0f /* prescale */, 1.0f /* alpha */, 1.0f /* beta */);
  for (auto _ : state) {
    elu(num_elements * sizeof(float), x.data(), y.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = num_elements;
  state.counters["elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(float);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(f32_velu, neonfma_lut16_p3_u4,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u4,
                    xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_lut16_p3_u8,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u8,
                    xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_lut16_p3_u12,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u12,
                    xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_lut16_p3_u16,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u16,
                    xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_lut16_p3_u20,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u20,
                    xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_lut16_p3_u24,
                    xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u24,
                    xnn_init_f32_elu_neonfma_rr1_lut16_p3_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, neonfma_p6_u4,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u4,
                    xnn_init_f32_elu_neonfma_rr1_p6_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_p6_u8,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u8,
                    xnn_init_f32_elu_neonfma_rr1_p6_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_p6_u12,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u12,
                    xnn_init_f32_elu_neonfma_rr1_p6_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_p6_u16,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u16,
                    xnn_init_f32_elu_neonfma_rr1_p6_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_p6_u20,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u20,
                    xnn_init_f32_elu_neonfma_rr1_p6_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neonfma_p6_u24,
                    xnn_f32_velu_ukernel__neonfma_rr1_p6_u24,
                    xnn_init_f32_elu_neonfma_rr1_p6_params,
                    benchmark::utils::CheckNEONFMA)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, neon_lut16_p3_u4,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_neon_rr2_lut16_p3_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_lut16_p3_u8,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_neon_rr2_lut16_p3_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_lut16_p3_u12,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_neon_rr2_lut16_p3_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_lut16_p3_u16,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_neon_rr2_lut16_p3_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_lut16_p3_u20,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_neon_rr2_lut16_p3_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_lut16_p3_u24,
                    xnn_f32_velu_ukernel__neon_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_neon_rr2_lut16_p3_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, neon_p6_u4,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u4,
                    xnn_init_f32_elu_neon_rr2_p6_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_p6_u8,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u8,
                    xnn_init_f32_elu_neon_rr2_p6_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_p6_u12,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u12,
                    xnn_init_f32_elu_neon_rr2_p6_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_p6_u16,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u16,
                    xnn_init_f32_elu_neon_rr2_p6_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_p6_u20,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u20,
                    xnn_init_f32_elu_neon_rr2_p6_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, neon_p6_u24,
                    xnn_f32_velu_ukernel__neon_rr2_p6_u24,
                    xnn_init_f32_elu_neon_rr2_p6_params,
                    benchmark::utils::CheckNEON)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u16,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u16,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u32,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u32,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u48,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u48,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u64,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u64,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u80,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u80,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u96,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u96,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u112,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u112,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_lut16_p3_u128,
                    xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_u128,
                    xnn_init_f32_elu_avx512_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u16,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u16,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u32,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u32,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u48,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u48,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u64,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u64,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u80,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u80,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u96,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u96,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u112,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u112,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx512f_p6_u128,
                    xnn_f32_velu_ukernel__avx512f_rr1_p6_u128,
                    xnn_init_f32_elu_avx512_rr1_p6_params,
                    benchmark::utils::CheckAVX512F)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u8,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u16,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u24,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u32,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u40,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u48,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u56,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u64,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u72,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut4_p4_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_u80,
                    xnn_init_f32_elu_avx2_rr1_lut4_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u8,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u16,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u24,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u32,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u40,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u48,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u56,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u64,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u72,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut8_p4_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_u80,
                    xnn_init_f32_elu_avx2_rr1_lut8_p4_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u8,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u16,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u24,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u32,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u40,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u48,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u56,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u64,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u72,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_lut16_p3_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_u80,
                    xnn_init_f32_elu_avx2_rr1_lut16_p3_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u8,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u8,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u16,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u16,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u24,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u24,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u32,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u32,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u40,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u40,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u48,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u48,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u56,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u56,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u64,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u64,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u72,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u72,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx2_p6_u80,
                    xnn_f32_velu_ukernel__avx2_rr1_p6_u80,
                    xnn_init_f32_elu_avx2_rr1_p6_params,
                    benchmark::utils::CheckAVX2)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx_lut4_p4_u8,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u8,
                    xnn_init_f32_elu_avx_rr2_lut4_p4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut4_p4_u16,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u16,
                    xnn_init_f32_elu_avx_rr2_lut4_p4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut4_p4_u24,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u24,
                    xnn_init_f32_elu_avx_rr2_lut4_p4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut4_p4_u32,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u32,
                    xnn_init_f32_elu_avx_rr2_lut4_p4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut4_p4_u40,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u40,
                    xnn_init_f32_elu_avx_rr2_lut4_p4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut4_p4_u48,
                    xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_u48,
                    xnn_init_f32_elu_avx_rr2_lut4_p4_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx_lut16_p3_u8,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_avx_rr2_lut16_p3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut16_p3_u16,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_avx_rr2_lut16_p3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut16_p3_u24,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_avx_rr2_lut16_p3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut16_p3_u32,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u32,
                    xnn_init_f32_elu_avx_rr2_lut16_p3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut16_p3_u40,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u40,
                    xnn_init_f32_elu_avx_rr2_lut16_p3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_lut16_p3_u48,
                    xnn_f32_velu_ukernel__avx_rr2_lut16_p3_u48,
                    xnn_init_f32_elu_avx_rr2_lut16_p3_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, avx_p6_u8,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u8,
                    xnn_init_f32_elu_avx_rr2_p6_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_p6_u16,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u16,
                    xnn_init_f32_elu_avx_rr2_p6_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_p6_u24,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u24,
                    xnn_init_f32_elu_avx_rr2_p6_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_p6_u32,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u32,
                    xnn_init_f32_elu_avx_rr2_p6_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_p6_u40,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u40,
                    xnn_init_f32_elu_avx_rr2_p6_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, avx_p6_u48,
                    xnn_f32_velu_ukernel__avx_rr2_p6_u48,
                    xnn_init_f32_elu_avx_rr2_p6_params,
                    benchmark::utils::CheckAVX)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, sse41_lut16_p3_u4,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_lut16_p3_u8,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_lut16_p3_u12,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_lut16_p3_u16,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_lut16_p3_u20,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_lut16_p3_u24,
                    xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, sse41_p6_u4,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u4,
                    xnn_init_f32_elu_sse2_rr2_p6_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_p6_u8,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u8,
                    xnn_init_f32_elu_sse2_rr2_p6_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_p6_u12,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u12,
                    xnn_init_f32_elu_sse2_rr2_p6_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_p6_u16,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u16,
                    xnn_init_f32_elu_sse2_rr2_p6_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_p6_u20,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u20,
                    xnn_init_f32_elu_sse2_rr2_p6_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse41_p6_u24,
                    xnn_f32_velu_ukernel__sse41_rr2_p6_u24,
                    xnn_init_f32_elu_sse2_rr2_p6_params,
                    benchmark::utils::CheckSSE41)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, sse2_lut16_p3_u4,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_lut16_p3_u8,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_lut16_p3_u12,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_lut16_p3_u16,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_lut16_p3_u20,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_lut16_p3_u24,
                    xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, sse2_p6_u4,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u4,
                    xnn_init_f32_elu_sse2_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_p6_u8,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u8,
                    xnn_init_f32_elu_sse2_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_p6_u12,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u12,
                    xnn_init_f32_elu_sse2_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_p6_u16,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u16,
                    xnn_init_f32_elu_sse2_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_p6_u20,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u20,
                    xnn_init_f32_elu_sse2_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, sse2_p6_u24,
                    xnn_f32_velu_ukernel__sse2_rr2_p6_u24,
                    xnn_init_f32_elu_sse2_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_p6_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_p6_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_p6_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_p6_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_p6_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_fma_p6_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_p6_u4,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_p6_u8,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_p6_u12,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_p6_u16,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_p6_u20,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmrelaxedsimd_p6_u24,
                    xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_lut16_p3_u8,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_lut16_p3_u12,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_lut16_p3_u16,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_lut16_p3_u20,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_lut16_p3_u24,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_p6_u4,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_p6_u8,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_p6_u12,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_p6_u16,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_p6_u20,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_arm_p6_u24,
                    xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_p6_u4,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u4,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_p6_u8,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u8,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_p6_u12,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u12,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_p6_u16,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u16,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_p6_u20,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u20,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasmsimd_x86_p6_u24,
                    xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_u24,
                    xnn_init_f32_elu_wasmsimd_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#if XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  BENCHMARK_CAPTURE(f32_velu, wasm_lut16_p3_u1,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u1,
                    xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_lut16_p3_u2,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u2,
                    xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_lut16_p3_u3,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u3,
                    xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_lut16_p3_u4,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u4,
                    xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_lut16_p3_u5,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u5,
                    xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_lut16_p3_u6,
                    xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_u6,
                    xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();

  BENCHMARK_CAPTURE(f32_velu, wasm_p6_u1,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u1,
                    xnn_init_f32_elu_scalar_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_p6_u2,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u2,
                    xnn_init_f32_elu_scalar_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_p6_u3,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u3,
                    xnn_init_f32_elu_scalar_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_p6_u4,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u4,
                    xnn_init_f32_elu_scalar_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_p6_u5,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u5,
                    xnn_init_f32_elu_scalar_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
  BENCHMARK_CAPTURE(f32_velu, wasm_p6_u6,
                    xnn_f32_velu_ukernel__wasm_rr2_p6_u6,
                    xnn_init_f32_elu_scalar_rr2_p6_params)
    ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
    ->UseRealTime();
#endif  // XNN_ARCH_WASM || XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

BENCHMARK_CAPTURE(f32_velu, scalar_lut16_p3_u1,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u1,
                  xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_lut16_p3_u2,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u2,
                  xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_lut16_p3_u3,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u3,
                  xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_lut16_p3_u4,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u4,
                  xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_lut16_p3_u5,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u5,
                  xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_lut16_p3_u6,
                  xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_u6,
                  xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_velu, scalar_p6_u1,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u1,
                  xnn_init_f32_elu_scalar_rr2_p6_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_p6_u2,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u2,
                  xnn_init_f32_elu_scalar_rr2_p6_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_p6_u3,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u3,
                  xnn_init_f32_elu_scalar_rr2_p6_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_p6_u4,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u4,
                  xnn_init_f32_elu_scalar_rr2_p6_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_p6_u5,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u5,
                  xnn_init_f32_elu_scalar_rr2_p6_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_velu, scalar_p6_u6,
                  xnn_f32_velu_ukernel__scalar_rr2_p6_u6,
                  xnn_init_f32_elu_scalar_rr2_p6_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
