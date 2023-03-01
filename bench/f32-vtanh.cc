// Copyright 2023 Google LLC
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


static void f32_vtanh(
  benchmark::State& state,
  xnn_f32_vtanh_ukernel_fn tanh,
  xnn_init_f32_tanh_params_fn init_params,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-10.0f, 10.0f), std::ref(rng));

  std::vector<float, AlignedAllocator<float, 64>> x(num_elements);
  std::vector<float, AlignedAllocator<float, 64>> y(num_elements);
  std::generate(x.begin(), x.end(), std::ref(f32rng));
  std::fill(y.begin(), y.end(), std::nanf(""));

  xnn_f32_tanh_params params;
  init_params(&params);
  for (auto _ : state) {
    tanh(num_elements * sizeof(float), x.data(), y.data(), &params);
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

BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_lut8_p4h3_div_x1,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_lut8_p4h3_div_x2,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_lut8_p4h3_div_x4,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_lut8_p4h3_div_x4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_p6h5_div_x1,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_p6h5_div_x2,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, fma_expm1minus_rr1_p6h5_div_x4,
                  xnn_f32_vtanh_ukernel__fma_expm1minus_rr1_p6h5_div_x4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_lut8_p4h3_div_x1,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_lut8_p4h3_div_x2,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_lut8_p4h3_div_x4,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_lut8_p4h3_div_x4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_p6h5_div_x1,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x1,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_p6h5_div_x2,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x2,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();
BENCHMARK_CAPTURE(f32_vtanh, scalar_expm1minus_rr1_p6h5_div_x4,
                  xnn_f32_vtanh_ukernel__scalar_expm1minus_rr1_p6h5_div_x4,
                  xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
  ->Apply(benchmark::utils::UnaryElementwiseParameters<float, float>)
  ->UseRealTime();

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
