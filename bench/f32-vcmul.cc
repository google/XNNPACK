// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/vbinary.h"
#include <benchmark/benchmark.h>

static void f32_vcmul(benchmark::State& state, uint64_t arch_flags,
                      xnn_f32_vbinary_ukernel_fn vcmul,
                      xnn_init_f32_default_params_fn init_params = nullptr,
                      benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(-1.0f, 1.0f),
                          std::ref(rng));

  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> a(num_elements * 2,
                                                     xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> b(num_elements * 2,
                                                     xnnpack::XnnExtraBytes);
  xnnpack::Buffer<float, XNN_ALLOCATION_ALIGNMENT> product(num_elements * 2);
  std::generate(a.begin(), a.end(), std::ref(f32rng));
  std::generate(b.begin(), b.end(), std::ref(f32rng));

  struct xnn_f32_default_params params;
  if (init_params != nullptr) {
    init_params(&params);
  }
  for (auto _ : state) {
    vcmul(num_elements * sizeof(float), a.data(), b.data(), product.data(),
          init_params == nullptr ? nullptr : &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t num_elements_per_iteration = num_elements;
  state.counters["num_elements"] = benchmark::Counter(
      uint64_t(state.iterations()) * num_elements_per_iteration,
      benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration =
      3 * num_elements * sizeof(std::complex<float>);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype,   \
                    params_type, init_params)                                 \
  BENCHMARK_CAPTURE(f32_vcmul, ukernel, arch_flags, ukernel, init_params)     \
      ->Apply(                                                                \
          benchmark::utils::BinaryElementwiseParameters<std::complex<float>,  \
                                                        std::complex<float>>) \
      ->UseRealTime();
#include "src/f32-vbinary/f32-vcmul.inc"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif
