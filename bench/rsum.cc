// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <random>

#include "bench/utils.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"  // IWYU pragma: keep
#include "src/xnnpack/reduce.h"  // IWYU pragma: keep
#include <benchmark/benchmark.h>

// Microkernel function, templated on the `params` type.
template <typename Input, typename Output, typename UKernelParams>
using UKernelFn = void (*)(size_t, const Input*, Output*, const UKernelParams*);

template <typename Input, typename Output, typename UKernelParams>
static void reduce(benchmark::State& state, uint64_t arch_flags,
                   UKernelFn<Input, Output, UKernelParams> ukernel) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t channels = state.range(0);
  const size_t rows = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  xnnpack::Buffer<Input, XNN_ALLOCATION_ALIGNMENT> input(
      channels * rows, xnnpack::XnnExtraBytes);
  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);

  UKernelParams params;
  memset(&params, 0, sizeof(params));

  Output output = 0;
  for (auto _ : state) {
    for (size_t r = 0; r < rows; ++r) {
      ukernel(channels * sizeof(Input), input.data() + r * channels, &output,
              &params);
    }
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = rows * channels;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = rows * channels * sizeof(Input);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out, params_type, init_params)                                              \
  BENCHMARK_CAPTURE(reduce, ukernel, arch_flags, ukernel)                      \
      ->Apply(benchmark::utils::ReduceParameters<datatype_in>)                 \
      ->UseRealTime();
#include "src/f16-f32acc-rsum/f16-f32acc-rsum.inc"
#include "src/f16-rsum/f16-rsum.inc"
#include "src/f32-rsum/f32-rsum.inc"
#include "src/qs8-rsum/qs8-rsum.inc"
#include "src/qu8-rsum/qu8-rsum.inc"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif