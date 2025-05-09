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
template <typename T, typename UKernelParams>
using UKernelFn = void (*)(size_t, size_t, const T*, size_t, const T*, T*,
                           const UKernelParams*);

template <typename T, typename UKernelParams>
static void reduce(benchmark::State& state, uint64_t arch_flags,
                   UKernelFn<T, UKernelParams> ukernel) {
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t channels = state.range(0);
  const size_t rows = state.range(1);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> input(channels * rows,
                                                     xnnpack::XnnExtraBytes);
  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> zero(channels, 0,
                                                    xnnpack::XnnExtraBytes);
  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);
  xnnpack::Buffer<T, XNN_ALLOCATION_ALIGNMENT> output(channels);

  for (auto _ : state) {
    ukernel(rows, channels, input.data(), channels * sizeof(T), zero.data(),
            output.data(), nullptr);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t elements_per_iteration = channels * rows;
  state.counters["elements"] =
      benchmark::Counter(uint64_t(state.iterations()) * elements_per_iteration,
                         benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = channels * rows * sizeof(T);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, row_tile, batch_tile, vector_tile, \
                    datatype_in, datatype_out)                              \
  BENCHMARK_CAPTURE(reduce, ukernel, arch_flags, ukernel)                   \
      ->Apply(benchmark::utils::ReduceDiscontiguousParameters<datatype_in>) \
      ->UseRealTime();
#include "src/f16-rdminmax/f16-rdmax.h"
#include "src/f16-rdminmax/f16-rdmin.h"
#include "src/f32-rdminmax/f32-rdmax.h"
#include "src/f32-rdminmax/f32-rdmin.h"
#include "src/s8-rdminmax/s8-rdmax.h"
#include "src/s8-rdminmax/s8-rdmin.h"
#include "src/u8-rdminmax/u8-rdmax.h"
#include "src/u8-rdminmax/u8-rdmin.h"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif