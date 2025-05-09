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
using UKernelFn = void (*)(size_t, const T*, T*, const UKernelParams*);

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
  xnnpack::fill_uniform_random_bits(input.data(), input.size(), rng);

  T output[2];
  for (auto _ : state) {
    for (size_t r = 0; r < rows; ++r) {
      ukernel(channels * sizeof(T), input.data() + r * channels, output,
              nullptr);
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

  const size_t bytes_per_iteration = rows * channels * sizeof(T);
  state.counters["bytes"] =
      benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration,
                         benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype_in, \
                    datatype_out)                                              \
  BENCHMARK_CAPTURE(reduce, ukernel, arch_flags, ukernel)                      \
      ->Apply(benchmark::utils::ReduceParameters<datatype_in>)                 \
      ->UseRealTime();
#include "src/f16-rminmax/f16-rmax.h"
#include "src/f16-rminmax/f16-rmin.h"
#include "src/f16-rminmax/f16-rminmax.h"
#include "src/f32-rminmax/f32-rmax.h"
#include "src/f32-rminmax/f32-rmin.h"
#include "src/f32-rminmax/f32-rminmax.h"
#include "src/s8-rminmax/s8-rmax.h"
#include "src/s8-rminmax/s8-rmin.h"
#include "src/s8-rminmax/s8-rminmax.h"
#include "src/u8-rminmax/u8-rmax.h"
#include "src/u8-rminmax/u8-rmin.h"
#include "src/u8-rminmax/u8-rminmax.h"
#undef XNN_UKERNEL

#ifndef XNNPACK_BENCHMARK_NO_MAIN
XNN_BENCHMARK_MAIN();
#endif