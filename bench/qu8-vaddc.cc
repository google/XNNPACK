// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include "bench/utils.h"

#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/vbinary.h"


static void qu8_vaddc(
  benchmark::State& state,
  uint64_t arch_flags,
  xnn_qu8_vadd_minmax_ukernel_fn vaddc,
  xnn_init_qu8_add_minmax_params_fn init_params)
{
  if (!benchmark::utils::CheckArchFlags(state, arch_flags)) {
    return;
  }

  const size_t num_elements = state.range(0);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u8rng = std::bind(
    std::uniform_int_distribution<uint32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max()),
    std::ref(rng));

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> a(num_elements);
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> sum(num_elements);
  std::generate(a.begin(), a.end(), std::ref(u8rng));
  const uint8_t b = u8rng();

  struct xnn_qu8_add_minmax_params params;
  init_params(&params,
    127 /* a zero point */, 127 /* b zero point */, 127 /* output zero point */,
    0.5f /* a-output scale */, 0.75f /* b-output scale */,
    std::numeric_limits<uint8_t>::min() + 1, std::numeric_limits<uint8_t>::max() - 1);
  for (auto _ : state) {
    vaddc(num_elements * sizeof(uint8_t), a.data(), &b, sum.data(), &params);
  }

  const uint64_t cpu_frequency = benchmark::utils::GetCurrentCpuFrequency();
  if (cpu_frequency != 0) {
    state.counters["cpufreq"] = cpu_frequency;
  }

  const size_t num_elements_per_iteration = num_elements;
  state.counters["num_elements"] =
    benchmark::Counter(uint64_t(state.iterations()) * num_elements_per_iteration, benchmark::Counter::kIsRate);

  const size_t bytes_per_iteration = 2 * num_elements * sizeof(int8_t);
  state.counters["bytes"] =
    benchmark::Counter(uint64_t(state.iterations()) * bytes_per_iteration, benchmark::Counter::kIsRate);
}

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, datatype, params_type, init_params) \
  BENCHMARK_CAPTURE(qu8_vaddc, ukernel, arch_flags, ukernel, init_params) \
    ->Apply(benchmark::utils::BinaryElementwiseParameters<int8_t, int8_t>) \
    ->UseRealTime();
#include "src/qu8-vaddc/qu8-vaddc-minmax.h"
#undef XNN_UKERNEL_WITH_PARAMS

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
