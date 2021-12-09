// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/transpose.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

static void x32_transpose(
    benchmark::State& state, xnn_x32_transpose_ukernel_function transpose,
    size_t ukernel_size,
    benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (isa_check && !isa_check(state)) {
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);
  const size_t ukernel_bytes = ukernel_size * sizeof(uint32_t);

  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> x(
      ukernel_size * ukernel_size + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::vector<uint32_t, AlignedAllocator<uint32_t, 64>> y(
      ukernel_size * ukernel_size + XNN_EXTRA_BYTES / sizeof(uint32_t));
  std::generate(x.begin(), x.end(), std::ref(u32rng));
  std::fill(y.begin(), y.end(), 0);

  for (auto _ : state) {
    transpose(x.data(), y.data(), ukernel_bytes, ukernel_bytes, ukernel_size,
              ukernel_size);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(x32_transpose, sse_32, xnn_x32_transpose_ukernel__4x4_sse, 32)
      ->UseRealTime();
  BENCHMARK_CAPTURE(x32_transpose, sse_1024, xnn_x32_transpose_ukernel__4x4_sse, 1024)
      ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
