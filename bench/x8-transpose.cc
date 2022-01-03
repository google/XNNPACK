// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/transpose.h>

#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

static void x8_transpose(
    benchmark::State& state,
    xnn_x8_transpose_ukernel_function transpose,
    size_t ukernel_size,
    benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (isa_check && !isa_check(state)) {
    return;
  }

  const size_t ukernel_bytes = ukernel_size * sizeof(uint8_t);

  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> x(
      ukernel_size * ukernel_size + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::vector<uint8_t, AlignedAllocator<uint8_t, 64>> y(
      ukernel_size * ukernel_size + XNN_EXTRA_BYTES / sizeof(uint8_t));
  std::iota(x.begin(), x.end(), 0);
  std::fill(y.begin(), y.end(), 0);

  for (auto _ : state) {
    transpose(x.data(), y.data(), ukernel_bytes, ukernel_bytes, ukernel_size,
              ukernel_size);
  }
}

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(x8_transpose, sse2_32, xnn_x8_transpose_ukernel__16x16_sse2, 32)
      ->UseRealTime();
  BENCHMARK_CAPTURE(x8_transpose, sse2_64, xnn_x8_transpose_ukernel__16x16_sse2, 64)
      ->UseRealTime();
  BENCHMARK_CAPTURE(x8_transpose, sse2_117, xnn_x8_transpose_ukernel__16x16_sse2, 117)
      ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
