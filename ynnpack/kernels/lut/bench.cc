// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/kernels/lut/lut.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename A, typename X>
void bench(benchmark::State& state, uint64_t arch_flags, lut_kernel_fn kernel,
           A, X) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  size_t n = state.range(0);

  Buffer<X> lut(1 << (sizeof(A) * 8));
  std::iota(lut.begin(), lut.end(), 0);
  Buffer<A> a(n);
  TypeGenerator<A> a_gen;
  std::mt19937 rng;
  std::generate(a.begin(), a.end(), [&]() { return a_gen(rng); });

  Buffer<X> x(n);

  for (auto _ : state) {
    kernel(n, a.data(), lut.data(), x.data());
  }
}

template <typename A, typename X>
void Params(benchmark::internal::Benchmark* b) {
  b->ArgNames({"n"});
  b->Arg(1024 * 1024);
}

#define YNN_LUT_KERNEL(arch_flags, kernel, type_a, type_x)                 \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, type_a(), type_x()) \
      ->Apply(Params<type_a, type_x>)                                      \
      ->UseRealTime();
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_LUT_KERNEL

}  // namespace ynn
