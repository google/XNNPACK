// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <string>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/ternary/ternary.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename A, typename B, typename C, typename X>
void bench(benchmark::State& state, uint64_t arch_flags,
           ternary_kernel_fn kernel, A, B, C, X) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  size_t m = state.range(0);
  size_t n = state.range(1);
  bool broadcast_a = state.range(2);
  bool broadcast_b = state.range(3);
  bool broadcast_c = state.range(4);

  Tensor<A> a({m, broadcast_a ? 1 : n});
  Tensor<B> b({m, broadcast_b ? 1 : n});
  Tensor<C> c({m, broadcast_c ? 1 : n});
  Tensor<X> x({m, n});
  a.fill(1);
  b.fill(1);
  c.fill(1);
  broadcast_extent_1(a);
  broadcast_extent_1(b);
  broadcast_extent_1(c);

  for (auto _ : state) {
    kernel(m, n, a.stride(0) * sizeof(A), a.stride(1) * sizeof(A), a.base(),
           b.stride(0) * sizeof(B), b.stride(1) * sizeof(B), b.base(),
           c.stride(0) * sizeof(C), c.stride(1) * sizeof(C), c.base(),
           x.stride(0) * sizeof(X), x.base(), nullptr);
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes = m * n *
                       (sizeof(X) + sizeof(A) * !broadcast_a +
                        sizeof(B) * !broadcast_b + sizeof(C) * !broadcast_c);
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

template <typename A, typename B, typename C, typename X>
void Params(benchmark::Benchmark* b) {
  b->ArgNames({"m", "n", "broadcast_a", "broadcast_b", "broadcast_c"});
  b->Args({1, 4096, 0, 0, 0});
  b->Args({1, 4096, 0, 1, 0});
  b->Args({1, 4096, 0, 0, 1});
  b->Args({4, 1024, 0, 0, 0});
  b->Args({4, 1024, 0, 1, 0});
  b->Args({4, 1024, 0, 0, 1});
  b->Args({16, 256, 0, 0, 0});
  b->Args({16, 256, 0, 1, 0});
  b->Args({16, 256, 0, 0, 1});
}

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, type_a, type_b, type_c, \
                               type_x)                                         \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, type_a(), type_b(),     \
                    type_c(), type_x())                                        \
      ->Apply(Params<type_a, type_b, type_c, type_x>)                          \
      ->UseRealTime();
#include "ynnpack/kernels/ternary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
