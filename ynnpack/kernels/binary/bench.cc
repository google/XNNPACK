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
#include "ynnpack/kernels/binary/binary.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename A, typename B, typename X>
void bench(benchmark::State& state, uint64_t arch_flags,
           binary_kernel_fn kernel, A, B, X) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  size_t m = state.range(0);
  size_t n = state.range(1);
  bool broadcast_a = state.range(2);
  bool broadcast_b = state.range(3);

  Tensor<A> a({m, broadcast_a ? 1 : n});
  Tensor<B> b({m, broadcast_b ? 1 : n});
  Tensor<X> x({m, n});
  a.fill(1);
  b.fill(1);
  broadcast_extent_1(a);
  broadcast_extent_1(b);

  for (auto _ : state) {
    kernel(m, n, a.stride(0) * sizeof(A), a.stride(1) * sizeof(A), a.base(),
           b.stride(0) * sizeof(B), b.stride(1) * sizeof(B), b.base(),
           x.stride(0) * sizeof(X), x.base(), nullptr);
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes =
      m * n * (sizeof(X) + sizeof(A) * !broadcast_a + sizeof(B) * !broadcast_b);
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

void bench_reference(benchmark::State& state, binary_kernel_fn kernel) {
  return bench(state, arch_flag::none, kernel, float{}, float{}, float{});
}

template <typename A, typename B, typename X>
void Params(benchmark::Benchmark* b) {
  b->ArgNames({"m", "n", "broadcast_a", "broadcast_b"});
  b->Args({1, 4096, 0, 0});
  b->Args({1, 4096, 1, 0});
  b->Args({1, 4096, 0, 1});
  b->Args({4, 1024, 0, 0});
  b->Args({4, 1024, 1, 0});
  b->Args({4, 1024, 0, 1});
  b->Args({16, 256, 0, 0});
  b->Args({16, 256, 1, 0});
  b->Args({16, 256, 0, 1});
}

#define BENCHMARK_REFERENCE(op, type)                                \
  BENCHMARK_CAPTURE(                                                 \
      bench_reference, #op,                                          \
      get_binary_reference_kernel(ynn_binary_##op, type_of<type>())) \
      ->Apply(Params<type, type, type>)                              \
      ->UseRealTime();

BENCHMARK_REFERENCE(add, float);
BENCHMARK_REFERENCE(copysign, float);
BENCHMARK_REFERENCE(divide, float);
BENCHMARK_REFERENCE(max, float);
BENCHMARK_REFERENCE(min, float);
BENCHMARK_REFERENCE(multiply, float);
BENCHMARK_REFERENCE(pow, float);
BENCHMARK_REFERENCE(squared_difference, float);
BENCHMARK_REFERENCE(subtract, float);
BENCHMARK_REFERENCE(leaky_relu, float);

BENCHMARK_REFERENCE(add, int32_t);
BENCHMARK_REFERENCE(divide, int32_t);
BENCHMARK_REFERENCE(max, int32_t);
BENCHMARK_REFERENCE(min, int32_t);
BENCHMARK_REFERENCE(multiply, int32_t);
BENCHMARK_REFERENCE(pow, int32_t);
BENCHMARK_REFERENCE(subtract, int32_t);

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, type_a, type_b, type_c) \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, type_a(), type_b(),     \
                    type_c())                                                  \
      ->Apply(Params<type_a, type_b, type_c>)                                  \
      ->UseRealTime();
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
