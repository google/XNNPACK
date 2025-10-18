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
           binary_kernel_fn kernel, init_binary_params_fn init_params, A, B,
           X) {
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

  binary_params params;
  if (init_params) {
    // Use non-trivial quantization in case the kernel tries to optimize for
    // that.
    init_params(0.5f, 1, 0.25f, 2, 2.0f, 3, params);
  }

  for (auto _ : state) {
    kernel(m, n, a.stride(0) * sizeof(A), a.stride(1) * sizeof(A), a.base(),
           b.stride(0) * sizeof(B), b.stride(1) * sizeof(B), b.base(),
           x.stride(0) * sizeof(X), x.base(), &params);
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes =
      m * n * (sizeof(X) + sizeof(A) * !broadcast_a + sizeof(B) * !broadcast_b);
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

template <typename T>
void bench(benchmark::State& state, uint64_t arch_flags,
           const binary_kernel* kernel, T) {
  if (!kernel) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }
  bench(state, arch_flags, kernel->op, kernel->init_params, T(), T(), T());
}

template <typename A, typename B, typename X>
void Params(benchmark::internal::Benchmark* b) {
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

#define BENCHMARK_REFERENCE(op, type)                                       \
  BENCHMARK_CAPTURE(bench, reference_binary_##op##_##type, arch_flag::none, \
                    get_binary_reference_kernel(ynn_binary_##op, type{}),   \
                    type())                                                 \
      ->Apply(Params<type, type, type>)                                     \
      ->UseRealTime();

using qint8 = quantized<int8_t>;
using quint8 = quantized<uint8_t>;

#define BENCHMARK_REFERENCE_REAL(op) \
  BENCHMARK_REFERENCE(op, float);    \
  BENCHMARK_REFERENCE(op, half);     \
  BENCHMARK_REFERENCE(op, bfloat16); \
  BENCHMARK_REFERENCE(op, qint8);    \
  BENCHMARK_REFERENCE(op, quint8);

#define BENCHMARK_REFERENCE_INTEGER(op) BENCHMARK_REFERENCE(op, int32_t);

BENCHMARK_REFERENCE_INTEGER(add);
BENCHMARK_REFERENCE_INTEGER(copysign);
BENCHMARK_REFERENCE_INTEGER(divide);
BENCHMARK_REFERENCE_INTEGER(max);
BENCHMARK_REFERENCE_INTEGER(min);
BENCHMARK_REFERENCE_INTEGER(multiply);
BENCHMARK_REFERENCE_INTEGER(pow);
BENCHMARK_REFERENCE_INTEGER(subtract);

BENCHMARK_REFERENCE_REAL(add);
BENCHMARK_REFERENCE_REAL(copysign);
BENCHMARK_REFERENCE_REAL(divide);
BENCHMARK_REFERENCE_REAL(max);
BENCHMARK_REFERENCE_REAL(min);
BENCHMARK_REFERENCE_REAL(multiply);
BENCHMARK_REFERENCE_REAL(pow);
BENCHMARK_REFERENCE_REAL(squared_difference);
BENCHMARK_REFERENCE_REAL(subtract);
BENCHMARK_REFERENCE_REAL(leaky_relu);

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, init_params_fn, type_a, \
                               type_b, type_c)                                 \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, init_params_fn,         \
                    type_a(), type_b(), type_c())                              \
      ->Apply(Params<type_a, type_b, type_c>)                                  \
      ->UseRealTime();
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
