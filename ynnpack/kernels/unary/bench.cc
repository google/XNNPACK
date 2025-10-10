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
#include "ynnpack/kernels/unary/unary.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename TA, typename TX>
void bench(benchmark::State& state, uint64_t arch_flags, unary_kernel_fn kernel,
           init_unary_params_fn init_params, TA, TX) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  size_t m = state.range(0);
  size_t n = state.range(1);

  Tensor<TA> a({m, n});
  Tensor<TX> x({m, n});
  a.fill(1);
  x.fill(1);
  broadcast_extent_1(a);
  broadcast_extent_1(x);

  unary_params params;
  if (init_params) {
    // Use non-trivial quantization in case the kernel tries to optimize for
    // that.
    init_params(0.5f, 1, 2.0f, 3, params);
  }

  for (auto _ : state) {
    kernel(m, n, a.stride(0) * sizeof(TA), a.base(), x.stride(0) * sizeof(TX),
           x.base(), &params);
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes = m * n * (sizeof(TA) + sizeof(TX));
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

template <typename TA, typename TX>
void bench(benchmark::State& state, uint64_t arch_flags,
           const unary_kernel* kernel, TA, TX) {
  if (!kernel) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }
  bench(state, arch_flags, kernel->op, kernel->init_params, TA(), TX());
}

template <typename A, typename X>
void Params(benchmark::internal::Benchmark* b) {
  b->ArgNames({"m", "n"});
  b->Args({1, 4096});
  b->Args({4, 1024});
  b->Args({16, 256});
}

#define BENCHMARK_REFERENCE(op, type)                                     \
  BENCHMARK_CAPTURE(                                                      \
      bench, reference_unary_##op##_##type, arch_flag::none,              \
      get_unary_reference_kernel(ynn_unary_##op, type(), type()), type(), \
      type())                                                             \
      ->Apply(Params<type, type>)                                         \
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

BENCHMARK_REFERENCE_INTEGER(abs);
BENCHMARK_REFERENCE_INTEGER(negate);
BENCHMARK_REFERENCE_INTEGER(square);
BENCHMARK_REFERENCE_INTEGER(sign);

BENCHMARK_REFERENCE_REAL(abs);
BENCHMARK_REFERENCE_REAL(floor);
BENCHMARK_REFERENCE_REAL(ceil);
BENCHMARK_REFERENCE_REAL(round);
BENCHMARK_REFERENCE_REAL(negate);
BENCHMARK_REFERENCE_REAL(square);
BENCHMARK_REFERENCE_REAL(square_root);
BENCHMARK_REFERENCE_REAL(cube_root);
BENCHMARK_REFERENCE_REAL(reciprocal_square_root);
BENCHMARK_REFERENCE_REAL(log);
BENCHMARK_REFERENCE_REAL(exp);
BENCHMARK_REFERENCE_REAL(erf);
BENCHMARK_REFERENCE_REAL(tanh);
BENCHMARK_REFERENCE_REAL(sign);
BENCHMARK_REFERENCE_REAL(sine);
BENCHMARK_REFERENCE_REAL(cosine);
BENCHMARK_REFERENCE_REAL(sigmoid);
BENCHMARK_REFERENCE_REAL(hardswish);

#define BENCHMARK_REFERENCE_CONVERT(op, type_a, type_x)                   \
  BENCHMARK_CAPTURE(                                                      \
      bench, reference_unary_##op##_##type_a##_##type_x, arch_flag::none, \
      get_unary_reference_kernel(ynn_unary_##op, type_a(), type_x()),     \
      type_a(), type_x())                                                 \
      ->Apply(Params<type_a, type_x>)                                     \
      ->UseRealTime();

#define BENCHMARK_REFERENCE_CONVERT_FROM(type_a)          \
  BENCHMARK_REFERENCE_CONVERT(convert, type_a, float);    \
  BENCHMARK_REFERENCE_CONVERT(convert, type_a, half);     \
  BENCHMARK_REFERENCE_CONVERT(convert, type_a, bfloat16); \
  BENCHMARK_REFERENCE_CONVERT(convert, type_a, qint8);    \
  BENCHMARK_REFERENCE_CONVERT(convert, type_a, quint8);   \
  BENCHMARK_REFERENCE_CONVERT(convert, type_a, int32_t);

BENCHMARK_REFERENCE_CONVERT_FROM(float);
BENCHMARK_REFERENCE_CONVERT_FROM(half);
BENCHMARK_REFERENCE_CONVERT_FROM(bfloat16);
BENCHMARK_REFERENCE_CONVERT_FROM(qint8);
BENCHMARK_REFERENCE_CONVERT_FROM(quint8);
BENCHMARK_REFERENCE_CONVERT_FROM(int32_t);

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, init_params_fn, type_a, \
                               type_x)                                         \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, init_params_fn,         \
                    type_a(), type_x())                                        \
      ->Apply(Params<type_a, type_x>)                                          \
      ->UseRealTime();
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
