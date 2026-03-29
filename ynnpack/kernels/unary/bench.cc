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
           TA, TX) {
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

  for (auto _ : state) {
    kernel(m, n, a.stride(0) * sizeof(TA), a.base(), x.stride(0) * sizeof(TX),
           x.base());
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes = m * n * (sizeof(TA) + sizeof(TX));
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

void bench_reference(benchmark::State& state, unary_kernel_fn kernel) {
  return bench(state, arch_flag::none, kernel, float{}, float{});
}

template <typename TA, typename TX>
void bench_reference_convert(benchmark::State& state, unary_kernel_fn kernel,
                             TA, TX) {
  return bench(state, arch_flag::none, kernel, TA{}, TX{});
}

template <typename A, typename X>
void Params(benchmark::Benchmark* b) {
  b->ArgNames({"m", "n"});
  b->Args({1, 4096});
  b->Args({4, 1024});
  b->Args({16, 256});
}

#define BENCHMARK_REFERENCE(op, type)                              \
  BENCHMARK_CAPTURE(                                               \
      bench_reference, op##_##type,                                \
      get_unary_reference_kernel(ynn_unary_##op, type_of<type>())) \
      ->Apply(Params<type, type>)                                  \
      ->UseRealTime();

BENCHMARK_REFERENCE(abs, float);
BENCHMARK_REFERENCE(floor, float);
BENCHMARK_REFERENCE(ceil, float);
BENCHMARK_REFERENCE(round, float);
BENCHMARK_REFERENCE(negate, float);
BENCHMARK_REFERENCE(square, float);
BENCHMARK_REFERENCE(square_root, float);
BENCHMARK_REFERENCE(cube_root, float);
BENCHMARK_REFERENCE(reciprocal_square_root, float);
BENCHMARK_REFERENCE(log, float);
BENCHMARK_REFERENCE(exp, float);
BENCHMARK_REFERENCE(erf, float);
BENCHMARK_REFERENCE(tanh, float);
BENCHMARK_REFERENCE(sign, float);
BENCHMARK_REFERENCE(sine, float);
BENCHMARK_REFERENCE(cosine, float);
BENCHMARK_REFERENCE(sigmoid, float);
BENCHMARK_REFERENCE(hardswish, float);

BENCHMARK_REFERENCE(abs, int32_t);
BENCHMARK_REFERENCE(negate, int32_t);
BENCHMARK_REFERENCE(square, int32_t);
BENCHMARK_REFERENCE(sign, int32_t);

#define BENCHMARK_REFERENCE_CONVERT(type_a, type_x)                       \
  BENCHMARK_CAPTURE(                                                      \
      bench_reference_convert, type_a##_##type_x,                         \
      get_convert_reference_kernel(type_of<type_a>(), type_of<type_x>()), \
      type_a(), type_x())                                                 \
      ->Apply(Params<type_a, type_x>)                                     \
      ->UseRealTime();

#define BENCHMARK_REFERENCE_CONVERT_FROM(type_a) \
  BENCHMARK_REFERENCE_CONVERT(type_a, float);    \
  BENCHMARK_REFERENCE_CONVERT(type_a, half);     \
  BENCHMARK_REFERENCE_CONVERT(type_a, bfloat16); \
  BENCHMARK_REFERENCE_CONVERT(type_a, int8_t);   \
  BENCHMARK_REFERENCE_CONVERT(type_a, uint8_t);  \
  BENCHMARK_REFERENCE_CONVERT(type_a, int32_t);

BENCHMARK_REFERENCE_CONVERT_FROM(float);
BENCHMARK_REFERENCE_CONVERT_FROM(half);
BENCHMARK_REFERENCE_CONVERT_FROM(bfloat16);
BENCHMARK_REFERENCE_CONVERT_FROM(int8_t);
BENCHMARK_REFERENCE_CONVERT_FROM(uint8_t);
BENCHMARK_REFERENCE_CONVERT_FROM(int32_t);

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, type_a, type_x)     \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, type_a(), type_x()) \
      ->Apply(Params<type_a, type_x>)                                      \
      ->UseRealTime();
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
