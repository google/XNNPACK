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
           const unary_params& params, TA, TX) {
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
    kernel(m, n, a.stride_bytes(0), a.base(), x.stride_bytes(0), x.base(),
           &params);
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes = m * n * (sizeof(TA) + sizeof(TX));
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

template <typename T>
void bench_reference(benchmark::State& state, unary_kernel_fn kernel,
                     const unary_params& params, T) {
  return bench(state, arch_flag::none, kernel, params, T{}, T{});
}

template <typename TA, typename TX>
void bench_reference_convert(benchmark::State& state, unary_kernel_fn kernel,
                             const unary_params& params, TA, TX) {
  return bench(state, arch_flag::none, kernel, params, TA{}, TX{});
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
      get_unary_reference_kernel(ynn_unary_##op, type_of<type>()), \
      get_unary_params(ynn_unary_##op), type())                    \
      ->Apply(Params<type, type>)                                  \
      ->UseRealTime();

#define BENCHMARK_FLOAT_REFERENCE(op) \
  BENCHMARK_REFERENCE(op, float);     \
  BENCHMARK_REFERENCE(op, double);

BENCHMARK_FLOAT_REFERENCE(abs);
BENCHMARK_FLOAT_REFERENCE(floor);
BENCHMARK_FLOAT_REFERENCE(ceil);
BENCHMARK_FLOAT_REFERENCE(round);
BENCHMARK_FLOAT_REFERENCE(negate);
BENCHMARK_FLOAT_REFERENCE(square);
BENCHMARK_FLOAT_REFERENCE(square_root);
BENCHMARK_FLOAT_REFERENCE(cube_root);
BENCHMARK_FLOAT_REFERENCE(reciprocal_square_root);
BENCHMARK_FLOAT_REFERENCE(log);
BENCHMARK_FLOAT_REFERENCE(exp);
BENCHMARK_FLOAT_REFERENCE(erf);
BENCHMARK_FLOAT_REFERENCE(tanh);
BENCHMARK_FLOAT_REFERENCE(sign);
BENCHMARK_FLOAT_REFERENCE(sine);
BENCHMARK_FLOAT_REFERENCE(cosine);
BENCHMARK_FLOAT_REFERENCE(sigmoid);
BENCHMARK_FLOAT_REFERENCE(hardswish);

BENCHMARK_REFERENCE(abs, int32_t);
BENCHMARK_REFERENCE(negate, int32_t);
BENCHMARK_REFERENCE(square, int32_t);
BENCHMARK_REFERENCE(sign, int32_t);

#define BENCHMARK_REFERENCE_CONVERT(type_a, type_x)                       \
  BENCHMARK_CAPTURE(                                                      \
      bench_reference_convert, type_a##_##type_x,                         \
      get_convert_reference_kernel(type_of<type_a>(), type_of<type_x>()), \
      get_unary_params(ynn_unary_convert), type_a(), type_x())            \
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

BENCHMARK_REFERENCE_CONVERT(int4x2, int8_t);
BENCHMARK_REFERENCE_CONVERT(int2x4, int8_t);

#define YNN_ELEMENTWISE_KERNEL(arch_flags, kernel, op, type_a, type_x)    \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel,                    \
                    get_unary_params(ynn_unary_##op), type_a(), type_x()) \
      ->Apply(Params<type_a, type_x>)                                     \
      ->UseRealTime();
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
