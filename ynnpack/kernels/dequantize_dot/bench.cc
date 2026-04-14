// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <string>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dequantize_dot/dequantize_dot.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename Output>
void bench(benchmark::State& state, uint64_t arch_flags,
           dequantize_dot_kernel_fn kernel, Output) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  size_t m = state.range(0);
  size_t n = state.range(1);

  Tensor<int32_t> dot({m, n});
  Tensor<int32_t> a_offset({m});
  Tensor<int32_t> b_offset({n});
  Tensor<float> offset({n});
  Tensor<float> a_scale({m});
  Tensor<float> b_scale({n});
  Tensor<Output> output({m, n});
  dequantize_dot_params params = {};

  dot.fill(1);
  a_offset.fill(1);
  b_offset.fill(1);
  offset.fill(1.0f);
  a_scale.fill(1.0f);
  b_scale.fill(1.0f);

  for (auto _ : state) {
    kernel(m, n, dot.stride(0) * sizeof(int32_t), dot.base(),
           a_offset.stride(0) * sizeof(int32_t), a_offset.base(),
           b_offset.stride(0) * sizeof(int32_t), b_offset.base(),
           offset.stride(0) * sizeof(float), offset.base(),
           a_scale.stride(0) * sizeof(float), a_scale.base(),
           b_scale.stride(0) * sizeof(float), b_scale.base(),
           output.stride(0) * sizeof(Output), output.base(), &params);
  }

  const size_t ops = m * n;
  state.counters["Op"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);

  const size_t bytes = m * n * (sizeof(int32_t) + sizeof(Output)) +
                       m * (sizeof(int32_t) + sizeof(float)) +
                       n * (sizeof(int32_t) + sizeof(float) + sizeof(float));
  state.counters["Bytes"] = benchmark::Counter(state.iterations() * bytes,
                                               benchmark::Counter::kIsRate);
}

void Params(benchmark::Benchmark* b) {
  b->ArgNames({"m", "n"});
  b->Args({1, 4096});
  b->Args({4, 1024});
  b->Args({16, 256});
  b->Args({64, 64});
}

#define YNN_DEQUANTIZE_DOT_KERNEL(arch_flags, kernel, type)    \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, type()) \
      ->Apply(Params)                                          \
      ->UseRealTime();
#include "ynnpack/kernels/dequantize_dot/kernels.inc"
#undef YNN_DEQUANTIZE_DOT_KERNEL

}  // namespace ynn
