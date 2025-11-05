// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/transpose/interleave.h"
#include "ynnpack/kernels/transpose/switch_element_size.h"
#include "ynnpack/kernels/transpose/transpose.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename T>
void bench_impl(benchmark::State& state, uint64_t arch_flags,
                transpose_kernel_fn kernel, T) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }
  constexpr size_t element_count = type_info<T>::element_count();
  const size_t m = state.range(0);
  const size_t n = state.range(1);

  Tensor<T> a({n, m / element_count});
  Tensor<T> x({m, n / element_count});
  memset(a.data(), 0, a.size() * sizeof(T));

  const size_t a_stride = a.stride(0) * sizeof(T);
  const size_t n_bytes_a = m * sizeof(T) / element_count;
  const size_t x_stride = x.stride(0) * sizeof(T);

  for (auto _ : state) {
    kernel(m, n, n_bytes_a, a_stride, a.base(), x_stride, x.base());
  }
}

void bench(benchmark::State& state, uint64_t arch_flags,
           transpose_kernel_fn kernel, size_t elem_size_bits) {
  switch_element_size(elem_size_bits, [&](auto type) {
    bench_impl(state, arch_flags, kernel, type);
  });
}

template <typename T>
void bench_impl(benchmark::State& state, uint64_t arch_flags,
                interleave_kernel_fn kernel, T) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  constexpr size_t element_count = type_info<T>::element_count();
  const size_t factor = state.range(0);
  const size_t m = state.range(1);
  const size_t n = state.range(2);

  Tensor<T> a({m, n / element_count});
  Tensor<T> x({factor * n / element_count});
  memset(a.data(), 0, a.size() * sizeof(T));

  const size_t a_stride = a.stride(0);

  while (state.KeepRunningBatch(m * n)) {
    kernel(factor, m, n, a_stride, a.base(), x.base());
  }
}

void bench(benchmark::State& state, uint64_t arch_flags,
           interleave_kernel_fn kernel, size_t elem_size_bits) {
  switch_element_size(elem_size_bits, [&](auto type) {
    bench_impl(state, arch_flags, kernel, type);
  });
}

void TransposeParams(benchmark::internal::Benchmark* b) {
  b->ArgNames({"m", "n"});
  b->Args({30, 30});
  b->Args({32, 32});
  b->Args({64, 64});
  b->Args({128, 128});
}

#define YNN_TRANSPOSE_KERNEL(arch_flags, kernel, elem_size_bits)       \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, elem_size_bits) \
      ->Apply(TransposeParams)                                         \
      ->UseRealTime();
#include "ynnpack/kernels/transpose/transpose.inc"
#undef YNN_TRANSPOSE_KERNEL

void InterleaveParams(benchmark::internal::Benchmark* b, int factor) {
  b->ArgNames({"factor", "m", "n"});
  int size = 65536;
  if (factor == 0) {
    for (int f = 2; f <= 16; f *= 2) {
      b->Args({f, f / 2, size / f});
      b->Args({f, f, size / f});
    }
  } else {
    b->Args({factor, factor / 2, size / factor});
    b->Args({factor, factor, size / factor});
  }
}

#define YNN_INTERLEAVE_KERNEL(arch_flags, kernel, factor, elem_size_bits) \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, elem_size_bits)    \
      ->Apply([](benchmark::internal::Benchmark* b) {                     \
        InterleaveParams(b, factor);                                      \
      })                                                                  \
      ->UseRealTime();
#include "ynnpack/kernels/transpose/interleave.inc"
#undef YNN_INTERLEAVE_KERNEL

}  // namespace ynn
