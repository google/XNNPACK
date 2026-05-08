// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include <benchmark/benchmark.h>

namespace ynn {

struct Shape {
  int n, k;
};

template <typename TA, typename TC>
void bench(benchmark::State& state, uint64_t arch_flags,
           reduce_kernel_fn kernel, bool is_k1, TA, TC) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }
  const size_t n = state.range(0);
  const size_t k = state.range(1);

  Tensor<TA> a({n, k});
  Tensor<TC> c({2, is_k1 ? n : k});
  a.fill(1);
  c.fill(0);

  for (auto _ : state) {
    if (is_k1) {
      kernel(n, k, a.stride_bytes(0), a.base(), c.stride_bytes(0), c.base());
    } else {
      kernel(k, n, a.stride_bytes(0), a.base(), c.stride_bytes(0), c.base());
    }
  }

  const size_t ops = n * k;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

void reduce_k1_args(benchmark::Benchmark* b) {
  b->ArgPair(1, 4096);
  b->ArgPair(32, 32);
}

void reduce_kn_args(benchmark::Benchmark* b) { b->ArgPair(256, 32); }

#define YNN_REDUCE_K1_KERNEL(arch_flags, kernel, a_type, c_type)       \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, true, a_type(), \
                    c_type())                                          \
      ->UseRealTime()                                                  \
      ->Apply(reduce_k1_args);
#define YNN_REDUCE_KN_KERNEL(arch_flags, kernel, a_type, c_type)        \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, false, a_type(), \
                    c_type())                                           \
      ->UseRealTime()                                                   \
      ->Apply(reduce_kn_args);
#include "ynnpack/kernels/reduce/max.inc"
#include "ynnpack/kernels/reduce/min.inc"
#include "ynnpack/kernels/reduce/min_max.inc"
#include "ynnpack/kernels/reduce/sum.inc"
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL

}  // namespace ynn
