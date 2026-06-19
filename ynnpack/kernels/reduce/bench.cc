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

  static Shape parse(std::string str) {
    std::replace(str.begin(), str.end(), 'x', ' ');
    std::stringstream ss(str);
    Shape result;
    ss >> result.n >> result.k;
    return result;
  }
};

Shape shape = {256, 256};

template <typename TA, typename TC>
void bench(benchmark::State& state, uint64_t arch_flags,
           reduce_kernel_fn kernel, int k_dim, TA, TC) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  const size_t n = shape.n;
  const size_t k = shape.k;
  state.SetLabel(std::to_string(n) + "x" + std::to_string(k));

  Tensor<TA> a({n, k});
  Tensor<TC> c({2, n});
  a.fill(1);
  c.fill(0);

  if (k_dim != reduce_dim::k1) {
    a = a.reshape({k, n});
  }

  for (auto _ : state) {
    kernel(n, k, a.stride_bytes(0), a.base(), &c(0, 0), &c(1, 0));
  }

  const size_t ops = n * k;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

#define YNN_REDUCE_KERNEL(arch_flags, kernel, k_dim, a_type, c_type)    \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, k_dim, a_type{}, \
                    c_type{})                                           \
      ->UseRealTime();
#include "ynnpack/kernels/reduce/max.inc"
#include "ynnpack/kernels/reduce/min.inc"
#include "ynnpack/kernels/reduce/min_max.inc"
#include "ynnpack/kernels/reduce/sum.inc"
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_REDUCE_KERNEL

}  // namespace ynn

int main(int argc, char** argv) {
  for (int i = 1; i < argc;) {
    if (strncmp(argv[i], "--shape=", 8) == 0) {
      ynn::shape = ynn::Shape::parse(argv[i] + 8);
      std::copy(argv + i + 1, argv + argc, argv + i);
      argc -= 1;
    } else {
      ++i;
    }
  }

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
