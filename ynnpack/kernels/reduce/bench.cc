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
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include <benchmark/benchmark.h>

namespace ynn {

struct Shape {
  int n, k3, k2, k1;

  static Shape parse(std::string str) {
    std::replace(str.begin(), str.end(), 'x', ' ');
    std::stringstream ss(str);
    Shape result;
    ss >> result.n >> result.k3 >> result.k2 >> result.k1;
    return result;
  }
};

Shape shape = {256, 1, 1, 256};

template <typename TA, typename TC>
void bench(benchmark::State& state, uint64_t arch_flags,
           unary_reduce_kernel_fn kernel, TA, TC) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  const size_t n = shape.n;
  const size_t k3 = shape.k3;
  const size_t k2 = shape.k2;
  const size_t k1 = shape.k1;
  state.SetLabel(std::to_string(n) + "x" + std::to_string(k3) + "x" +
                 std::to_string(k2) + "x" + std::to_string(k1));

  Tensor<TA> a({k3, k2, n, k1});
  Tensor<TC> c({2, n});
  a.fill(1);
  c.fill(0);

  for (auto _ : state) {
    kernel(n, k3, k2, k1, a.stride(2) * sizeof(TA), a.stride(0) * sizeof(TA),
           a.stride(1) * sizeof(TA), a.base(), c.stride(0) * sizeof(TC),
           c.base());
  }

  const size_t ops = n * k3 * k2 * k1;
  ;
  state.counters["OP"] =
      benchmark::Counter(state.iterations() * ops, benchmark::Counter::kIsRate);
}

#define YNN_UNARY_REDUCE_KERNEL(arch_flags, kernel, a_type, c_type)        \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, a_type(), c_type()) \
      ->UseRealTime();
#include "ynnpack/kernels/reduce/max.inc"
#include "ynnpack/kernels/reduce/min.inc"
#include "ynnpack/kernels/reduce/min_max.inc"
#include "ynnpack/kernels/reduce/sum.inc"
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_UNARY_REDUCE_KERNEL

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
