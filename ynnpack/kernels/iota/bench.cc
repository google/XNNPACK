// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "ynnpack/base/arch.h"
#include "ynnpack/kernels/iota/iota.h"
#include <benchmark/benchmark.h>

namespace ynn {
namespace {

template <typename T>
void bench_iota(benchmark::State& state, uint64_t arch_flags,
                iota_kernel_fn kernel, T stride) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  size_t n = state.range(0);
  std::vector<T> output(n);

  T begin = 0;
  for (auto _ : state) {
    kernel(n, &begin, &stride, output.data());
  }

  state.counters["Op"] =
      benchmark::Counter(state.iterations() * n, benchmark::Counter::kIsRate);
  state.counters["Bytes"] = benchmark::Counter(
      state.iterations() * n * sizeof(T), benchmark::Counter::kIsRate);
}

#define YNN_IOTA_KERNEL(arch, name, type)                    \
  static void bench_##name##_fill(benchmark::State& state) { \
    bench_iota<type>(state, arch, name, 0);                  \
  }                                                          \
  BENCHMARK(bench_##name##_fill)->Arg(1024)->UseRealTime();  \
  static void bench_##name##_iota(benchmark::State& state) { \
    bench_iota<type>(state, arch, name, 1);                  \
  }                                                          \
  BENCHMARK(bench_##name##_iota)->Arg(1024)->UseRealTime();

#include "ynnpack/kernels/iota/kernels.inc"
#undef YNN_IOTA_KERNEL

}  // namespace
}  // namespace ynn
