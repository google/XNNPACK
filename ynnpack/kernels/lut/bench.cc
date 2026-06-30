// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/lut/lut.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename Idx, typename Elem>
void bench(benchmark::State& state, uint64_t arch_flags, lut_kernel_fn kernel,
           Idx, Elem) {
  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  using IdxInfo = type_info<Idx>;

  size_t n = state.range(0);

  // We assume that for 8-bit indices, we don't need to check bounds, but for
  // larger types, we do.
  Buffer<Elem> lut(256);
  std::iota(lut.begin(), lut.end(), 0);
  Buffer<Idx> idx(n);
  std::mt19937 rng;
  fill_random(idx.data(), idx.size(), rng);
  for (size_t i = 0; i < idx.size(); ++i) {
    idx[i] = static_cast<typename IdxInfo::element_type>(idx[i] % lut.size());
  }

  Buffer<Elem> x(n);

  for (auto _ : state) {
    kernel(n, idx.data(), lut.size(), lut.data(), x.data());
  }
}

template <typename Idx>
void bench(benchmark::State& state, uint64_t arch_flags, lut_kernel_fn kernel,
           Idx, size_t elem_size_bits) {
  switch (elem_size_bits) {
    case 8:
      bench(state, arch_flags, kernel, Idx(), uint8_t());
      break;
    case 16:
      bench(state, arch_flags, kernel, Idx(), uint16_t());
      break;
    case 32:
      bench(state, arch_flags, kernel, Idx(), uint32_t());
      break;
    default:
      state.SkipWithMessage("Unsupported element size");
  }
}

template <typename Idx, size_t ElemSizeBits>
void Params(benchmark::Benchmark* b) {
  b->ArgNames({"n"});
  b->Arg(1024 * 1024);
}

#define YNN_LUT_KERNEL(arch_flags, kernel, idx_type, elem_size_bits) \
  BENCHMARK_CAPTURE(bench, kernel, arch_flags, kernel, idx_type{},   \
                    elem_size_bits)                                  \
      ->Apply(Params<idx_type, elem_size_bits>)                      \
      ->UseRealTime();
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_LUT_KERNEL

}  // namespace ynn
