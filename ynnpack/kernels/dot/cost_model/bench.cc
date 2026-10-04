// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dot/cost_model/cost_model.h"
#include "ynnpack/kernels/dot/dot.h"
#include <benchmark/benchmark.h>

namespace ynn {

template <typename TA, typename TB, typename TC>
void dot(benchmark::State& state, uint64_t arch_flags, dot_kernel_fn kernel,
         size_t block_m, size_t block_n, size_t block_k, size_t tile_m,
         size_t tile_n, size_t tile_k, uint32_t flags,
         const dot_cost_model& cost_model, TA, TB, TC) {
  const size_t m = block_m;
  const size_t n = state.range(0);
  const size_t k = state.range(1);

  // Record the problem shape and the kernel's block and tile parameters as
  // counters, so the cost model fitter can read them directly from the CSV
  // columns. The alternative, parsing them out of the benchmark name, is
  // impossible for kernels like SME whose block and tile sizes are only known
  // at runtime. These are set before the skip check below so that every run
  // (including skipped ones) carries them, which keeps the CSV reporter's
  // counter columns consistent across all runs.
  state.counters["m"] = static_cast<double>(m);
  state.counters["n"] = static_cast<double>(n);
  state.counters["k"] = static_cast<double>(k);
  state.counters["block_m"] = static_cast<double>(block_m);
  state.counters["block_n"] = static_cast<double>(block_n);
  state.counters["block_k"] = static_cast<double>(block_k);
  state.counters["tile_m"] = static_cast<double>(tile_m);
  state.counters["tile_n"] = static_cast<double>(tile_n);
  state.counters["tile_k"] = static_cast<double>(tile_k);

  if (!is_arch_supported(arch_flags)) {
    state.SkipWithMessage("Unsupported hardware");
    return;
  }

  const bool transpose_a = flags & dot_flag::transpose_a;

  Tensor<TA> a({align_up(m, tile_m), k});
  Tensor<TB> b({k, align_up(n, tile_n * tile_k)},
               Alignment{.bytes = tile_n * tile_k * sizeof(TB)});
  Tensor<TC> c({m, n});
  a.fill(1);
  b.fill(1);
  c.fill(0);
  b = b.crop_padding({0, 0}, {b.extent(0) - k, b.extent(1) - n});

  if (transpose_a) {
    // This mangles the data, but we don't care here.
    a = a.reshape({k / tile_k, m * tile_k});
  }

  for (auto _ : state) {
    dot_kernel_state kernel_state = {};
    kernel(m, n, 1, 1, k, a.stride_bytes(0) / (transpose_a ? tile_k : 1), 0, 0,
           a.base(), 0, 0, b.stride_bytes(0) / tile_k, b.base(),
           /*init_c_stride_m=*/0, nullptr, c.stride_bytes(0), c.base(),
           &kernel_state);
  }

  // Check that the kernel didn't compute the wrong thing. We assume the kernel
  // is correct, but we have some logic here that needs validation too. We
  // filled a and b with 1, so the result should be k everywhere.
  if (!std::all_of(c.begin(), c.end(), [=](TC x) { return x == k; })) {
    state.SkipWithError("Incorrect result");
  }
}

template <typename TA, typename TB, typename TC>
void dot_args(benchmark::Benchmark* b, size_t block_m, size_t block_n,
              size_t block_k, TA, TB, TC) {
  constexpr size_t max_l1_bytes = 24 * 1024;
  const size_t block_a_bytes =
      block_m * block_k * sizeof(TA) / type_info<TA>::element_count();

  for (size_t mult : {1, 2, 4, 8}) {
    const size_t n = mult * block_n;
    const size_t c_bytes =
        block_m * n * sizeof(TC) / type_info<TC>::element_count();
    const size_t block_b_bytes =
        block_k * n * sizeof(TB) / type_info<TB>::element_count();
    const size_t bytes_per_block_k = block_a_bytes + block_b_bytes;
    if (bytes_per_block_k == 0) {
      b->Args({static_cast<int64_t>(n), 0});
      break;
    }
    if (c_bytes + bytes_per_block_k > max_l1_bytes) {
      if (mult == 1) {
        b->Args({static_cast<int64_t>(block_n), static_cast<int64_t>(block_k)});
      }
      break;
    }

    const size_t max_ab_bytes = max_l1_bytes - c_bytes;
    const size_t max_k_mult = max_ab_bytes / bytes_per_block_k;

    std::set<size_t> km_set = {
        1,
        std::max<size_t>(1, max_k_mult / 4),
        std::max<size_t>(1, max_k_mult / 2),
        std::max<size_t>(1, (3 * max_k_mult) / 4),
        max_k_mult,
    };
    for (size_t km : km_set) {
      b->Args({static_cast<int64_t>(n), static_cast<int64_t>(km * block_k)});
    }
  }
}

const dot_cost_models& cost_models = get_dot_cost_models();

#define YNN_DOT_KERNEL(arch_flags, kernel, block_m, block_n, block_k, tile_m, \
                       tile_n, tile_k, flags, a_type, b_type, c_type,         \
                       cost_model)                                            \
  BENCHMARK_CAPTURE(dot, kernel, arch_flags, kernel, block_m, block_n,        \
                    block_k, tile_m, tile_n, tile_k, flags, cost_model,       \
                    a_type(), b_type(), c_type())                             \
      ->Apply([](benchmark::Benchmark* b) {                                   \
        dot_args(b, block_m, block_n, block_k, a_type(), b_type(), c_type()); \
      })                                                                      \
      ->UseRealTime();
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL

}  // namespace ynn
