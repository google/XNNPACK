// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <cassert>
#include <cstddef>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/span.h"

namespace ynn {

span<dot_loop> schedule_dot(span<const size_t> cache_sizes, size_t m, size_t n,
                            span<const size_t> ks, size_t block_m,
                            size_t block_n, size_t block_k, size_t a_elem_size,
                            size_t b_elem_size, dot_loop* storage) {
  dot_loop* begin = storage;
  dot_loop* loop = begin;

  size_t k1 = ks[0];
  size_t k2 = 1;
  for (size_t i = 1; i < ks.size(); ++i) {
    k2 *= ks[i];
  }

  // When we make a loop in a dimension, the extent of that dimension becomes
  // the step size of that loop.
  auto make_m_loop = [&](size_t blocks) {
    if (blocks == 0 || m <= block_m * blocks) return;
    *loop++ = dot_loop{dot_loop::m, blocks};
    m = block_m * blocks;
  };
  auto make_n_loop = [&](size_t blocks) {
    if (blocks == 0 || n <= block_n * blocks) return;
    *loop++ = dot_loop{dot_loop::n, blocks};
    n = block_n * blocks;
  };
  auto make_k_loop = [&](size_t blocks_max) {
    if (blocks_max == 0) return;
    const size_t k_blocks = ceil_div(k1, block_k);
    // Fits in a single iteration — no loop needed.
    if (k_blocks <= blocks_max) return;
    // Split into N near-equal iterations rather than one cache-max iter plus
    // a tail. Two reasons: (1) a small tail amortises kernel-call overhead
    // poorly, and (2) a kc that almost-fills the cache with no headroom
    // causes the B stripe to spill into the outer cache, which has much
    // higher variance than a slightly smaller stripe that fits cleanly.
    const size_t niter = ceil_div(k_blocks, blocks_max);
    const size_t blocks = ceil_div(k_blocks, niter);
    *loop++ = dot_loop{dot_loop::k, blocks};
    k1 = block_k * blocks;
  };

  for (size_t cache_size : cache_sizes) {
    // Size kc so that a (kc × n) stripe of B fits in this cache. Inside each
    // outer k-iteration we sweep all (m, n) tiles; the B stripe is loaded
    // once from memory on the first m-iteration and reused from cache on
    // subsequent m-iterations, so what matters is kc×n×b_elem_size, not the
    // per-kernel-call kc×block_n stripe.
    // ~6% headroom so the B stripe doesn't exactly fill the budget — at
    // that boundary, concurrent A/C/TLB traffic evicts B into the outer
    // cache, hurting both mean and run-to-run variance.
    const size_t kc_budget = cache_size - cache_size / 16;
    make_k_loop(floor_div(kc_budget, k2 * n * b_elem_size * block_k));
    if (n * b_elem_size <= m * a_elem_size) {
      // Tiles of B are smaller than tiles of A, we should assume B fits in
      // cache.
      make_m_loop(1);
      make_n_loop(1);
    } else {
      // Tiles of A are smaller than tiles of B, we should assume A fits in
      // cache.
      make_n_loop(1);
      make_m_loop(1);
    }
  }
  assert(m <= block_m);
  assert(n <= block_n);
  if (loop == begin) {
    // We need to make at least one loop for `run_dot`.
    *loop++ = dot_loop{dot_loop::m, 1};
  }

  span<dot_loop> loops = {begin, loop};
  return loops;
}

}  // namespace ynn
