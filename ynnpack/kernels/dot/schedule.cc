// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <cassert>
#include <cstddef>

#include "ynnpack/base/arithmetic.h"
#include "slinky/base/span.h"

namespace ynn {

slinky::span<dot_loop> schedule_dot(slinky::span<const size_t> cache_sizes,
                                    size_t m, size_t n, size_t k1, size_t k2,
                                    size_t k3, size_t block_m, size_t block_n,
                                    size_t block_k, size_t a_elem_size,
                                    size_t b_elem_size, dot_loop* storage) {
  dot_loop* begin = storage;
  dot_loop* loop = begin;

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
  auto make_k_loop = [&](size_t blocks) {
    if (blocks == 0 || k1 <= block_k * blocks) return;
    *loop++ = dot_loop{dot_loop::k, blocks};
    k1 = block_k * blocks;
  };

  for (size_t cache_size : cache_sizes) {
    // TODO(b/447988052): We can be way smarter about this than we are now.
    make_k_loop(
        floor_div(cache_size, k2 * k3 * block_n * b_elem_size * block_k));
    if (k1 * k2 * k3 * n * b_elem_size <= m * k1 * k2 * k3 * a_elem_size) {
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

  slinky::span<dot_loop> loops = {begin, loop};
  return loops;
}

}  // namespace ynn
