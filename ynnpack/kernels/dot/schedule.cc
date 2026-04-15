// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/span.h"

namespace ynn {

// Generates the hierarchical cache-tiling schedule for the dot microkernel.
//
// The goal of this function is to slice a matrix multiplication
// (m x n x k1 x k2 x k3) into smaller chunks that fit into the caches. This is
// done by making loops over the dimensions m, n, and k, and tiling the
// dimensions to block_m, block_n, and block_k respectively.
//
// Heuristics and their effects:
// 1. Fast-path Cache Optimization: If all three matrices A, B, and C fit in
//    the L2 cache, we bypass K-tiling completely and only tile M and N.
//    This eliminates loop overhead for small tasks.
// 2. Outermost K-Loop: When K is large, we slice it into chunks (`k_blocks`)
//    that ensure the working set of A and B remain resident in the L2 cache.
// 3. Cache-Aware Working Sets: Depending on whether M or N is larger, we assume
//    the smaller matrix tile stays stationary in L2 while the larger one
//    streams past it.
span<dot_loop> schedule_dot(const cpu_info& cpu_info, size_t m, size_t n,
                            span<const size_t> ks, size_t block_m,
                            size_t block_n, size_t block_k, size_t a_elem_size,
                            size_t b_elem_size, size_t c_elem_size,
                            dot_loop* storage) {
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
  auto make_k_loop = [&](size_t blocks) {
    if (blocks == 0 || k1 <= block_k * blocks) return;
    *loop++ = dot_loop{dot_loop::k, blocks};
    k1 = block_k * blocks;
  };
  auto finish_schedule = [&]() -> span<dot_loop> {
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
    assert(m <= block_m);
    assert(n <= block_n);
    if (loop == begin) {
      // We need to make at least one loop for `run_dot`.
      *loop++ = dot_loop{dot_loop::m, 1};
    }
    return {begin, loop};
  };

  size_t cache_capacity = cpu_info.cache_sizes[1];
  // Fast-path Cache Optimization: If all three matrices A, B, and C fit in
  // the L2 cache (or L3 if packed), we bypass K-tiling completely and only
  // tile M and N.
  const size_t a_size = m * k1 * k2 * a_elem_size;
  const size_t b_size = n * k1 * k2 * b_elem_size;
  const size_t c_size = m * n * c_elem_size;
  if (a_size + b_size + c_size <= cache_capacity) {
    return finish_schedule();
  }

  // Outermost K-Loop: When K is large, we slice it into chunks (`k_blocks`)
  // that ensure the working set of A and B remain resident in the L2 cache
  // (or L3 if packed).
  size_t k_blocks;
  if (n * b_elem_size <= m * a_elem_size) {
    // Tiles of B are smaller than A. We reuse B in the L2 cache while A
    // streams. Working set is B (n * k_slice) and A micro-block (block_m *
    // k_slice).
    size_t footprint_per_k =
        k2 * (n * b_elem_size + block_m * a_elem_size) * block_k;
    k_blocks = floor_div(cache_capacity, footprint_per_k);
  } else {
    // Tiles of A are smaller than B. We reuse A in the L2 cache while B
    // streams. Working set is A (m * k_slice) and B micro-block (block_n *
    // k_slice).
    size_t footprint_per_k =
        k2 * (m * a_elem_size + block_n * b_elem_size) * block_k;
    k_blocks = floor_div(cache_capacity, footprint_per_k);
  }
  if (k_blocks == 0) {
    k_blocks = 1;
  }
  make_k_loop(k_blocks);
  return finish_schedule();
}

}  // namespace ynn
