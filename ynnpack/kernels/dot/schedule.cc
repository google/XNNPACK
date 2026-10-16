// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/schedule.h"

#include <algorithm>
#include <cassert>
#include <cstddef>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/span.h"

namespace ynn {

span<dot_loop> schedule_dot(span<const size_t> cache_sizes, size_t m, size_t n,
                            span<const size_t> ks, size_t block_m,
                            size_t block_n, size_t a_elem_size,
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
    const size_t step_m = block_m * blocks;
    if (step_m == 0 || m <= step_m) return;
    *loop++ = dot_loop{dot_loop::m, step_m};
    m = step_m;
  };
  auto make_n_loop = [&](size_t blocks) {
    const size_t step_n = block_n * blocks;
    if (step_n == 0 || n <= step_n) return;
    *loop++ = dot_loop{dot_loop::n, step_n};
    n = step_n;
  };
  auto make_k_loop = [&](size_t step_k) {
    if (step_k == 0 || k1 <= step_k) return;
    *loop++ = dot_loop{dot_loop::k, step_k};
    k1 = step_k;
  };

  // For consistent arithmetic across CPUs, `step_k` must depend only on `ks`
  // and `b_elem_size` rather than hardware-dependent cache or kernel block
  // sizes. Target 2048 bytes per column of B (128 KiB for 64 columns) aligned
  // to 64 elements so every step lands on a packed k-tile boundary.
  constexpr size_t k_target_bytes = 2048;
  constexpr size_t k_alignment = 64;
  make_k_loop(
      align_up(std::max<size_t>(1, floor_div(k_target_bytes, k2 * b_elem_size)),
               k_alignment));

  const size_t a_block_bytes = k1 * k2 * block_m * a_elem_size;
  const size_t b_block_bytes = k1 * k2 * block_n * b_elem_size;
  for (size_t cache_size : cache_sizes) {
    if (n * b_elem_size <= m * a_elem_size) {
      // Tiles of B are smaller than tiles of A; tile n so the active B working
      // set fits in `cache_size` across the m loop.
      const size_t sub_blocks_n =
          std::max<size_t>(1, floor_div(cache_size, b_block_bytes));
      make_n_loop(sub_blocks_n);
    } else {
      // Tiles of A are smaller than tiles of B; tile m so the active A working
      // set fits in `cache_size` across the n loop.
      const size_t sub_blocks_m =
          std::max<size_t>(1, floor_div(cache_size, a_block_bytes));
      make_m_loop(sub_blocks_m);
    }
  }
  if (n * b_elem_size <= m * a_elem_size) {
    make_m_loop(1);
    make_n_loop(1);
  } else {
    make_n_loop(1);
    make_m_loop(1);
  }
  assert(m <= block_m);
  assert(n <= block_n);
  if (loop == begin) {
    // We need to make at least one loop for `run_dot`.
    *loop++ = dot_loop{dot_loop::m, block_m};
  }

  span<dot_loop> loops = {begin, loop};
  return loops;
}

}  // namespace ynn
