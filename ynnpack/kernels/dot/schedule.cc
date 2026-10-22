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

span<dot_loop> schedule_dot(span<const size_t> cache_budgets, size_t m,
                            size_t n, span<const size_t> ks, size_t block_m,
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
  auto make_m_loop = [&](size_t step_m) {
    assert(step_m % block_m == 0);
    if (step_m == 0 || m <= step_m) return;
    *loop++ = dot_loop{dot_loop::m, step_m};
    m = step_m;
  };
  auto make_n_loop = [&](size_t step_n) {
    assert(step_n % block_n == 0);
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
  // sizes. Target `dot_k_step_bytes` per column of B aligned to 64 elements so
  // every step lands on a packed k-tile boundary.
  constexpr size_t k_alignment = 64;
  make_k_loop(
      align_up(std::max<size_t>(k_alignment,
                                floor_div(dot_k_step_bytes, k2 * b_elem_size)),
               k_alignment));

  const bool tile_n_for_cache = n * b_elem_size <= m * a_elem_size;
  const size_t a_bytes = m * k1 * k2 * a_elem_size;
  const size_t b_bytes = n * k1 * k2 * b_elem_size;
  const bool fits_in_l1 =
      !cache_budgets.empty() && a_bytes + b_bytes <= cache_budgets[0];
  if (m > block_m && n > block_n && !cache_budgets.empty() && !fits_in_l1) {
    const size_t a_block_bytes = k1 * k2 * block_m * a_elem_size;
    const size_t b_block_bytes = k1 * k2 * block_n * b_elem_size;
    for (size_t i = cache_budgets.size(); i-- > 0;) {
      // Skip unconfigured cache levels, or outer cache levels whose budget is
      // no larger than L1's (which can happen when an L2 cache is shared
      // across many cores, or in containerized environments where sysfs reports
      // all processors on the socket as sharing a private L2 cache).
      const size_t cache_budget = cache_budgets[i];
      if (cache_budget == 0 || (i > 0 && cache_budget <= cache_budgets[0])) {
        continue;
      }
      if (tile_n_for_cache) {
        // Tiles of B are smaller than tiles of A; tile n so the active B
        // working set fits in `cache_budget` alongside one block of A across
        // the m loop.
        if (cache_budget > a_block_bytes) {
          const size_t sub_blocks_n =
              floor_div(cache_budget - a_block_bytes, b_block_bytes);
          if (sub_blocks_n > 1) {
            make_n_loop(sub_blocks_n * block_n);
          }
        }
      } else {
        // Tiles of A are smaller than tiles of B; tile m so the active A
        // working set fits in `cache_budget` alongside one block of B across
        // the n loop.
        if (cache_budget > b_block_bytes) {
          const size_t sub_blocks_m =
              floor_div(cache_budget - b_block_bytes, a_block_bytes);
          if (sub_blocks_m > 1) {
            make_m_loop(sub_blocks_m * block_m);
          }
        }
      }
    }
  }
  if (fits_in_l1 ? (m * block_n < n * block_m) : tile_n_for_cache) {
    make_m_loop(block_m);
    make_n_loop(block_n);
  } else {
    make_n_loop(block_n);
    make_m_loop(block_m);
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
