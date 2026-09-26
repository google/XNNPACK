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

namespace {

size_t calculate_k_chunk(size_t l2_cache_size, size_t m, span<const size_t> ks,
                         size_t b_elem_size) {
  const size_t k1 = ks[0];
  size_t k2 = 1;
  for (size_t i = 1; i < ks.size(); ++i) {
    k2 *= ks[i];
  }
  const size_t block_n_estimate =
      b_elem_size < 4 ? 16 : (8 < m && m <= 32 ? 64 : 32);
  const size_t bytes_per_k1 = k2 * block_n_estimate * b_elem_size;
  if (k1 * bytes_per_k1 <= l2_cache_size) return k1;

  constexpr size_t granularity = 64;
  const size_t groups = ceil_div(k1, granularity);
  const size_t max_groups =
      std::max<size_t>(1, floor_div(l2_cache_size, granularity * bytes_per_k1));
  // Balance the chunks of k, so we don't have a small chunk at the end.
  const size_t chunks = ceil_div(groups, max_groups);
  return granularity * ceil_div(groups, chunks);
}

}  // namespace

span<dot_loop> schedule_dot(size_t l2_cache_size, size_t m, size_t n,
                            span<const size_t> k, size_t block_m,
                            size_t block_n, size_t block_k, size_t a_elem_size,
                            size_t b_elem_size, bool pack_a, bool pack_b,
                            dot_loop* storage) {
  dot_loop* begin = storage;
  dot_loop* loop = begin;

  assert(!k.empty() && k.size() <= 3);
  const size_t k_chunk = calculate_k_chunk(l2_cache_size, m, k, b_elem_size);
  if (k_chunk < k[0]) {
    assert(k_chunk % block_k == 0);
    *loop++ = dot_loop{dot_loop::k, k_chunk / block_k};
  }
  if (pack_a || !pack_b || n * b_elem_size <= m * a_elem_size) {
    // Either:
    // - A is packed one panel (block_m x chunk of k) at a time, so we want to
    //   reuse each packed panel of A for all of n.
    // - B is not packed, so we want to keep n contiguous for the kernel.
    // - Tiles of B are smaller than tiles of A, we should assume B fits in
    //   cache.
    if (m > block_m) *loop++ = dot_loop{dot_loop::m, 1};
    if (n > block_n) *loop++ = dot_loop{dot_loop::n, 1};
  } else {
    // Tiles of A are smaller than tiles of B, we should assume A fits in
    // cache.
    if (n > block_n) *loop++ = dot_loop{dot_loop::n, 1};
    if (m > block_m) *loop++ = dot_loop{dot_loop::m, 1};
  }
  if (loop == begin) {
    // We need to make at least one loop for `run_dot`.
    *loop++ = dot_loop{dot_loop::m, 1};
  }

  return {begin, loop};
}

}  // namespace ynn
