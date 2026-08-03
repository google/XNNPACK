// Copyright 2026 Google LLC
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

// Generates the hierarchical cache-tiling schedule for the dot microkernel.
//
// The goal of this function is to slice a matrix multiplication
// (m x n x k1 x k2 x k3) into smaller chunks that fit into the cache hierarchy.
// This is done by looping over the dimensions m, n, and k, and tiling the
// dimensions to block_m, block_n, and block_k respectively.
//
// We first decide which matrix A and B is smaller. We then assume the smaller
// matrix is stationary in the L2 cache, and the larger matrix is streaming.
// The large matrix streams panels of size `proxy_m x k_tile` or `proxy_n x
// k_tile`. Our active working set is the smaller matrix plus a panel proxy of
// the larger matrix.
//
// If the active working set can fit in the L2 cache, we bypass k-tiling
// completely (outermost k-loop). Othwerwise, we slice k into chunks that are
// multiples of `min_block_k` to ensure the active working set remains
// resident in the L2 cache.
//
// We then perform secondary cache blocking on the smaller (stationary)
// dimension (N or M) to maximize reuse in the shared L3 cache.
span<dot_loop> schedule_dot(const cpu_info& cpu, size_t m, size_t n,
                            span<const size_t> ks, size_t block_m,
                            size_t block_n, size_t block_k, size_t a_elem_size,
                            size_t b_elem_size, dot_loop* storage) {
  dot_loop* begin = storage;
  dot_loop* loop = begin;

  size_t k1 = ks[0];
  // Scale the k-step size by data-type. The smaller the data type, the
  // higher the step size.
  size_t min_k_step = block_k * a_elem_size;
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
  auto make_k_loop = [&](size_t k_step) {
    if (k_step == 0 || k1 <= k_step) return;
    size_t blocks = k_step / min_k_step;
    if (blocks == 0) return;
    *loop++ = dot_loop{dot_loop::k, blocks};
    k1 = k_step;
  };

  const bool is_n_blocked = (n * b_elem_size <= m * a_elem_size);
  auto finish_schedule = [&]() -> span<dot_loop> {
    if (is_n_blocked) {
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
      *loop++ = dot_loop{dot_loop::m, 1};
    }
    return {begin, loop};
  };

  const size_t cache_capacity = cpu.cache_sizes[1];
  const size_t a_size = m * k1 * k2 * a_elem_size;
  const size_t b_size = n * k1 * k2 * b_elem_size;

  // We use proxies instead of microkernel block dimensions to maintain
  // numeric consistency across CPUs. The reduction dimensions are calculated
  // based on these proxies so the values must not vary based on CPU.
  //
  // Default to f32 values.
  size_t proxy_m = 5;
  size_t proxy_n = 64;
  if (a_elem_size == 8) {
    proxy_m = 5;
    proxy_n = 32;
  } else if (a_elem_size == 2) {
    proxy_m = 32;
    proxy_n = 32;
  } else if (a_elem_size == 1) {
    proxy_m = 6;
    proxy_n = 32;
  }

  size_t working_set_size = is_n_blocked
                                ? (b_size + proxy_m * k1 * k2 * a_elem_size)
                                : (a_size + proxy_n * k1 * k2 * b_elem_size);

  // Fast path: bypass k-tiling if the active working set fits in L2.
  if (working_set_size <= cache_capacity) {
    return finish_schedule();
  }

  // Outermost k-loop: When k is large, we slice it into chunks of `min_block_k`
  // that ensure the working set of A and B remain resident in the L2 cache.
  size_t desired_k_step;
  if (is_n_blocked) {
    size_t footprint_per_k = k2 * (n * b_elem_size + proxy_m * a_elem_size);
    desired_k_step = floor_div(cache_capacity, footprint_per_k);
  } else {
    size_t footprint_per_k = k2 * (m * a_elem_size + proxy_n * b_elem_size);
    desired_k_step = floor_div(cache_capacity, footprint_per_k);
  }
  desired_k_step = std::max<size_t>(64, align_up<size_t>(desired_k_step, 64));
  make_k_loop(desired_k_step);

  // Implement m and n blocking to maximize L3 cache reuse.
  const size_t l3_budget = cpu.cache_sizes[2] / cpu.num_shared_l3_cores;
  if (is_n_blocked) {
    size_t divisor = k1 * k2 * b_elem_size * block_n;
    make_n_loop(l3_budget / divisor);
  } else {
    size_t divisor = k1 * k2 * a_elem_size * block_m;
    make_m_loop(l3_budget / divisor);
  }

  return finish_schedule();
}

}  // namespace ynn
