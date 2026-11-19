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

struct k_steps {
  size_t step_k1;
  size_t step_k2;
  size_t step_k3;
  size_t effective_k1;
  size_t effective_k2;
  size_t effective_k3;
};

// For consistent arithmetic across CPUs, k-loop steps must depend only on `ks`
// and operand element sizes rather than hardware-dependent cache or kernel
// block sizes. Keep each inner k-tile <= `dot_k_step_bytes` per row of A and
// column of B, with `step_k1` aligned down to a multiple of 64 elements (when
// `k2 * k3` fits) so every `k1` step lands on a packed k-tile boundary without
// exceeding `dot_k_step_bytes`.
k_steps compute_k_steps(size_t k1, size_t k2, size_t k3, size_t a_elem_size,
                        size_t b_elem_size) {
  constexpr size_t k_alignment = 64;
  const size_t elem_size = std::max(a_elem_size, b_elem_size);
  assert(elem_size > 0);
  assert(k1 > 0 && k2 > 0 && k3 > 0);
  assert(k_alignment * elem_size <= dot_k_step_bytes);
  const size_t step_k1 =
      std::max<size_t>(
          1, floor_div(dot_k_step_bytes, k_alignment * k2 * k3 * elem_size)) *
      k_alignment;
  const size_t effective_k1 = std::min(k1, step_k1);
  const size_t step_k2 = std::max<size_t>(
      1, floor_div(dot_k_step_bytes, effective_k1 * elem_size));
  const size_t effective_k2 = std::min(k2, step_k2);
  const size_t step_k3 = std::max<size_t>(
      1, floor_div(dot_k_step_bytes, effective_k1 * effective_k2 * elem_size));
  const size_t effective_k3 = std::min(k3, step_k3);
  return {step_k1, step_k2, step_k3, effective_k1, effective_k2, effective_k3};
}

}  // namespace

bool should_pack_a(span<const size_t> cache_budgets, size_t m, size_t n,
                   span<const size_t> ks, size_t block_m, size_t block_n,
                   size_t kernel_block_n, size_t tile_m, size_t tile_k,
                   size_t a_elem_size, size_t b_elem_size, size_t a_stride_m,
                   span<const size_t> a_k_strides) {
  assert(ks.size() <= 3);
  constexpr size_t min_panel_m = 4;
  const size_t panel_m = std::min(m, block_m);
  if (panel_m < min_panel_m || a_stride_m == 0 || kernel_block_n == 0 ||
      n < 2 * kernel_block_n) {
    return false;
  }

  const size_t k1 = !ks.empty() ? ks[0] : 1;
  const size_t k2 = ks.size() >= 2 ? ks[1] : 1;
  const size_t k3 = ks.size() >= 3 ? ks[2] : 1;
  const size_t k1_padded = align_up(k1, tile_k);
  const k_steps steps =
      compute_k_steps(k1_padded, k2, k3, a_elem_size, b_elem_size);

  const size_t row_bytes = steps.effective_k1 * a_elem_size;
  const bool k1_contiguous =
      steps.effective_k1 == 1 ||
      (!a_k_strides.empty() && a_k_strides[0] == a_elem_size);
  const bool k2_contiguous =
      steps.effective_k2 == 1 ||
      (a_k_strides.size() >= 2 && a_k_strides[1] == panel_m * row_bytes);
  const bool k3_contiguous =
      steps.effective_k3 == 1 ||
      (a_k_strides.size() >= 3 &&
       a_k_strides[2] == steps.effective_k2 * panel_m * row_bytes);
  if (k1 == k1_padded && k1_contiguous && a_stride_m == row_bytes &&
      k2_contiguous && k3_contiguous) {
    return false;
  }

  if (k1 == k1_padded && tile_m > 1) {
    return false;
  }

  const size_t panel_k[3] = {steps.effective_k1, steps.effective_k2,
                             steps.effective_k3};
  const size_t packed_bytes = dot_packer::packed_bytes(
      panel_m, {panel_k, ks.size()}, a_elem_size * 8, tile_k, /*tile_p=*/1);
  if (packed_bytes > dot_max_panel_bytes) {
    return false;
  }
  const size_t l1_budget = !cache_budgets.empty() ? cache_budgets[0] : 0;
  if (l1_budget > 0 && packed_bytes > 2 * l1_budget) {
    return false;
  }

  const size_t k = steps.effective_k1 * steps.effective_k2 * steps.effective_k3;
  size_t panel_n = n;
  const size_t a_bytes = m * k * a_elem_size;
  const size_t b_bytes = n * k * b_elem_size;
  const bool fits_in_l1 = l1_budget > 0 && a_bytes + b_bytes <= l1_budget;
  if (m > block_m && n > block_n && !cache_budgets.empty() && !fits_in_l1) {
    const size_t a_block_bytes = k * block_m * a_elem_size;
    const size_t b_block_bytes = k * block_n * b_elem_size;
    for (size_t i = cache_budgets.size(); i-- > 0;) {
      const size_t cache_budget = cache_budgets[i];
      if (cache_budget == 0 || (i > 0 && cache_budget <= l1_budget)) {
        continue;
      }
      if (cache_budget > a_block_bytes) {
        const size_t sub_blocks_n =
            floor_div(cache_budget - a_block_bytes, b_block_bytes);
        if (sub_blocks_n > 1 && sub_blocks_n * block_n < panel_n) {
          panel_n = sub_blocks_n * block_n;
        }
      }
    }
  }

  if (panel_n < 2 * kernel_block_n) {
    return false;
  }

  // Packing A forces the inner loop over `n`; avoid flipping from an `n`-outer
  // schedule when the active B panel would exceed L2 across `m`-blocks.
  if (m > block_m && n * b_elem_size > m * a_elem_size &&
      cache_budgets.size() >= 2 && cache_budgets[1] > l1_budget &&
      panel_n * k * b_elem_size > cache_budgets[1]) {
    return false;
  }

  if (k1 != k1_padded) {
    return true;
  }
  if (!k1_contiguous) {
    return row_bytes >= 16;
  }

  // 4 KiB-multiple row strides map all `panel_m` rows to the same L1D cache set
  // (`l1_ways_4k` ways on 4 KiB-way-stride caches).
  if (l1_budget > 0 && packed_bytes <= l1_budget &&
      panel_n >= 4 * kernel_block_n) {
    constexpr size_t page_bytes = 4096;
    const size_t l1_ways_4k = (2 * l1_budget) / page_bytes;
    if ((a_stride_m % page_bytes) == 0 && l1_ways_4k > 0 &&
        panel_m >= l1_ways_4k) {
      return true;
    }
  }

  if ((l1_budget == 0 || packed_bytes <= l1_budget) &&
      panel_n >= 12 * kernel_block_n) {
    return true;
  }

  return false;
}

span<dot_loop> schedule_dot(span<const size_t> cache_budgets, size_t m,
                            size_t n, span<const size_t> ks, size_t block_m,
                            size_t block_n, size_t a_elem_size,
                            size_t b_elem_size, dot_loop* storage,
                            bool pack_a) {
  dot_loop* begin = storage;
  dot_loop* loop = begin;

  assert(ks.size() <= 3);
  size_t k1 = !ks.empty() ? ks[0] : 1;
  size_t k2 = ks.size() >= 2 ? ks[1] : 1;
  size_t k3 = ks.size() >= 3 ? ks[2] : 1;

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
  auto make_k_loop = [&](int dim, size_t& extent, size_t step_k) {
    if (step_k == 0 || extent <= step_k) return;
    *loop++ = dot_loop{dim, step_k};
    extent = step_k;
  };

  const k_steps steps = compute_k_steps(k1, k2, k3, a_elem_size, b_elem_size);
  make_k_loop(dot_loop::k3, k3, steps.step_k3);
  make_k_loop(dot_loop::k2, k2, steps.step_k2);
  make_k_loop(dot_loop::k1, k1, steps.step_k1);
  const size_t k = k1 * k2 * k3;

  // When packing each `block_m` panel of A, the inner loop must iterate over
  // `n` to reuse that packed panel, so cache blocking can only tile `n` outside
  // `m`.
  const bool tile_n_for_cache = pack_a || n * b_elem_size <= m * a_elem_size;
  const size_t a_bytes = m * k * a_elem_size;
  const size_t b_bytes = n * k * b_elem_size;
  const bool fits_in_l1 =
      !cache_budgets.empty() && a_bytes + b_bytes <= cache_budgets[0];
  if (m > block_m && n > block_n && !cache_budgets.empty() && !fits_in_l1) {
    const size_t a_block_bytes = k * block_m * a_elem_size;
    const size_t b_block_bytes = k * block_n * b_elem_size;
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
        // Tile n so the active B working set fits in `cache_budget` alongside
        // one block of A across the m loop.
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
  if (pack_a) {
    make_m_loop(block_m);
    *loop++ = dot_loop{dot_loop::pack_a, 0};
    make_n_loop(block_n);
  } else if (fits_in_l1 ? (m * block_n < n * block_m) : tile_n_for_cache) {
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

  return {begin, loop};
}

}  // namespace ynn
