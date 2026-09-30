// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microkernel-utils.h"

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/operator-utils.h"

static bool gemm_fits_in_cache(size_t mr, size_t nc, size_t m_stride,
                               size_t n_stride, size_t cn_stride,
                               size_t cache_size, size_t cache_line_size) {
  if (cache_line_size == 0 || cache_size == 0) {
    return false;
  }

  size_t mr_bytes, nc_bytes, output_bytes;
  if (!xnn_safe_mul(mr, m_stride, &mr_bytes) ||
      !xnn_safe_mul(nc, n_stride, &nc_bytes) ||
      !xnn_safe_mul(nc, cn_stride, &output_bytes)) {
    return false;
  }

  const size_t lines_mr = divide_round_up(mr_bytes, cache_line_size);
  const size_t lines_nc = divide_round_up(nc_bytes, cache_line_size);
  const size_t lines_output_per_m =
      divide_round_up(output_bytes, cache_line_size);
  size_t lines_output;
  if (!xnn_safe_mul(mr, lines_output_per_m, &lines_output)) {
    return false;
  }

  size_t lines_per_row;
  if (!xnn_safe_add(lines_mr, lines_nc, &lines_per_row) ||
      !xnn_safe_add(lines_per_row, lines_output, &lines_per_row)) {
    return false;
  }

  size_t required_cache_size;
  if (!xnn_safe_mul(lines_per_row, cache_line_size, &required_cache_size)) {
    return false;
  }

  if (cache_size < required_cache_size) {
    return false;
  }

  return true;
}

bool xnn_should_inline_lhs_packing(const struct xnn_gemm_config *gemm_config,
                                   size_t m_packed_stride, size_t n_stride,
                                   size_t cn_stride, size_t mc, size_t nc) {
  if (gemm_config == NULL) {
    return false;
  }

  const struct xnn_hardware_config *hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return false;
  }

  // Select which cache we want the tiles to fit in.
  const size_t cache_bytes = hardware_config->l2_data_cache_bytes;
  const size_t cache_line_size = hardware_config->l2_data_cache_line_size;

  // If we don't have any information on the cache size, then all bets are off.
  if (!cache_bytes || !cache_line_size) {
    return false;
  }

  const size_t mr = min(gemm_config->mr, mc);

  // We only want to inline the LHS packing if it's possible to
  // compute an entire row of the GEMM without flooding the cache.
  const bool gemm_row_fits_in_cache =
      gemm_fits_in_cache(mr, nc, m_packed_stride, n_stride, cn_stride,
                         cache_bytes, cache_line_size);

  xnn_log_debug(
      "mr=%zu, nr=%hhu, m_packed_stride=%zu, n_stride=%zu, mc=%zu, nc=%zu, "
      "gemm_row_fits_in_cache=%s.",
      mr, gemm_config->nr, m_packed_stride, n_stride, mc, nc,
      gemm_row_fits_in_cache ? "true" : "false");

  return gemm_row_fits_in_cache;
}

size_t xnn_gemm_best_tile_size(size_t num_groups, size_t m, size_t n,
                               size_t m_stride, size_t n_stride,
                               size_t cn_stride, size_t mr, size_t nr,
                               size_t num_threads) {
  if (m == 0 || n == 0 || mr == 0 || nr == 0) {
    return 0;
  }

  const struct xnn_hardware_config *hardware_config =
      xnn_init_hardware_config();
  if (hardware_config == NULL) {
    return min(nr, n);
  }

  // Adjust `mr` and `nr` if they are larger than `m` and `n`, respectively.
  mr = min(mr, m);
  nr = min(nr, n);

  // We only care about the number of tiles if we have more than one thread.
  const size_t min_num_tiles =
      num_threads > 1 ? XNN_GEMM_MIN_TILES_PER_THREAD * num_threads : 1;

  // Start with a `mr`x`nr` tile.
  size_t nc = nr;
  const size_t num_tiles_m = divide_round_up(m, mr);

  // Select which cache we want the tiles to fit in. Start with L1, and if the
  // smallest possible tile won't fit, try L2. If the smallest tile still won't
  // fit, then don't try to fit to the cache size.
  size_t cache_size = hardware_config->l1_data_cache_bytes;
  size_t cache_line_size = hardware_config->l1_data_cache_line_size;
  if (XNN_ARCH_X86 || XNN_ARCH_X86_64 ||
      (cache_size && !gemm_fits_in_cache(mr, nr, m_stride, n_stride, cn_stride,
                                         cache_size, cache_line_size))) {
    cache_size = hardware_config->l2_data_cache_bytes;
    cache_line_size = hardware_config->l2_data_cache_line_size;
    if (cache_size && !gemm_fits_in_cache(mr, nr, m_stride, n_stride, cn_stride,
                                          cache_size, cache_line_size)) {
      // Don't check for cache fit.
      cache_size = 0;
    }
  }

  int max_j = divide_round_up(n, nr) + 1;

  // Find maximum nc such that a tile still fits into cache.
  if (mr < m && cache_size) {
    int l = 1, r = max_j;

    size_t r_nr;
    const bool r_valid = xnn_safe_mul((size_t) r, nr, &r_nr);
    if (!r_valid || !gemm_fits_in_cache(mr, r_nr, m_stride, n_stride, cn_stride,
                                        cache_size, cache_line_size)) {
      while (r - l > 1) {
        int mid = (l + r) / 2;
        size_t mid_nr;
        const bool mid_valid = xnn_safe_mul((size_t) mid, nr, &mid_nr);
        if (!mid_valid || !gemm_fits_in_cache(mr, mid_nr, m_stride, n_stride,
                                              cn_stride, cache_size,
                                              cache_line_size)) {
          r = mid;
        } else {
          l = mid;
        }
      }
      max_j = r;
    }
  }

  size_t j_estimate = 1;
  // Find j so it satisfies num_tiles >= min_num_tiles
  {
    int l = 1, r = max_j;
    while (r - l > 1) {
      int mid = (l + r) / 2;
      size_t mid_nr;
      if (!xnn_safe_mul((size_t) mid, nr, &mid_nr) || mid_nr == 0) {
        r = mid;
        continue;
      }
      const size_t num_tiles_n = divide_round_up(n, mid_nr);
      size_t num_tiles;
      if (!xnn_safe_mul(num_tiles_n, num_tiles_m, &num_tiles) ||
          !xnn_safe_mul(num_tiles, num_groups, &num_tiles)) {
        // More than enough tiles.
        l = mid;
        continue;
      }
      if (num_tiles < min_num_tiles) {
        r = mid;
      } else {
        l = mid;
      }
    }

    // Find smallest j_estimate such that the number of tiles is the same as j.
    size_t l_nr;
    if (xnn_safe_mul((size_t) l, nr, &l_nr) && l_nr != 0) {
      const size_t num_tiles_n_estimate = divide_round_up(n, l_nr);
      size_t est_div;
      if (xnn_safe_mul(num_tiles_n_estimate, nr, &est_div) && est_div != 0) {
        j_estimate = divide_round_up(n, est_div);
      }
    }
  }

  size_t est_nc;
  if (xnn_safe_mul(j_estimate, nr, &est_nc) && est_nc != 0) {
    nc = est_nc;
  }

  // Restrict the resulting `nc` to `n`.
  nc = min(nc, n);
  if (nc == 0) {
    nc = min(nr, n);
  }

  xnn_log_debug(
      "Tile size for GEMM with num_groups=%zi, m=%zu, n=%zu and mr=%zu, "
      "nr=%zu set to [%zu, %zu] (%zu tiles)",
      num_groups, m, n, mr, nr, mr, nc,
      nc != 0 ? num_tiles_m * divide_round_up(n, nc) * num_groups : 0);
  return nc;
}

// Checks whether to use the `nr2` config or not.
bool xnn_use_nr2(size_t nr, size_t nr2, size_t output_channels) {
  if (nr == 0 || nr2 == 0) {
    return false;
  }
  const size_t nr_rem = output_channels % nr;
  const size_t nr2_rem = output_channels % nr2;
  const size_t nr_overcompute = nr_rem == 0 ? 0 : nr - nr_rem;
  const size_t nr2_overcompute = nr2_rem == 0 ? 0 : nr2 - nr2_rem;

  size_t nr_overcompute_percent;
  const bool waste_significant =
      xnn_safe_mul(nr_overcompute, 100, &nr_overcompute_percent)
          ? nr_overcompute_percent > output_channels
          : true;

  // Switch to alternative microkernel when:
  // 1. Alternative microkernel better supports fewer output channels, or
  // 2. Alternative microkernel has less overcompute and default wastes >1% of
  // output channels
  if (nr > output_channels ||
      (nr2_overcompute < nr_overcompute && waste_significant)) {
    // Default microkernel is suboptimal, use a microkernel that better
    // supports fewer output channels.
    return true;
  }
  return false;
}
