// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/microkernel-utils.h"

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"

static bool fits_in_cache(size_t mr, size_t nc, size_t m_stride,
                          size_t n_stride, size_t cm_stride, size_t cn_stride,
                          size_t cache_size, size_t cache_line_size) {
  // Check if the bytes fit.
  const size_t lines_mr = divide_round_up(mr * m_stride, cache_line_size);
  const size_t lines_nc = divide_round_up(nc * n_stride, cache_line_size);
  const size_t lines_output =
      mr * divide_round_up(nc * cn_stride, cache_line_size);
  const size_t lines_per_row = lines_mr + lines_nc + lines_output;
  if (cache_size < lines_per_row * cache_line_size) {
    return false;
  }

  // Otherwiese, we're good.
  return true;
}

size_t xnn_gemm_best_tile_size(size_t num_groups, size_t m, size_t n,
                               size_t m_stride, size_t n_stride,
                               size_t cm_stride, size_t cn_stride, size_t mr,
                               size_t nr, size_t num_threads) {
  // Adjust `mr` and `nr` if they are larger than `m` and `n`, respectively.
  mr = min(mr, m);
  nr = min(nr, n);

  // We only care about the number of tiles if we have more than one thread.
  const size_t min_num_tiles =
      num_threads > 1 ? XNN_GEMM_MIN_TILES_PER_THREAD * num_threads : 1;

  // Start with a `mr`x`nr` tile.
  size_t nc = nr;
  const size_t num_tiles_m = divide_round_up(m, mr);
  size_t best_num_tiles = num_tiles_m * divide_round_up(n, nc) * num_groups;

  // Select which cache we want the tiles to fit in. Start with L1, and if the
  // smallest possible tile won't fit, try L2. If the smallest tile still won't
  // fit, then don't try to fit to the cache size.
  const struct xnn_hardware_config *hardware_config =
      xnn_init_hardware_config();
  size_t cache_size = hardware_config->l1_data_cache_bytes;
  size_t cache_line_size = hardware_config->l1_data_cache_line_size;
  if (XNN_ARCH_X86 || XNN_ARCH_X86_64 ||
      (cache_size && !fits_in_cache(mr, nr, m_stride, n_stride, cm_stride,
                                    cn_stride, cache_size, cache_line_size))) {
    cache_size = hardware_config->l2_data_cache_bytes;
    cache_line_size = hardware_config->l2_data_cache_line_size;
    if (cache_size && !fits_in_cache(mr, nr, m_stride, n_stride, cm_stride,
                                     cn_stride, cache_size, cache_line_size)) {
      // Don't check for cache fit.
      cache_size = 0;
    }
  }

  // Loop over all multiples of `nr`.
  for (int j = 1; (j - 1) * nr < n; j++) {
    // Skip this `j` if it results in the same number of tiles as `j - 1`.
    if (1 < j &&
        divide_round_up(n, j * nr) == divide_round_up(n, (j - 1) * nr)) {
      continue;
    }

    // If we have at most `mr` rows per group, then there will be no cache
    // re-use across tile rows and we don't care about whether the data fits
    // in cache or not.
    // If, however, we have more than one tile row, then we want the data used
    // to compute a tile of size `mr`x`j*nr` to fit in the cache.
    if (mr < m && cache_size &&
        !fits_in_cache(mr, j * nr, m_stride, n_stride, cm_stride, cn_stride,
                       cache_size, cache_line_size)) {
      break;
    }

    // Make sure this pair of `i` and `j` generates enough tiles.
    const size_t num_tiles_n = divide_round_up(n, j * nr);
    const size_t num_tiles = num_tiles_n * num_tiles_m * num_groups;
    if (num_tiles < min_num_tiles) {
      break;
    }

    // New best tile size? We define the "best" tiling as the smallest total
    // number of tiles, and for tilings with the same number of tiles, we take
    // the tiling with the largest `nc`.
    if (num_tiles < best_num_tiles ||
        (num_tiles == best_num_tiles && nc < j * nr)) {
      nc = j * nr;
      best_num_tiles = num_tiles;
    }
  }

  // Restrict the resulting `nc` to `n`.
  nc = min(nc, n);
  xnn_log_debug(
      "Tile size for GEMM with num_groups=%zi, m=%zu, n=%zu and mr=%zu, nr=%zu "
      "set to [%zu, %zu] (%zu tiles)",
      num_groups, m, n, mr, nr, mr, nc, best_num_tiles);
  return nc;
}
