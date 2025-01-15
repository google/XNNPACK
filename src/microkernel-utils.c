// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/microkernel-utils.h"

#include <assert.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"

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
  xnn_log_info(
      "Tile size for GEMM with num_groups=%zi, m=%zu, n=%zu and mr=%zu, nr=%zu "
      "set to [%zu, %zu] (%zu tiles)",
      num_groups, m, n, mr, nr, mr, nc, best_num_tiles);
  return nc;
}

static size_t dwconv_num_middle_pass(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile)
{
  return divide_round_up(doz(kernel_size, first_pass_tile + last_pass_tile), middle_pass_tile);
}

size_t xnn_dwconv_multipass_tile_size(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile)
{
  assert(kernel_size > first_pass_tile);
  // We always have a first and last pass. We run as many middle pass as possible.
  // E.g. kernel_size == 9, first_pass_tile = 2, middle_pass_tile = 3, last_pass_tile == 3.
  // 1 first pass (8 left), 2 middle pass (2 left), last pass (with remainder 1).
  return (first_pass_tile + last_pass_tile +
          round_up(doz(kernel_size, first_pass_tile + last_pass_tile), middle_pass_tile));
}

size_t xnn_dwconv_multipass_weights_size(
  size_t tile_size,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  size_t bias_element_size,
  size_t log2_filter_element_size,
  size_t extra_weights_byte)
{
  // First and middle pass runs as many channel_tile-sized loops as possible, and can over-read up to channel_round.
  const size_t subtiled_channels = round_up_po2(channels, channel_round);
  // Always have a first and last pass.
  size_t c_stride = (round_down_po2(subtiled_channels, channel_tile) +
                     // handle the remainder in channel_subtile loops.
                     round_up_po2(mod_po2(subtiled_channels, channel_tile), channel_subtile));
  return ((tile_size << log2_filter_element_size) + bias_element_size + extra_weights_byte) * c_stride;
}

size_t xnn_dwconv_multipass_bytes_read(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round,
  size_t log2_input_size,
  size_t log2_filter_size,
  size_t bias_element_size,
  size_t log2_accumulator_size)
{
  const size_t num_middle_pass = dwconv_num_middle_pass(kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile);
  const size_t tile_size = first_pass_tile + num_middle_pass * middle_pass_tile + last_pass_tile;
  const size_t rounded_channels = round_up_po2(channels, channel_round);
  const size_t input_elements_read = tile_size * rounded_channels;
  const size_t weight_elements_read = tile_size * rounded_channels;
  const size_t bias_elements_read = rounded_channels;
  // Middle pass reads num_middle_pass * rounded_channels buffers. Last pass reads tiled_channel_buffers.
  // This is equivalent to (num_middle_pass + 1) * rounded_channels.
  const size_t buffer_elements_read = (num_middle_pass + 1) * rounded_channels;
  return (input_elements_read << log2_input_size) + (weight_elements_read << log2_filter_size) +
    (bias_elements_read * bias_element_size) + (buffer_elements_read << log2_accumulator_size);
}

size_t xnn_dwconv_multipass_bytes_written(
  size_t kernel_size,
  size_t first_pass_tile,
  size_t middle_pass_tile,
  size_t last_pass_tile,
  size_t channels,
  size_t channel_round,
  size_t log2_accumulator_size,
  size_t log2_output_size)
{
  // First pass writes rounded_channels elements to buffer, middle pass writes num_middle_pass * rounded_channels
  // elements to buffer. Last pass writes channels elements to output.
  // This is equivalent to (1 + num_middle_pass) * rounded_channels + channels elements.
  const size_t num_middle_pass = dwconv_num_middle_pass(kernel_size, first_pass_tile, middle_pass_tile, last_pass_tile);
  const size_t rounded_channels = round_up_po2(channels, channel_round);
  const size_t buffer_elements_written = (1 + num_middle_pass) * rounded_channels;
  const size_t output_elements_written = channels;
  return (buffer_elements_written << log2_accumulator_size) + (output_elements_written << log2_output_size);
}
