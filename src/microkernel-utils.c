// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/microkernel-utils.h>

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

size_t xnn_dwconv_multipass_weights_count(
  size_t tile_size,
  size_t channels,
  size_t channel_tile,
  size_t channel_subtile,
  size_t channel_round)
{
  // First and middle pass runs as many channel_tile-sized loops as possible, and can over-read up to channel_round.
  const size_t subtiled_channels = round_up_po2(channels, channel_round);
  // 1 for bias, we always have a first and last pass.
  return (1 + tile_size) *
         // as many channel_tile-sized loops as possible.
         (round_down_po2(subtiled_channels, channel_tile) +
          // handle the remainder in channel_subtile loops.
          round_up_po2(mod_po2(subtiled_channels, channel_tile), channel_subtile));
}
