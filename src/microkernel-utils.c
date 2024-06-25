// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/math.h"
#include "xnnpack/microkernel-utils.h"

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
