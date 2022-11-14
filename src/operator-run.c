// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/operator.h>
#include <xnnpack/log.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/microkernel-type.h>
#include <xnnpack/params.h>
#include <xnnpack/compute.h>

void xnn_compute_batch_to_space_v1d(
    const struct batch_to_space_context* context,
    size_t offset,
    size_t size)
{
  // This is never tiled, we ignore those arguments.
  (void) size;
  (void) offset;

  // We are running through the output buffer and computing the corresponding
  // input offsets. This is easier and avoids branching out of the cropped
  // areas.
  const size_t input_batch_stride = context->identity.input_batch_stride;
  const size_t input_row_stride = context->identity.input_row_stride;
  const size_t output_batch = context->identity.output_batch;
  const size_t output_height = context->output_height;
  const size_t output_row_stride = context->identity.output_row_stride;

  const xnn_vunary_ukernel_fn ukernel = context->univector_contiguous.ukernel;
  const void* const params = &context->univector_contiguous.params;

  uintptr_t x_base = (uintptr_t) context->univector_contiguous.x + context->identity.initial_input_offset;
  uintptr_t y = (uintptr_t) context->univector_contiguous.y;

  for (size_t b = 0; b < output_batch; ++b) {
    uintptr_t x = x_base;
    for (size_t i = 0; i < output_height; ++i) {
      ukernel(output_row_stride, (void*)x, (void*)y, params);
      y += output_row_stride;
      x += input_row_stride;
    }
    x_base += input_batch_stride;
  }
}

struct batch_to_space_subtile {
  uintptr_t x;
  size_t height;
  size_t width;
  uintptr_t y_offset;
};

void transpose_subtile_v2d(
    const struct batch_to_space_context* const context,
    uintptr_t x,
    uintptr_t y,
    size_t height,
    size_t width)
{

  context->transpose.variable_size_ukernel(
      (void*)x,
      (void*)y,
      context->tile_2d.ld_input,
      context->tile_2d.ld_output,
      context->tile_2d.input_element_stride,
      context->tile_2d.output_element_stride,
      context->tile_2d.element_size,
      height,
      width);
}

void transpose_subtile_c2d(
    const struct batch_to_space_context* const context,
    uintptr_t x,
    uintptr_t y,
    size_t height,
    size_t width)
{
  context->transpose.const_size_ukernel(
      (void*)x,
      (void*)y,
      context->tile_2d.ld_input,
      context->tile_2d.ld_output,
      height,
      width,
      &context->transpose.params);
}

void transpose_subtile_v2d_by_row(
    const struct batch_to_space_context* const context,
    size_t row,
    uintptr_t x,
    uintptr_t y,
    size_t height,
    size_t width)
{
  const struct batch_to_space_context_tile_2d* const tile_2d = &context->tile_2d;
  for (size_t a = 0; a < height; ++a) {
    const size_t b_start = row > tile_2d->top_crop_row ? row : tile_2d->top_crop_row;
    const size_t b_end = row + width < tile_2d->bottom_crop_row ? row + width : tile_2d->bottom_crop_row;
    if (b_end > b_start) {
      transpose_subtile_v2d(
          context,
          x + (b_start - row) * tile_2d->ld_input,
          y + (b_start - row) * tile_2d->output_element_stride,
          1,
          b_end - b_start);
    }
    // We never overshoot by more than one output batch element because
    // output_height = block_height * input_height.
    row += context->block_height;
    if (row >= context->output_height_nocrop) {
      row -= context->output_height_nocrop;
      y += tile_2d->output_batch_stride_compensation;
    }
    x += tile_2d->input_element_stride;
    y += tile_2d->ld_output;
  }
}

// In this case, the initial 4D transposition is reduced to a 2D transposition
// where each element will be a full row in the output tensor.
//
// Let's note:
//   - o: an input tensor element (an output row).
//   - x: a tile element (an output row).
//   - |: the delimiter between batch items.
//
// The final output tensor will be:
//
// o1 | o5 | o9
// o2 | o6 | o10
// o3 | o7 | o11  ...
// o4 | o8 | o12
//
// The kernel input tensor will be (x is the current tile):
//
// x1  x2  x3  o4  o5
// x6  x7  x8  o9  o10
// o11 o12 o13 o14 o15
// ...
void xnn_compute_batch_to_space_2dv(
    const struct batch_to_space_context* const context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  const struct batch_to_space_context_tile_2d* const tile_2d = &context->tile_2d;
  const size_t global_first_row = i * context->block_height + j;
  const size_t batch = global_first_row / context->output_height_nocrop;
  const size_t start_row = global_first_row % context->output_height_nocrop;
  // The end row is computed from the start row to the last element held by this
  // subtile. It may overshoot the output tensor height, which means the tile
  // covers multiple batch items.
  //
  // Note: subtile rows may not be contiguous in the final output tensor, which
  // is why we don't do `tile_i * tile_j`.
  const size_t end_row = start_row + (tile_i-1) * context->block_height + tile_j;

  uintptr_t x = (uintptr_t)context->transpose.x
      + i * context->transpose.input_stride[0]
      + j * context->transpose.input_stride[1]
      + tile_2d->base_input_offset;
  uintptr_t y = (uintptr_t)context->transpose.y
      + i * context->transpose.output_stride[0]
      + j * context->transpose.output_stride[1]
      // Previous top and bottom crops.
      - batch * tile_2d->vertical_crop_bytes - tile_2d->crop_top_bytes
      // Previous left and right crops.
      - (batch * context->output_height + start_row - tile_2d->top_crop_row)
        * tile_2d->horizontal_crop_bytes;

  // The full tile can be transposed in one call when the output stride is constant.
  if (tile_2d->vertical_crop == 0 || (start_row > tile_2d->top_crop_row
                                     && end_row <= tile_2d->bottom_crop_row))
  {
    transpose_subtile_v2d(context, x, y, tile_i, tile_j);
  } else {
    transpose_subtile_v2d_by_row(context, start_row, x, y, tile_i, tile_j);
  }
}

void xnn_compute_batch_to_space_2dh(
    void (*kernel)(const struct batch_to_space_context*, size_t, size_t, size_t, size_t),
    const struct batch_to_space_context* const context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  const size_t input_stride_i = context->transpose.input_stride[0];
  const size_t input_stride_j = context->transpose.input_stride[1];
  const size_t output_stride_i = context->transpose.output_stride[0];
  const size_t output_stride_j = context->transpose.output_stride[1];
  const size_t output_height_nocrop = context->output_height_nocrop;
  const size_t output_width_nocrop = context->output_width_nocrop;
  const size_t left_crop_col = context->tile_2d.left_crop_col;
  const size_t right_crop_col = context->tile_2d.right_crop_col;
  const size_t top_crop_row = context->tile_2d.top_crop_row;
  const size_t bottom_crop_row = context->tile_2d.bottom_crop_row;
  const size_t crop_left_bytes = context->tile_2d.crop_left_bytes;
  const size_t vertical_crop_bytes = context->tile_2d.vertical_crop_bytes;
  const size_t horizontal_crop_bytes = context->tile_2d.horizontal_crop_bytes;
  const size_t crop_top_bytes = context->tile_2d.crop_top_bytes;
  const size_t crop_left = context->crop_left;
  const size_t block_width = context->block_width;

  size_t row_idx = i * block_width + j;
  size_t output_row_global = row_idx / output_width_nocrop;
  size_t batch = output_row_global / output_height_nocrop ;
  size_t output_row = output_row_global % output_height_nocrop;
  size_t output_col = row_idx % output_width_nocrop;

  size_t last_element_idx = (i + tile_i - 1) * block_width + j + tile_j - 1;
  size_t last_output_row_global = last_element_idx / output_width_nocrop;
  size_t last_output_col = last_element_idx % output_width_nocrop;

  size_t horizontal_crop_offset = output_row_global * horizontal_crop_bytes + crop_left_bytes;
  size_t vertical_crop_offset = batch * vertical_crop_bytes + crop_top_bytes;

  uintptr_t x = (uintptr_t)context->transpose.x + i * input_stride_i + j * input_stride_j;
  uintptr_t y = (uintptr_t)context->transpose.y + i * output_stride_i + j * output_stride_j;

  // Transpose full tile at once.
  if (last_output_row_global == output_row_global
     && output_col >= left_crop_col
     && last_output_col < right_crop_col
     && output_row >= top_crop_row
     && output_row < bottom_crop_row)
  {
    uintptr_t yy = y - horizontal_crop_offset - vertical_crop_offset;
    kernel(context, x, yy, tile_i, tile_j);
    return;
  }

  // Transpose tile one row at a time.
  for (size_t tile_row = 0; tile_row < tile_i; ++tile_row) {
    uintptr_t xx = x;
    uintptr_t yy = y;
    size_t width = tile_j;
    // Check that the output row is not within the vertical crop spec.
    if (output_row < top_crop_row || output_row >= bottom_crop_row) {
      goto next_row;
    }
    // Left crop.
    if (output_col < left_crop_col) {
      const size_t shift = crop_left - output_col;
      if (width > shift) {
        width -= shift;
        xx += input_stride_j * shift;
        yy += output_stride_j * shift;
      } else {
        goto next_row;
      }
    }
    // Right crop.
    const size_t row_last_output_col = output_col + tile_j - 1;
    if (row_last_output_col >= right_crop_col) {
      const size_t shift = row_last_output_col - right_crop_col + 1;
      if (width > shift) {
        width -= shift;
      } else {
        goto next_row;
      }
    }
    // Transpose row.
    yy -= horizontal_crop_offset + vertical_crop_offset;
    kernel(context, xx, yy, 1, width);
next_row:;
    x += input_stride_i;
    y += output_stride_i;
    if ((output_col += block_width) >= output_width_nocrop) {
      // Update output line.
      output_col -= output_width_nocrop;
      ++output_row_global;
      ++output_row;
      horizontal_crop_offset += horizontal_crop_bytes;
      if (output_row >= output_height_nocrop) {
        // Update output image.
        output_row = 0;
        ++batch;
        vertical_crop_offset += vertical_crop_bytes;
      }
    }
  }
}

void xnn_compute_batch_to_space_c2dh(
    const struct batch_to_space_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  xnn_compute_batch_to_space_2dh(transpose_subtile_c2d, context, i, j, tile_i, tile_j);
}

void xnn_compute_batch_to_space_v2dh(
    const struct batch_to_space_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  xnn_compute_batch_to_space_2dh(transpose_subtile_v2d, context, i, j, tile_i, tile_j);
}


struct batch_to_space_compute_subtiles_4d {
  bool skip;
  uintptr_t y;
  struct batch_to_space_subtile same_line_subtile;
  struct batch_to_space_subtile left_crop_subtile;
  struct batch_to_space_subtile full_subtile;
  struct batch_to_space_subtile right_crop_subtile;
};

struct batch_to_space_compute_subtiles_4d batch_to_space_setup_subtiles_4d(
    const struct batch_to_space_context * restrict context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  const size_t output_width_nocrop = context->output_width_nocrop;
  const size_t output_height_nocrop = context->output_height_nocrop;
  const size_t k_output_stride = context->tile_4d.k_output_stride;
  const size_t block_height = context->block_height;
  const size_t top_crop_row = context->tile_4d.top_crop_row;
  const size_t bottom_crop_row = context->tile_4d.bottom_crop_row;
  const size_t crop_left = context->crop_left;
  const size_t crop_right = context->crop_right;
  const size_t crop_top_bytes = context->tile_4d.crop_top_bytes;
  const size_t crop_left_bytes = context->tile_4d.crop_left_bytes;
  const size_t vertical_crop_bytes = context->tile_4d.vertical_crop_bytes;
  const size_t horizontal_crop_bytes = context->tile_4d.horizontal_crop_bytes;
  const size_t element_size = context->element_size;
  const size_t ld_output = context->tile_4d.ld_output;
  const struct transpose_context* const transpose = &context->transpose;
  struct batch_to_space_compute_subtiles_4d precompute = {0};

  // The line in the output tensor assuming all images are stacked along the
  // height.
  const size_t line = (i * block_height + j);
  const size_t batch = line / output_height_nocrop;
  const size_t output_row = line % output_height_nocrop;

  // The output line is within the top/bottom crop specification, skip this
  // tile.
  if (output_row < top_crop_row || output_row >= bottom_crop_row) {
    precompute.skip = true;
    return precompute;
  }

  const size_t tile_size = (tile_k-1) * k_output_stride + tile_l;
  const size_t elements_left_to_tile = k * k_output_stride + l;
  const size_t remaining_left_crop = elements_left_to_tile <= crop_left ?
      crop_left - elements_left_to_tile : 0;
  if (remaining_left_crop >= tile_size) {
    precompute.skip = true;
    return precompute;
  }

  // Position of first element that isn't cropped in the tile.
  const size_t extended_tile_left_crop_k = remaining_left_crop / k_output_stride;
  const size_t extended_tile_left_crop_l = remaining_left_crop % k_output_stride;
  const bool left_crop_oob = (extended_tile_left_crop_l >= tile_l);
  const size_t left_crop_k = extended_tile_left_crop_k + left_crop_oob;
  const size_t left_crop_l = left_crop_oob ? 0 : extended_tile_left_crop_l;

  const size_t elements_right_to_tile = output_width_nocrop - elements_left_to_tile - tile_size;
  const size_t remaining_right_crop = elements_right_to_tile <= crop_right
        ? crop_right - elements_right_to_tile : 0;

  if (remaining_right_crop >= tile_size) {
    precompute.skip = true;
    return precompute;
  }

  const size_t remaining_right_crop_idx = tile_size - remaining_right_crop;
  const size_t extended_tile_right_crop_k = remaining_right_crop_idx / k_output_stride;
  const size_t extended_tile_right_crop_l = remaining_right_crop_idx % k_output_stride;

  // Position of first element that is cropped by the righ crop specification.
  const bool right_crop_oob = (extended_tile_right_crop_l >= tile_l);
  const size_t right_crop_k = extended_tile_right_crop_k + right_crop_oob;
  const size_t right_crop_l = right_crop_oob ? 0 : extended_tile_right_crop_l;

  if (left_crop_k > right_crop_k || (left_crop_k == right_crop_k && left_crop_l >= right_crop_l)) {
    precompute.skip = true;
    return precompute;
  }

  const uintptr_t x = ((uintptr_t) transpose->x
                       + i * transpose->input_stride[0]
                       + j * transpose->input_stride[1]
                       + k * transpose->input_stride[2]
                       + l * transpose->input_stride[3]);
  const uintptr_t y = ((uintptr_t) transpose->y
                       + i * transpose->output_stride[0]
                       + j * transpose->output_stride[1]
                       + k * transpose->output_stride[2]
                       + l * transpose->output_stride[3]
                       // Compensate for in-tile cropping. i.e. elements must be
                       // at their output position as if there weren't any
                       // cropping.
                       + left_crop_k * transpose->output_stride[2]
                       + left_crop_l * transpose->output_stride[3]
                       // Take cropping into account.
                       - crop_top_bytes
                       - crop_left_bytes
                       - batch * vertical_crop_bytes
                       - line * horizontal_crop_bytes
                       );
  precompute.y = y;

  precompute.same_line_subtile = (struct batch_to_space_subtile) {
    .x = x + left_crop_k * transpose->input_stride[2]
           + left_crop_l * transpose->input_stride[3],
    .height = left_crop_k == right_crop_k,
    .width = right_crop_l - left_crop_l,
    .y_offset = 1,
  };


  if (precompute.same_line_subtile.height) {
    return precompute;
  }

  precompute.left_crop_subtile = (struct batch_to_space_subtile) {
    .x = x + left_crop_k * transpose->input_stride[2]
           + left_crop_l * transpose->input_stride[3],
    .height = left_crop_l != 0,
    .width = tile_l - left_crop_l,
    .y_offset = ld_output - left_crop_l * element_size,
  };

  const size_t full_subtile_height = right_crop_k - left_crop_k - precompute.left_crop_subtile.height;
  precompute.full_subtile = (struct batch_to_space_subtile) {
    .x = x + (left_crop_k + precompute.left_crop_subtile.height) * transpose->input_stride[2],
    .height = full_subtile_height,
    .width = tile_l,
    .y_offset = full_subtile_height * ld_output,
  };

  precompute.right_crop_subtile = (struct batch_to_space_subtile) {
    .x = x + right_crop_k * transpose->input_stride[2],
    .height = right_crop_l != tile_l && right_crop_k < tile_k,
    .width = right_crop_l,
    .y_offset = 0,
  };

  return precompute;
}

size_t transpose_subtile_const_size_ukernel(
    const struct batch_to_space_context* const context,
    const struct batch_to_space_subtile* const subtile,
    uintptr_t y,
    size_t i,
    size_t j)
{
  if (subtile->height == 0) {
    return 0;
  }
  context->transpose.const_size_ukernel(
      (void*)subtile->x,
      (void*)y,
      context->tile_4d.ld_input,
      context->tile_4d.ld_output,
      subtile->height,
      subtile->width,
      &context->transpose.params);
  return subtile->y_offset;
}

size_t transpose_subtile_variable_size_ukernel(
    const struct batch_to_space_context* const context,
    const struct batch_to_space_subtile* const subtile,
    uintptr_t y,
    size_t i,
    size_t j)
{
  if (subtile->height == 0) {
    return 0;
  }
  context->transpose.variable_size_ukernel(
      (void*)subtile->x,
      (void*)y,
      context->tile_4d.ld_input,
      context->tile_4d.ld_output,
      context->element_size,
      context->element_size,
      context->element_size,
      subtile->height,
      subtile->width);
  return subtile->y_offset;
}

void xnn_compute_batch_to_space_4d(
    size_t (transpose_subtile) (
        const struct batch_to_space_context* const,
        const struct batch_to_space_subtile* const,
        uintptr_t, size_t, size_t),
    const struct batch_to_space_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  const struct batch_to_space_compute_subtiles_4d precompute = batch_to_space_setup_subtiles_4d(
      context, i, j, k, l, tile_k, tile_l);
  if (precompute.skip) {
    return;
  }

  uintptr_t y = precompute.y;
  if (!transpose_subtile(context, &precompute.same_line_subtile, y, i, j)) {
    y += transpose_subtile(context, &precompute.left_crop_subtile, y, i, j);
    y += transpose_subtile(context, &precompute.full_subtile, y, i, j);
    transpose_subtile(context, &precompute.right_crop_subtile, y, i, j);
  }
}

void xnn_compute_batch_to_space_c4d(
    const struct batch_to_space_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  xnn_compute_batch_to_space_4d(transpose_subtile_const_size_ukernel,
                                context, i, j, k, l, tile_k, tile_l);
}

void xnn_compute_batch_to_space_v4d(
    const struct batch_to_space_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  xnn_compute_batch_to_space_4d(transpose_subtile_variable_size_ukernel,
                                context, i, j, k, l, tile_k, tile_l);
}

void xnn_compute_transposec_2d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  const size_t ld_input = context->input_stride[1];
  const size_t ld_output = context->output_stride[0];
  context->const_size_ukernel(
      (const void*) ((uintptr_t) context->x + i * context->input_stride[0] + j * context->input_stride[1]),
      (void*) ((uintptr_t) context->y + j * context->output_stride[1] + i * context->output_stride[0]),
      ld_input,
      ld_output,
      tile_i,
      tile_j,
      &context->params);
}

void xnn_compute_transposec_3d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t tile_j,
    size_t tile_k)
{
  const size_t ld_input = context->input_stride[2];
  const size_t ld_output = context->output_stride[1];
  const void* x = (const void*) ((uintptr_t) context->x +
                                 i * context->input_stride[0] + j * context->input_stride[1] + k * context->input_stride[2]);
  void* y = (void*) ((uintptr_t) context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_j,
      tile_k,
      &context->params);
}

void xnn_compute_transposec_4d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  const size_t ld_input = context->input_stride[3];
  const size_t ld_output = context->output_stride[2];
  const void* x = (const void*) ((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3]);
  void* y = (void*) ((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2] + l * context->output_stride[3]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_k,
      tile_l,
      &context->params);
}

void xnn_compute_transposec_5d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t tile_l,
    size_t tile_m)
{
  const size_t ld_input = context->input_stride[4];
  const size_t ld_output = context->output_stride[3];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] + m * context->input_stride[4]);
  void* y = (void*)((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2] + l * context->output_stride[3] + m * context->output_stride[4]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_l,
      tile_m,
      &context->params);
}

void xnn_compute_transposec_6d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t n,
    size_t tile_m,
    size_t tile_n)
{
  const size_t ld_input = context->input_stride[5];
  const size_t ld_output = context->output_stride[4];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] +
                                 m * context->input_stride[4] + n * context->input_stride[5]);
  void* y = (void*)((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2] + l * context->output_stride[3] + m * context->output_stride[4] +
                     n * context->output_stride[5]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_m,
      tile_n,
      &context->params);
}

void xnn_compute_transposev_2d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  const size_t element_size = context->output_stride[1];
  const size_t ld_input = context->input_stride[1];
  const size_t ld_output = context->output_stride[0];
  const void* x = (const void*) ((uintptr_t) context->x +
                                 i * context->input_stride[0] + j * context->input_stride[1]);
  void* y = (void*) ((uintptr_t) context->y + context->output_stride[1] * j + i * context->output_stride[0]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[0],
      context->output_stride[1],
      element_size,
      tile_i,
      tile_j);
}

void xnn_compute_transposev_3d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t tile_j,
    size_t tile_k)
{
  const size_t element_size = context->output_stride[2];
  const size_t ld_input = context->input_stride[2];
  const size_t ld_output = context->output_stride[1];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2]);
  void* y = (void*)((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[1],
      context->output_stride[2],
      element_size,
      tile_j,
      tile_k);
}

void xnn_compute_transposev_4d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  const size_t element_size = context->output_stride[3];
  const size_t ld_input = context->input_stride[3];
  const size_t ld_output = context->output_stride[2];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[3] * l + i * context->output_stride[0] +
                     j * context->output_stride[1] + k * context->output_stride[2]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[2],
      context->output_stride[3],
      element_size,
      tile_k,
      tile_l);
}

void xnn_compute_transposev_5d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t tile_l,
    size_t tile_m)
{
  const size_t element_size = context->output_stride[4];
  const size_t ld_input = context->input_stride[4];
  const size_t ld_output = context->output_stride[3];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] + m * context->input_stride[4]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[4] * m + i * context->output_stride[0] +
                     j * context->output_stride[1] + k * context->output_stride[2] + l * context->output_stride[3]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[3],
      context->output_stride[4],
      element_size,
      tile_l,
      tile_m);
}

void xnn_compute_transposev_6d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t n,
    size_t tile_m,
    size_t tile_n)
{
  const size_t element_size = context->output_stride[5];
  const size_t ld_input = context->input_stride[5];
  const size_t ld_output = context->output_stride[4];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] +
                                 m * context->input_stride[4] + n * context->input_stride[5]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[5] * n + i * context->output_stride[0] +
                     j * context->output_stride[1] + k * context->output_stride[2] + l * context->output_stride[3] +
                     m * context->output_stride[4]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[4],
      context->output_stride[5],
      element_size,
      tile_m,
      tile_n);
}

void xnn_compute_grouped_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t k_scaled  = context->k_scaled;
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      k_scaled,
      (const void*) ((uintptr_t) context->a + mr_block_start * a_stride + group_index * k_scaled),
      a_stride,
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->wg_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize) + group_index * context->cg_stride),
      cm_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->k_scaled,
      (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
      a_stride,
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->fused_params);
}

void xnn_compute_spmm(
    const struct spmm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t mr_block_start,
    size_t mr_block_size)
{
  context->ukernel(
      mr_block_size,
      context->n,
      (const void*) ((uintptr_t) context->input + batch_index * context->batched_input_stride + mr_block_start),
      context->nonzero_weights,
      context->input_increments,
      context->output_channel_nonzeros,
      (void*) ((uintptr_t) context->output + batch_index * context->batched_output_stride + mr_block_start),
      context->scaled_m,
      &context->params);
}

void xnn_compute_grouped_batch_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) context->c + group_index * context->gc_stride + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset + group_index * context->ga_stride + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_grouped_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) context->c + group_index * context->gc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset + group_index * context->ga_stride,
      context->zero,
      &context->params);
}

void xnn_compute_batch_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset,
      context->zero,
      &context->params);
}

void xnn_compute_grouped_subgemm2d(
      const struct subgemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t ax_stride = context->ax_stride;
  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      (const void*) ((uintptr_t) context->a + group_index * context->ga_stride + slice_y * context->ay_stride + slice_x_start * ax_stride + batch_index * context->ba_stride),
      ax_stride,
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) subconvolution_params->output + group_index * context->gc_stride + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_subgemm2d(
      const struct subgemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t ax_stride = context->ax_stride;
  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      (const void*) ((uintptr_t) context->a + slice_y * context->ay_stride + slice_x_start * ax_stride + batch_index * context->ba_stride),
      ax_stride,
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride),
      (void*) ((uintptr_t) subconvolution_params->output + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_grouped_subconv2d(
      const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**) ((uintptr_t) subconvolution_params->indirection_buffer + slice_y * subconvolution_params->indirection_y_stride + slice_x_start * subconvolution_params->indirection_x_stride),
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) subconvolution_params->output + group_index * context->gc_stride + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      context->a_offset + group_index * context->ga_stride + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_subconv2d(
      const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**) ((uintptr_t) subconvolution_params->indirection_buffer + slice_y * subconvolution_params->indirection_y_stride + slice_x_start * subconvolution_params->indirection_x_stride),
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride),
      (void*) ((uintptr_t) subconvolution_params->output + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      context->a_offset + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_conv2d_hwc2chw(
      const struct conv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y_start,
      size_t output_y_slice)
{
  context->hwc2chw_ukernel(
      context->input_height,
      context->input_width,
      output_y_start,
      output_y_start + output_y_slice,
      (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride),
      context->zero,
      context->packed_weights,
      (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride),
      context->input_padding_top,
      context->output_channels,
      context->output_height_stride,
      context->output_channel_stride,
      &context->params);
}

void xnn_compute_dwconv_unipass(
    const struct dwconv_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
    context->groups, context->output_width,
    indirect_input, context->packed_weights, output,
    context->indirect_input_width_stride, context->output_increment,
    input_offset, context->zero,
    &context->params);
}

void xnn_compute_dwconv2d_chw(
    const struct dwconv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t channel)
{
  context->chw_ukernel(
    context->input_height,
    context->input_width,
    (const void*) ((uintptr_t) context->input + channel * context->input_channel_stride + batch_index * context->input_batch_stride),
    (const void*) ((uintptr_t) context->packed_weights + channel * context->weights_channel_stride),
    context->zero,
    (void*) ((uintptr_t) context->output + channel * context->output_channel_stride + batch_index * context->output_batch_stride),
    context->input_padding_top,
    &context->params);
}

void xnn_compute_argmax_pooling_unipass(
    const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input = (const void**) ((uintptr_t) context->indirect_input +
    output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  uint32_t* index = (uint32_t*) ((uintptr_t) context->index +
    batch_index * context->index_batch_stride + output_y * context->index_height_stride);

  context->unipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, output, index,
    context->input_increment, context->output_increment);
}

void xnn_compute_argmax_pooling_multipass(
    const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input = (const void**) ((uintptr_t) context->indirect_input +
    output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  uint32_t* index = (uint32_t*) ((uintptr_t) context->index +
    batch_index * context->index_batch_stride + output_y * context->index_height_stride);

  void* multipass_accumulation_buffer = XNN_SIMD_ALLOCA(context->channels * sizeof(float) + XNN_EXTRA_BYTES);
  void* multipass_index_buffer = XNN_SIMD_ALLOCA(context->channels * sizeof(uint32_t) + XNN_EXTRA_BYTES);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, multipass_accumulation_buffer, multipass_index_buffer, output, index,
    context->input_increment, context->output_increment);
}

void xnn_compute_max_pooling(
    const struct max_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input = (const void**) ((uintptr_t) context->indirect_input +
    output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_unpooling(
    const struct unpooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t input_y,
    size_t input_x)
{
  const void* input = (const void*) ((uintptr_t) context->input +
      input_y * context->input_height_stride + input_x * context->input_width_stride);
  const uint32_t* index = (const uint32_t*) ((uintptr_t) context->index +
      input_y * context->index_height_stride + input_x * context->index_width_stride);
  void** indirect_output =
    (void**) ((uintptr_t) context->indirect_output +
      input_y * context->indirect_output_height_stride + input_x * context->indirect_output_width_stride);

  context->ukernel(
    context->pooling_size,
    context->channels,
    context->fill_value,
    input, index, indirect_output);
}

void xnn_compute_average_pooling_unipass(
    const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_average_pooling_multipass(
    const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  void* multipass_buffer =
    XNN_SIMD_ALLOCA(context->channels * sizeof(int32_t) + XNN_EXTRA_BYTES * sizeof(int32_t) / sizeof(uint8_t));

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, multipass_buffer, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_pixelwise_average_pooling_unipass(
    const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  const void* pixelwise_buffer =
    (const void*) ((uintptr_t) context->pixelwise_buffer + output_y * context->pixelwise_buffer_height_stride);
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, pixelwise_buffer, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_pixelwise_average_pooling_multipass(
    const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  const void* pixelwise_buffer =
    (const void*) ((uintptr_t) context->pixelwise_buffer + output_y * context->pixelwise_buffer_height_stride);
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  void* multipass_buffer = XNN_SIMD_ALLOCA(context->channels * sizeof(int32_t) + XNN_EXTRA_BYTES * sizeof(int32_t) / sizeof(uint8_t));

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, pixelwise_buffer, multipass_buffer, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_global_average_pooling_nwc_unipass(
    const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* input =
    (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride);

  context->unipass_ukernel(
    context->input_elements,
    context->channels,
    input,
    context->input_pixel_stride,
    context->zero,
    output,
    &context->params);
}

void xnn_compute_global_average_pooling_nwc_multipass(
    const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* input =
    (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride);

  void* multipass_buffer =
    XNN_SIMD_ALLOCA(context->channels * sizeof(int32_t) + XNN_EXTRA_BYTES * sizeof(int32_t) / sizeof(uint8_t));

  context->multipass_ukernel(
    context->input_elements,
    context->channels,
    input,
    context->input_pixel_stride,
    context->zero,
    multipass_buffer,
    output,
    &context->params);
}

void xnn_compute_global_average_pooling_ncw(
    const struct global_average_pooling_ncw_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t channels_start,
    size_t channels_slice)
{
  const void* input = (const void*) ((uintptr_t) context->input +
    channels_start * context->input_channel_stride + batch_index * context->input_batch_stride);
  void* output = (void*) ((uintptr_t) context->output +
    channels_start * context->output_channel_stride + batch_index * context->output_batch_stride);

  context->ukernel(
    context->input_elements,
    channels_slice,
    input,
    output,
    &context->params);
}

void xnn_compute_resize_bilinear(
    const struct resize_bilinear_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t pixel_start,
    size_t pixel_range)
{
  void* output =
    (void*) ((uintptr_t) context->output + pixel_start * context->output_pixel_stride + batch_index * context->output_batch_stride);

  context->ukernel(
    pixel_range,
    context->scaled_channels,
    context->indirect_input + pixel_start * 4,
    context->input_offset + batch_index * context->input_batch_stride,
    (const void*) ((uintptr_t) context->packed_weights + (pixel_start << context->log2_wsize)),
    output,
    context->output_pixel_stride - context->scaled_channels);
}

void xnn_compute_resize_bilinear_chw(
    const struct resize_bilinear_chw_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t channel_start,
    size_t channel_range)
{
  void* output =
    (void*) ((uintptr_t) context->output + channel_start * context->output_channel_stride + batch_index * context->output_batch_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + channel_start * context->input_channel_stride;

  context->ukernel(
    context->output_pixels,
    channel_range,
    context->indirect_input,
    input_offset,
    context->packed_weights,
    output,
    context->input_channel_stride);
}

void xnn_compute_prelu(
    const struct prelu_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_start,
    size_t batch_range)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;
  const void* x = (const void*) ((uintptr_t) context->x + x_stride * batch_start);
  void* y = (void*) ((uintptr_t) context->y + y_stride * batch_start);

  context->ukernel(batch_range, context->n, x, x_stride, context->w, y, y_stride);
}

void xnn_compute_pad_5d(
    const struct pad_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l, size_t m)
{
  const void* input = (const void*) ((uintptr_t) context->input +
    i * context->input_stride[4] + j * context->input_stride[3] + k * context->input_stride[2] + l * context->input_stride[1] + m * context->input_stride[0]);
  void* output = (void*) ((uintptr_t) context->output +
    i * context->output_stride[4] + j * context->output_stride[3] + k * context->output_stride[2] + l * context->output_stride[1] + m * context->output_stride[0]);

  const size_t i_padding = context->pre_paddings[5];
  const size_t j_padding = context->pre_paddings[4];
  const size_t k_padding = context->pre_paddings[3];
  const size_t l_padding = context->pre_paddings[2];
  const size_t m_padding = context->pre_paddings[1];

  const size_t i_size = context->input_size[5];
  const size_t j_size = context->input_size[4];
  const size_t k_size = context->input_size[3];
  const size_t l_size = context->input_size[2];
  const size_t m_size = context->input_size[1];

  if XNN_LIKELY(i - i_padding < i_size && j - j_padding < j_size && k - k_padding < k_size &&
                l - l_padding < l_size && m - m_padding < m_size)
  {
    context->pad_ukernel(
      1 /* rows */,
      context->input_size[0], context->pre_paddings[0], context->post_paddings[0],
      input, 0 /* input stride */, output, 0 /* output stride */,
      context->padding_value);
  } else {
    context->fill_ukernel(1 /* rows */, context->output_size[0], output, 0 /* output stride */, context->padding_value);
  }
}

void xnn_compute_slice_1d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i)
{
  const void* input = (const void*) ((uintptr_t) context->input + i * context->input_stride[0]);
  void* output = (void*) ((uintptr_t) context->output + i * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_2d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j)
{
  const void* input =
      (const void*) ((uintptr_t) context->input +
                     i * context->input_stride[1] +
                     j * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[1] + j * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_3d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k)
{
  const void* input =
      (const void*) ((uintptr_t) context->input +
                     i * context->input_stride[2] +
                     j * context->input_stride[1] +
                     k * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[2] +
               j * context->output_stride[1] + k * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_4d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l)
{
  const void* input =
      (const void*) ((uintptr_t) context->input +
                     i * context->input_stride[3] +
                     j * context->input_stride[2] +
                     k * context->input_stride[1] +
                     l * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[3] +
               j * context->output_stride[2] + k * context->output_stride[1] + l * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_5d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l, size_t m)
{
  const void* input =
      (const void* ) ((uintptr_t) context->input +
                      i * context->input_stride[4] +
                      j * context->input_stride[3] +
                      k * context->input_stride[2] +
                      l * context->input_stride[1] +
                      m * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[4] +
               j * context->output_stride[3] + k * context->output_stride[2] +
               l * context->output_stride[1] + m * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_elementwise_binary_1d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i)
{
  const void* a = (const void*) ((uintptr_t) context->a + i * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b + i * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y + i * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_2d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j)
{
  const void* a = (const void*) ((uintptr_t) context->a + i * context->a_stride[3] + j * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b + i * context->b_stride[3] + j * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y + i * context->y_stride[3] + j * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_3d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k)
{
  const void* a = (const void*) ((uintptr_t) context->a +
    i * context->a_stride[2] + j * context->a_stride[3] + k * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b +
    i * context->b_stride[2] + j * context->b_stride[3] + k * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y +
    i * context->y_stride[2] + j * context->y_stride[3] + k * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_4d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l)
{
  const void* a = (const void*) ((uintptr_t) context->a +
    i * context->a_stride[1] + j * context->a_stride[2] + k * context->a_stride[3] + l * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b +
    i * context->b_stride[1] + j * context->b_stride[2] + k * context->b_stride[3] + l * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y +
    i * context->y_stride[1] + j * context->y_stride[2] + k * context->y_stride[3] + l * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_5d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l, size_t m)
{
  const void* a = (const void*) ((uintptr_t) context->a +
    i * context->a_stride[0] + j * context->a_stride[1] + k * context->a_stride[2] + l * context->a_stride[3] + m * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b +
    i * context->b_stride[0] + j * context->b_stride[1] + k * context->b_stride[2] + l * context->b_stride[3] + m * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y +
    i * context->y_stride[0] + j * context->y_stride[1] + k * context->y_stride[2] + l * context->y_stride[3] + m * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_channel_shuffle_fixed(
    const struct channel_shuffle_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t index)
{
  const void* x = (const void*) ((uintptr_t) context->x + index * context->x_stride);
  void* y = (void*) ((uintptr_t) context->y + index * context->y_stride);

  context->fixed_ukernel(context->n, x, y);
}

void xnn_compute_channel_shuffle_variable(
    const struct channel_shuffle_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t index)
{
  const void* x = (const void*) ((uintptr_t) context->x + index * context->x_stride);
  void* y = (void*) ((uintptr_t) context->y + index * context->y_stride);

  context->variable_ukernel(context->n, context->m, x, y);
}

void xnn_compute_lut_strided(
    const struct lut_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* x = (const void*) ((uintptr_t) context->x + context->x_stride * batch_index);
  void* y = (void*) ((uintptr_t) context->y + context->y_stride * batch_index);

  context->ukernel(context->n, x, y, context->t);
}

void xnn_compute_lut_contiguous(
    const struct lut_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t offset,
    size_t size)
{
  const void* x = (const void*) ((uintptr_t) context->x + offset);
  void* y = (void*) ((uintptr_t) context->y + offset);

  context->ukernel(size, x, y, context->t);
}

void xnn_compute_univector_strided(
    const struct univector_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t batch_range)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;

  const void* x = (const void*) ((uintptr_t) context->x + x_stride * batch_index);
  void* y = (void*) ((uintptr_t) context->y + y_stride * batch_index);
  do {
    context->ukernel(context->n, x, y, &context->params);
    x = (const void*) ((uintptr_t) x + x_stride);
    y = (void*) ((uintptr_t) y + y_stride);
  } while (--batch_range != 0);
}

void xnn_compute_univector_contiguous(
    const struct univector_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t offset,
    size_t size)
{
  const uint32_t log2_xsize = context->log2_xsize;
  const uint32_t log2_ysize = context->log2_ysize;
  const void* x = (const void*) ((uintptr_t) context->x + offset);
  void* y = (void*) ((uintptr_t) context->y + ((offset >> log2_xsize) << log2_ysize));
  context->ukernel(size, x, y, &context->params);
}

void xnn_compute_u8_softmax(
    const struct u8_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const uint8_t* x = (const uint8_t*) ((uintptr_t) context->x + context->x_stride * batch_index);
  uint8_t* y = (uint8_t*) ((uintptr_t) context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  uint8_t x_max = 0;
  context->rmax_ukernel(n, x, &x_max);
  const size_t adjustment = x_max ^ 255;
  const uint32_t* t = (const uint32_t*) context->t + adjustment;
  context->lut_norm_ukernel(n, x, t, y);
}

void xnn_compute_floating_point_softmax(
    const struct floating_point_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* x = (const void*) ((uintptr_t) context->x + context->x_stride * batch_index);
  void* y = (void*) ((uintptr_t) context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  // First pass: reduce-max
  union {
    float as_float;
    uint16_t as_half;
  } x_max;
  context->rmax_ukernel(n, x, &x_max);

  // Second pass: reduce-add & store exp(x-x_max)
  union {
    float as_float;
    uint16_t as_half;
  } y_sum;
  context->raddstoreexpminusmax_ukernel(n, x, &x_max, y, &y_sum, &context->expminus_params);

  // Third pass: scale y
  union {
    float as_float;
    uint16_t as_half;
  } y_scale;
  context->compute_reciprocal(&y_sum, &y_scale);
  context->vmulc_ukernel(n, y, &y_scale, y, &context->minmax_params);
}

void xnn_compute_vmulcaddc(
    const struct vmulcaddc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_start,
    size_t batch_size)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;

  const void* x = (const void*) ((uintptr_t) context->x + x_stride * batch_start);
  void* y = (void*) ((uintptr_t) context->y + y_stride * batch_start);

  context->ukernel(
    batch_size,
    context->n,
    x, x_stride,
    context->w,
    y, y_stride,
    &context->params);
}

#if XNN_MAX_UARCH_TYPES > 1
  void xnn_compute_hmp_grouped_gemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t k_scaled  = context->k_scaled;
    const size_t a_stride  = context->a_stride;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        k_scaled,
        (const void*) ((uintptr_t) context->a + mr_block_start * a_stride + group_index * k_scaled),
        a_stride,
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->wg_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize) + group_index * context->cg_stride),
        cm_stride,
        context->cn_stride,
        &context->params);
  }

  void xnn_compute_hmp_gemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t a_stride  = context->a_stride;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->k_scaled,
        (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
        a_stride,
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->fused_params);
  }

  void xnn_compute_hmp_grouped_batch_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t batch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
        (void*) ((uintptr_t) context->c + group_index * context->gc_stride + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset + group_index * context->ga_stride + batch_index * context->ba_stride,
        context->zero,
        &context->params);
  }

  void xnn_compute_hmp_grouped_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
        (void*) ((uintptr_t) context->c + group_index * context->gc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset + group_index * context->ga_stride,
        context->zero,
        &context->params);
  }

  void xnn_compute_batch_hmp_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t batch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset + batch_index * context->ba_stride,
        context->zero,
        &context->params);
  }

  void xnn_compute_hmp_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset,
        context->zero,
        &context->params);
  }
#endif  // XNN_MAX_UARCH_TYPES > 1


enum xnn_status xnn_run_operator(xnn_operator_t op, pthreadpool_t threadpool)
{
  return xnn_run_operator_with_index(op, 0, 0, threadpool);
}

enum xnn_status xnn_run_operator_with_index(
  xnn_operator_t op,
  size_t opdata_index,
  size_t operator_object_index,
  pthreadpool_t threadpool)
{
  switch (op->state) {
    case xnn_run_state_invalid:
      xnn_log_error("failed to run operator: operator was not successfully setup");
      return xnn_status_invalid_state;
    case xnn_run_state_ready:
      xnn_log_debug("running operator %zu:%zu (%s %s)", opdata_index,
                    operator_object_index,
                    xnn_operator_type_to_string(op->type),
                    xnn_microkernel_type_to_string(op->ukernel.type));
      break;
    case xnn_run_state_skip:
      xnn_log_debug("skip running operator %zu:%zu (%s %s)", opdata_index,
                    operator_object_index,
                    xnn_operator_type_to_string(op->type),
                    xnn_microkernel_type_to_string(op->ukernel.type));
      return xnn_status_success;
  }

  uint32_t flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  if (op->flags & XNN_FLAG_YIELD_WORKERS) {
    flags |= PTHREADPOOL_FLAG_YIELD_WORKERS;
  }
  switch (op->compute.type) {
    case xnn_parallelization_type_invalid:
      break;
    case xnn_parallelization_type_1d:
      assert(op->compute.range[0] != 0);
      pthreadpool_parallelize_1d(
          threadpool,
          op->compute.task_1d,
          &op->context,
          op->compute.range[0],
          flags);
      break;
    case xnn_parallelization_type_1d_tile_1d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.tile[0] != 0);
      pthreadpool_parallelize_1d_tile_1d(
          threadpool,
          op->compute.task_1d_tile_1d,
          &op->context,
          op->compute.range[0],
          op->compute.tile[0],
          flags);
      break;
    case xnn_parallelization_type_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      pthreadpool_parallelize_2d(
          threadpool,
          op->compute.task_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1],
          flags);
      break;
    case xnn_parallelization_type_2d_tile_1d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.tile[0] != 0);
      pthreadpool_parallelize_2d_tile_1d(
          threadpool,
          op->compute.task_2d_tile_1d,
          &op->context,
          op->compute.range[0], op->compute.range[1],
          op->compute.tile[0],
          flags);
      break;
    case xnn_parallelization_type_2d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_2d_tile_2d(
          threadpool,
          op->compute.task_2d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_3d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      pthreadpool_parallelize_3d(
          threadpool,
          op->compute.task_3d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2],
          flags);
      break;
    case xnn_parallelization_type_3d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_3d_tile_2d(
          threadpool,
          op->compute.task_3d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_4d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      pthreadpool_parallelize_4d(
          threadpool,
          op->compute.task_4d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3],
          flags);
      break;
    case xnn_parallelization_type_4d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_4d_tile_2d(
          threadpool,
          op->compute.task_4d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_5d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.range[4] != 0);
      pthreadpool_parallelize_5d(
          threadpool,
          op->compute.task_5d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3], op->compute.range[4],
          flags);
      break;
    case xnn_parallelization_type_5d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.range[4] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_5d_tile_2d(
          threadpool,
          op->compute.task_5d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3], op->compute.range[4],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_6d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.range[4] != 0);
      assert(op->compute.range[5] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_6d_tile_2d(
          threadpool,
          op->compute.task_6d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3], op->compute.range[4], op->compute.range[5],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
#if XNN_MAX_UARCH_TYPES > 1
    case xnn_parallelization_type_2d_tile_2d_with_uarch:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_2d_tile_2d_with_uarch(
          threadpool,
          op->compute.task_2d_tile_2d_with_id,
          &op->context,
          0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
          op->compute.range[0], op->compute.range[1],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_3d_tile_2d_with_uarch:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_3d_tile_2d_with_uarch(
          threadpool,
          op->compute.task_3d_tile_2d_with_id,
          &op->context,
          0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
          op->compute.range[0], op->compute.range[1], op->compute.range[2],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_4d_tile_2d_with_uarch:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_4d_tile_2d_with_uarch(
          threadpool,
          op->compute.task_4d_tile_2d_with_id,
          &op->context,
          0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
#endif  // XNN_MAX_UARCH_TYPES > 1
    default:
      XNN_UNREACHABLE;
  }
  return xnn_status_success;
}
