// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include <fxdiv.h>

#include <xnnpack/indirection.h>
#include <xnnpack/operator.h>
#include <xnnpack/math.h>


void xnn_indirection_init_conv2d(
  xnn_operator_t op,
  size_t output_tile_size,
  uint32_t log2_element_size)
{
  const void** indirection_buffer          = op->indirection_buffer;
  const void* input                        = op->input;
  const void* zero                         = op->zero_buffer;
  const size_t input_pixel_stride          = op->input_pixel_stride << log2_element_size;
  const size_t input_height                = op->input_height;
  const size_t input_width                 = op->input_width;
  const size_t output_height               = op->output_height;
  const size_t output_width                = op->output_width;
  const size_t kernel_height               = op->kernel_height;
  const size_t kernel_width                = op->kernel_width;
  const size_t stride_height               = op->stride_height;
  const size_t stride_width                = op->stride_width;
  const size_t dilation_height             = op->dilation_height;
  const size_t dilation_width              = op->dilation_width;
  const size_t input_padding_top           = op->padding_top;
  const size_t input_padding_left          = op->padding_left;

  const size_t output_size = output_height * output_width;
  const size_t tiled_output_size = round_up(output_size, output_tile_size);
  const size_t kernel_size = kernel_height * kernel_width;

  const struct fxdiv_divisor_size_t output_width_divisor = fxdiv_init_size_t(output_width);

  for (size_t output_tile_start = 0; output_tile_start < tiled_output_size; output_tile_start += output_tile_size) {
    for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
      const size_t output_index = min(output_tile_start + output_tile_offset, output_size - 1);
      const struct fxdiv_result_size_t output_y_x = fxdiv_divide_size_t(output_index, output_width_divisor);
      const size_t output_x = output_y_x.remainder;
      const size_t output_y = output_y_x.quotient;
      for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const size_t input_y = output_y * stride_height + kernel_y * dilation_height - input_padding_top;
        if (input_y < input_height) {
          for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const size_t input_x = output_x * stride_width + kernel_x * dilation_width - input_padding_left;
            const size_t kernel_index = kernel_y * kernel_width + kernel_x;
            const size_t index = output_tile_start * kernel_size + kernel_index * output_tile_size + output_tile_offset;
            if (input_x < input_width) {
              indirection_buffer[index] = (const void*)
                ((uintptr_t) input + (input_y * input_width + input_x) * input_pixel_stride);
            } else {
              indirection_buffer[index] = zero;
            }
          }
        } else {
          for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
            const size_t kernel_index = kernel_y * kernel_width + kernel_x;
            const size_t index = output_tile_start * kernel_size + kernel_index * output_tile_size + output_tile_offset;
            indirection_buffer[index] = zero;
          }
        }
      }
    }
  }
}

void xnn_indirection_init_dwconv2d(
  xnn_operator_t op,
  size_t batch_start,
  size_t step_height,
  size_t step_width,
  uint32_t log2_element_size)
{
  const void** indirection_buffer = op->indirection_buffer;
  const void* input               = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride << log2_element_size;
  const void* zero                = op->zero_buffer;
  const size_t batch_size         = op->batch_size;
  const size_t input_height       = op->input_height;
  const size_t input_width        = op->input_width;
  const size_t output_height      = op->output_height;
  const size_t output_width       = op->output_width;
  const size_t kernel_height      = op->kernel_height;
  const size_t kernel_width       = op->kernel_width;
  const size_t stride_height      = op->stride_height;
  const size_t stride_width       = op->stride_width;
  const size_t dilation_height    = op->dilation_height;
  const size_t dilation_width     = op->dilation_width;
  const size_t input_padding_top  = op->padding_top;
  const size_t input_padding_left = op->padding_left;

  for (size_t batch_index = batch_start; batch_index < batch_size; batch_index++) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const size_t input_y = output_y * stride_height + kernel_y * dilation_height - input_padding_top;
        if (input_y < input_height) {
          for (size_t output_x = 0; output_x < output_width; output_x++) {
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t input_x = output_x * stride_width + kernel_x * dilation_width - input_padding_left;
              const size_t index = (batch_index * output_height + output_y) * step_height + output_x * step_width * kernel_height + kernel_x * kernel_height + kernel_y;
              if (input_x < input_width) {
                indirection_buffer[index] =
                  (const void*) ((uintptr_t) input + ((batch_index * input_height + input_y) * input_width + input_x) * input_pixel_stride);
              } else {
                indirection_buffer[index] = zero;
              }
            }
          }
        } else {
          for (size_t output_x = 0; output_x < output_width; output_x++) {
            for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
              const size_t index = (batch_index * output_height + output_y) * step_height + output_x * step_width * kernel_height + kernel_x * kernel_height + kernel_y;
              indirection_buffer[index] = zero;
            }
          }
        }
      }
    }
  }
}

void xnn_indirection_init_deconv2d(
  xnn_operator_t op,
  size_t output_tile_size,
  uint32_t log2_element_size)
{
  const void** indirection_buffer = op->indirection_buffer;
  const void* input               = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride << log2_element_size;
  const void* zero                = op->zero_buffer;
  const size_t input_height       = op->input_height;
  const size_t input_width        = op->input_width;
  const size_t output_height      = op->output_height;
  const size_t output_width       = op->output_width;
  const size_t kernel_height      = op->kernel_height;
  const size_t kernel_width       = op->kernel_width;
  const size_t stride_height      = op->stride_height;
  const size_t stride_width       = op->stride_width;
  const size_t dilation_height    = op->dilation_height;
  const size_t dilation_width     = op->dilation_width;
  const size_t padding_top        = op->padding_top;
  const size_t padding_left       = op->padding_left;

  const size_t output_size = output_height * output_width;
  const size_t tiled_output_size = round_up(output_size, output_tile_size);
  const size_t kernel_size = kernel_height * kernel_width;

  const struct fxdiv_divisor_size_t output_width_divisor = fxdiv_init_size_t(output_width);
  const struct fxdiv_divisor_size_t stride_height_divisor = fxdiv_init_size_t(stride_height);
  const struct fxdiv_divisor_size_t stride_width_divisor = fxdiv_init_size_t(stride_width);

  for (size_t output_tile_start = 0; output_tile_start < tiled_output_size; output_tile_start += output_tile_size) {
    for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
      const size_t output_index = min(output_tile_start + output_tile_offset, output_size - 1);
      const struct fxdiv_result_size_t output_y_x = fxdiv_divide_size_t(output_index, output_width_divisor);
      const size_t output_x = output_y_x.remainder;
      const size_t output_y = output_y_x.quotient;
      for (size_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
        const size_t y = output_y + padding_top - kernel_y * dilation_height;
        const size_t input_y = fxdiv_quotient_size_t(y, stride_height_divisor);
        for (size_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
          const size_t x = output_x + padding_left - kernel_x * dilation_width;
          const size_t input_x = fxdiv_quotient_size_t(x, stride_width_divisor);
          const size_t kernel_index = kernel_y * kernel_width + kernel_x;
          const size_t index = output_tile_start * kernel_size + kernel_index * output_tile_size + output_tile_offset;
          if (input_y * stride_height == y && input_y < input_height && input_x * stride_width == x && input_x < input_width) {
            indirection_buffer[index] = (const void*) ((uintptr_t) input + (input_y * input_width + input_x) * input_pixel_stride);
          } else {
            indirection_buffer[index] = zero;
          }
        }
      }
    }
  }
}

void xnn_indirection_init_subconv2d(
  xnn_operator_t op,
  size_t output_tile_size,
  uint32_t log2_element_size)
{
  const void** indirection_buffer                     = op->indirection_buffer;
  struct subconvolution_params* subconvolution_params = op->subconvolution_buffer;
  const void* input                                   = op->input;
  const size_t input_pixel_stride                     = op->input_pixel_stride << log2_element_size;
  const void* zero                                    = op->zero_buffer;
  const size_t input_height                           = op->input_height;
  const size_t input_width                            = op->input_width;
  const size_t output_height                          = op->output_height;
  const size_t output_width                           = op->output_width;
  const size_t kernel_height                          = op->kernel_height;
  const size_t kernel_width                           = op->kernel_width;
  const size_t stride_height                          = op->stride_height;
  const size_t stride_width                           = op->stride_width;
  const size_t padding_top                            = op->padding_top;
  const size_t padding_left                           = op->padding_left;

  const size_t modulo_padding_top = padding_top % stride_height;
  const size_t modulo_padding_left = padding_left % stride_width;
  for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
    const size_t output_y_start = subtract_modulo(offset_y, modulo_padding_top, stride_height);
    for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
      const size_t output_x_start = subtract_modulo(offset_x, modulo_padding_left, stride_width);
      const size_t sliced_output_width = divide_round_up(output_width - output_x_start, stride_width);

      subconvolution_params->indirection_buffer = indirection_buffer;
      subconvolution_params->indirection_y_stride =
        subconvolution_params->indirection_x_stride * round_up(sliced_output_width, output_tile_size);
      ++subconvolution_params;

      for (size_t output_y = output_y_start; output_y < output_height; output_y += stride_height) {
        for (size_t output_tile_start = 0; output_tile_start < sliced_output_width; output_tile_start += output_tile_size) {
          for (size_t kernel_y = offset_y; kernel_y < kernel_height; kernel_y += stride_height) {
            assert(doz(output_y + padding_top, kernel_y) % stride_height == 0);
            const size_t y = output_y + padding_top - kernel_y;
            const size_t input_y = y / stride_height;

            for (size_t kernel_x = offset_x; kernel_x < kernel_width; kernel_x += stride_width) {
              for (size_t output_tile_offset = 0; output_tile_offset < output_tile_size; output_tile_offset++) {
                const size_t sliced_output_x = min(output_tile_start + output_tile_offset, sliced_output_width - 1);
                const size_t output_x = output_x_start + sliced_output_x * stride_width;

                assert(doz(output_x + padding_left, kernel_x) % stride_width == 0);
                const size_t x = output_x + padding_left - kernel_x;
                const size_t input_x = x / stride_width;

                if (input_y < input_height && input_x < input_width) {
                  *indirection_buffer++ =
                    (const void*) ((uintptr_t) input + (input_y * input_width + input_x) * input_pixel_stride);
                } else {
                  *indirection_buffer++ = zero;
                }
              }
            }
          }
        }
      }
    }
  }
}

void xnn_indirection_init_maxpool2d(
  xnn_operator_t op,
  size_t batch_start,
  size_t step_height,
  size_t step_width,
  uint32_t log2_element_size)
{
  const void** indirection_buffer = op->indirection_buffer;
  const void* input               = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride << log2_element_size;
  const size_t batch_size         = op->batch_size;
  const size_t input_height       = op->input_height;
  const size_t input_width        = op->input_width;
  const size_t output_height      = op->output_height;
  const size_t output_width       = op->output_width;
  const size_t pooling_height     = op->kernel_height;
  const size_t pooling_width      = op->kernel_width;
  const size_t stride_height      = op->stride_height;
  const size_t stride_width       = op->stride_width;
  const size_t dilation_height    = op->dilation_height;
  const size_t dilation_width     = op->dilation_width;
  const size_t input_padding_top  = op->padding_top;
  const size_t input_padding_left = op->padding_left;

  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        const size_t input_y = doz(output_y * stride_height + pooling_y * dilation_height, input_padding_top);
        const size_t clamped_input_y = min(input_y, input_height - 1);
        for (size_t output_x = 0; output_x < output_width; output_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            const size_t input_x = doz(output_x * stride_width + pooling_x * dilation_width, input_padding_left);
            const size_t clamped_input_x = min(input_x, input_width - 1);
            const size_t index = (image * output_height + output_y) * step_height + output_x * step_width * pooling_height + pooling_x * pooling_height + pooling_y;
            indirection_buffer[index] = input + ((image * input_height + clamped_input_y) * input_width + clamped_input_x) * input_pixel_stride;
          }
        }
      }
    }
  }
}

void xnn_indirection_init_unpool2d(
  xnn_operator_t op,
  size_t batch_start,
  uint32_t log2_element_size)
{
  const void** indirection_buffer  = op->indirection_buffer;
  const void* output               = op->output;
  const size_t output_pixel_stride = op->output_pixel_stride << log2_element_size;
  const size_t batch_size          = op->batch_size;
  const size_t input_height        = op->input_height;
  const size_t input_width         = op->input_width;
  const size_t output_height       = op->output_height;
  const size_t output_width        = op->output_width;
  const size_t pooling_height      = op->kernel_height;
  const size_t pooling_width       = op->kernel_width;
  const size_t output_padding_top  = op->padding_top;
  const size_t output_padding_left = op->padding_left;

  for (size_t image = batch_start; image < batch_size; image++) {
    for (size_t input_y = 0; input_y < input_height; input_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        const size_t output_y = min(doz(input_y * pooling_height + pooling_y, output_padding_top), output_height - 1);
        for (size_t input_x = 0; input_x < input_width; input_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            const size_t output_x = min(doz(input_x * pooling_width + pooling_x, output_padding_left), output_width - 1);
            indirection_buffer[(((image * input_height + input_y) * input_width + input_x) * pooling_width + pooling_x) * pooling_height + pooling_y] =
              output + ((image * output_height + output_y) * output_width + output_x) * output_pixel_stride;
          }
        }
      }
    }
  }
}
