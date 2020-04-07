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
  size_t step_height,
  size_t step_width,
  uint32_t log2_element_size)
{
  const void** indirection_buffer = op->indirection_buffer;
  const void* input               = op->input;
  const size_t input_pixel_stride = op->input_pixel_stride << log2_element_size;
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

  const bool any_dilation = (dilation_height | dilation_width) > 1;

  if (any_dilation) {
    // Clamp to the border doesn't work for pooling with dilation.
    const size_t adjusted_padding_top = input_padding_top % dilation_height;
    const size_t adjusted_padding_left = input_padding_left % dilation_width;
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        size_t safe_input_y = output_y * stride_height;
        if XNN_UNPREDICTABLE(safe_input_y < adjusted_padding_top) {
          safe_input_y += dilation_height;
        }
        safe_input_y -= adjusted_padding_top;

        size_t input_y = output_y * stride_height + pooling_y * dilation_height - input_padding_top;
        if XNN_UNPREDICTABLE(input_y >= input_height) {
          input_y = safe_input_y;
        }

        for (size_t output_x = 0; output_x < output_width; output_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            size_t safe_input_x = output_x * stride_width;
            if XNN_UNPREDICTABLE(safe_input_x < adjusted_padding_left) {
              safe_input_x += dilation_width;
            }
            safe_input_x -= adjusted_padding_left;

            size_t input_x = output_x * stride_width + pooling_x * dilation_width - input_padding_left;
            if XNN_UNPREDICTABLE(input_x >= input_width) {
              input_x = safe_input_x;
            }

            const size_t index = output_y * step_height + output_x * step_width * pooling_height + pooling_x * pooling_height + pooling_y;
            indirection_buffer[index] = input + (input_y * input_width + input_x) * input_pixel_stride;
          }
        }
      }
    }
  } else {
    const size_t input_x_max = input_width - 1;
    const size_t input_y_max = input_height - 1;
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      for (size_t pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
        const size_t input_y = min(doz(output_y * stride_height + pooling_y * dilation_height, input_padding_top), input_y_max);
        for (size_t output_x = 0; output_x < output_width; output_x++) {
          for (size_t pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
            const size_t input_x = min(doz(output_x * stride_width + pooling_x * dilation_width, input_padding_left), input_x_max);
            const size_t index = output_y * step_height + output_x * step_width * pooling_height + pooling_x * pooling_height + pooling_y;
            indirection_buffer[index] = input + (input_y * input_width + input_x) * input_pixel_stride;
          }
        }
      }
    }
  }
}

void xnn_indirection_init_resize_bilinear2d_f32(
  size_t input_pixel_stride,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  const void** indirection_buffer,
  float* packed_weights,
  bool align_corners,
  bool tensorflow_legacy)
{
  assert(input_height != 0);
  assert(input_height < 16777216 /* 2**24 */);
  assert(input_width != 0);
  assert(input_width < 16777216 /* 2**24 */);
  assert(output_height != 0);
  assert(output_height < 16777216 /* 2**24 */);
  assert(output_width != 0);
  assert(output_width < 16777216 /* 2**24 */);

  const int32_t width_adjustment = (int32_t) (align_corners && output_width != 1);
  const int32_t height_adjustment = (int32_t) (align_corners && output_height != 1);
  const float width_scale =
    (float) ((int32_t) input_width - width_adjustment) / (float) ((int32_t) output_width - width_adjustment);
  const float height_scale =
    (float) ((int32_t) input_height - height_adjustment) / (float) ((int32_t) output_height - height_adjustment);

  const uint32_t input_y_max = (uint32_t) input_height - 1;
  const uint32_t input_x_max = (uint32_t) input_width - 1;
  if (tensorflow_legacy) {
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      const float input_y = (float) (int32_t) output_y * height_scale;
      assert(input_y >= 0.0f);
      assert(input_y < (float) input_height);

      const uint32_t input_y_top = (uint32_t) (int32_t) input_y;
      const uint32_t input_y_bottom = math_min_u32(input_y_top + 1, input_y_max);
      const float alpha_y = input_y - (float) input_y_top;
      for (size_t output_x = 0; output_x < output_width; output_x++) {
        const float input_x = (float) (int32_t) output_x * width_scale;
        assert(input_x >= 0.0f);
        assert(input_x < (float) input_width);

        const uint32_t input_x_left = (uint32_t) (int32_t) input_x;
        const uint32_t input_x_right = math_min_u32(input_x_left + 1, input_x_max);
        const float alpha_x = input_x - (float) input_x_left;
        indirection_buffer[0] =
          (void*) ((uintptr_t) input + (input_y_top * input_width + input_x_left) * input_pixel_stride);
        indirection_buffer[1] =
          (void*) ((uintptr_t) input + (input_y_top * input_width + input_x_right) * input_pixel_stride);
        indirection_buffer[2] =
          (void*) ((uintptr_t) input + (input_y_bottom * input_width + input_x_left) * input_pixel_stride);
        indirection_buffer[3] =
          (void*) ((uintptr_t) input + (input_y_bottom * input_width + input_x_right) * input_pixel_stride);
        packed_weights[0] = alpha_x;
        packed_weights[1] = alpha_y;
        indirection_buffer += 4;
        packed_weights += 2;
      }
    }
  } else {
    const float height_offset = 0.5f * height_scale - 0.5f;
    const float width_offset = 0.5f * width_scale - 0.5f;
    for (size_t output_y = 0; output_y < output_height; output_y++) {
      float input_y = (float) (int32_t) output_y * height_scale + height_offset;
      input_y = math_min_f32(math_max_f32(input_y, 0.0f), (float) input_y_max);
      const uint32_t input_y_top = (uint32_t) (int32_t) input_y;
      assert((int32_t) input_y_top >= 0);
      const uint32_t input_y_bottom = math_min_u32(input_y_top + 1, input_y_max);
      const float alpha_y = input_y - (float) input_y_top;
      for (size_t output_x = 0; output_x < output_width; output_x++) {
        float input_x = (float) (int32_t) output_x * width_scale + width_offset;
        input_x = math_min_f32(math_max_f32(input_x, 0.0f), (float) input_x_max);
        const uint32_t input_x_left = (uint32_t) (int32_t) input_x;
        assert((int32_t) input_x_left >= 0);
        const uint32_t input_x_right = math_min_u32(input_x_left + 1, input_x_max);
        const float alpha_x = input_x - (float) input_x_left;
        indirection_buffer[0] =
          (void*) ((uintptr_t) input + (input_y_top * input_width + input_x_left) * input_pixel_stride);
        indirection_buffer[1] =
          (void*) ((uintptr_t) input + (input_y_top * input_width + input_x_right) * input_pixel_stride);
        indirection_buffer[2] =
          (void*) ((uintptr_t) input + (input_y_bottom * input_width + input_x_left) * input_pixel_stride);
        indirection_buffer[3] =
          (void*) ((uintptr_t) input + (input_y_bottom * input_width + input_x_right) * input_pixel_stride);
        packed_weights[0] = alpha_x;
        packed_weights[1] = alpha_y;
        indirection_buffer += 4;
        packed_weights += 2;
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
