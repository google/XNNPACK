// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/dwconv.h"

size_t xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme_get_channel_tile(void) {
  return svcntsw();
}

size_t xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme_get_output_height_tile(void) {
  return 4;
}

__arm_locally_streaming
static void f32_dwconv_9p4xvc_sme_impl(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top,
    float output_min, float output_max)
{
  const size_t channels = input_pixel_stride / sizeof(float);
  const size_t input_width = input_height_stride / input_pixel_stride;
  const size_t output_width = output_height_stride / output_pixel_stride;
  const size_t channel_tile = svcntw();
  const svfloat32_t vmin = svdup_n_f32(output_min);
  const svfloat32_t vmax = svdup_n_f32(output_max);

  if (input_padding_top == 0 && valid_input_rows >= 6 && valid_output_rows == 4) {
    for (size_t c = 0; c < channels; c += channel_tile) {
      const svbool_t channel_pred = svwhilelt_b32_u64(c, channels);
      const float* w = (const float*) packed_weights + c * 10;
      const svfloat32_t vbias = svld1_f32(channel_pred, w);
      const svfloat32_t vk0 = svld1_f32(channel_pred, w + channel_tile);
      const svfloat32_t vk1 = svld1_f32(channel_pred, w + 2 * channel_tile);
      const svfloat32_t vk2 = svld1_f32(channel_pred, w + 3 * channel_tile);
      const svfloat32_t vk3 = svld1_f32(channel_pred, w + 4 * channel_tile);
      const svfloat32_t vk4 = svld1_f32(channel_pred, w + 5 * channel_tile);
      const svfloat32_t vk5 = svld1_f32(channel_pred, w + 6 * channel_tile);
      const svfloat32_t vk6 = svld1_f32(channel_pred, w + 7 * channel_tile);
      const svfloat32_t vk7 = svld1_f32(channel_pred, w + 8 * channel_tile);
      const svfloat32_t vk8 = svld1_f32(channel_pred, w + 9 * channel_tile);

      const uint8_t* in_ch_base = (const uint8_t*) ((const float*) input + c);
      uint8_t* out_ch_base = (uint8_t*) ((float*) output + c);

      const bool col_a_init_in_bounds = 0 - input_padding_left < input_width;
      const svbool_t col_a_init_pred = col_a_init_in_bounds ? channel_pred : svpfalse_b();
      const uint8_t* in_col_a = in_ch_base + (col_a_init_in_bounds ? (0 - input_padding_left) * input_pixel_stride : 0);
      svfloat32_t vi0_col_a = svld1_f32(col_a_init_pred, (const float*) in_col_a); in_col_a += input_height_stride;
      svfloat32_t vi1_col_a = svld1_f32(col_a_init_pred, (const float*) in_col_a); in_col_a += input_height_stride;
      svfloat32_t vi2_col_a = svld1_f32(col_a_init_pred, (const float*) in_col_a); in_col_a += input_height_stride;
      svfloat32_t vi3_col_a = svld1_f32(col_a_init_pred, (const float*) in_col_a); in_col_a += input_height_stride;
      svfloat32_t vi4_col_a = svld1_f32(col_a_init_pred, (const float*) in_col_a); in_col_a += input_height_stride;
      svfloat32_t vi5_col_a = svld1_f32(col_a_init_pred, (const float*) in_col_a);

      const bool col_b_init_in_bounds = 1 - input_padding_left < input_width;
      const svbool_t col_b_init_pred = col_b_init_in_bounds ? channel_pred : svpfalse_b();
      const uint8_t* in_col_b = in_ch_base + (col_b_init_in_bounds ? (1 - input_padding_left) * input_pixel_stride : 0);
      svfloat32_t vi0_col_b = svld1_f32(col_b_init_pred, (const float*) in_col_b); in_col_b += input_height_stride;
      svfloat32_t vi1_col_b = svld1_f32(col_b_init_pred, (const float*) in_col_b); in_col_b += input_height_stride;
      svfloat32_t vi2_col_b = svld1_f32(col_b_init_pred, (const float*) in_col_b); in_col_b += input_height_stride;
      svfloat32_t vi3_col_b = svld1_f32(col_b_init_pred, (const float*) in_col_b); in_col_b += input_height_stride;
      svfloat32_t vi4_col_b = svld1_f32(col_b_init_pred, (const float*) in_col_b); in_col_b += input_height_stride;
      svfloat32_t vi5_col_b = svld1_f32(col_b_init_pred, (const float*) in_col_b);

      size_t out_x = 0;
      for (; out_x + 2 <= output_width; out_x += 2) {
        {
          svfloat32_t vacc0 = svmla_f32_x(channel_pred, vbias, vi0_col_a, vk0);
          svfloat32_t vacc1 = svmla_f32_x(channel_pred, vbias, vi1_col_a, vk0);
          svfloat32_t vacc2 = svmla_f32_x(channel_pred, vbias, vi2_col_a, vk0);
          svfloat32_t vacc3 = svmla_f32_x(channel_pred, vbias, vi3_col_a, vk0);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk3);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk3);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk3);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk3);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk6);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk6);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk6);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk6);

          const bool col_a_in_bounds = out_x + 2 - input_padding_left < input_width;
          const svbool_t col_a_pred = col_a_in_bounds ? channel_pred : svpfalse_b();
          const uint8_t* in_col = in_ch_base + (col_a_in_bounds ? (out_x + 2 - input_padding_left) * input_pixel_stride : 0);
          vi0_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
          vi1_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
          vi2_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
          vi3_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
          vi4_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
          vi5_col_a = svld1_f32(col_a_pred, (const float*) in_col);

          vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_b, vk1);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_b, vk1);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_b, vk1);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_b, vk1);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk4);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk4);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk4);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk4);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk7);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk7);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk7);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk7);

          vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_a, vk2);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_a, vk2);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_a, vk2);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_a, vk2);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk5);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk5);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk5);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk5);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk8);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk8);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk8);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk8);

          vacc0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc0, vmin), vmax);
          vacc1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc1, vmin), vmax);
          vacc2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc2, vmin), vmax);
          vacc3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc3, vmin), vmax);

          uint8_t* out_col = out_ch_base;
          svst1_f32(channel_pred, (float*) out_col, vacc0);
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, vacc1);
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, vacc2);
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, vacc3);
          out_ch_base += output_pixel_stride;
        }
        {
          svfloat32_t vacc0 = svmla_f32_x(channel_pred, vbias, vi0_col_b, vk0);
          svfloat32_t vacc1 = svmla_f32_x(channel_pred, vbias, vi1_col_b, vk0);
          svfloat32_t vacc2 = svmla_f32_x(channel_pred, vbias, vi2_col_b, vk0);
          svfloat32_t vacc3 = svmla_f32_x(channel_pred, vbias, vi3_col_b, vk0);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk3);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk3);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk3);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk3);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk6);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk6);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk6);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk6);

          const bool col_b_in_bounds = out_x + 3 - input_padding_left < input_width;
          const svbool_t col_b_pred = col_b_in_bounds ? channel_pred : svpfalse_b();
          const uint8_t* in_col = in_ch_base + (col_b_in_bounds ? (out_x + 3 - input_padding_left) * input_pixel_stride : 0);
          vi0_col_b = svld1_f32(col_b_pred, (const float*) in_col); in_col += input_height_stride;
          vi1_col_b = svld1_f32(col_b_pred, (const float*) in_col); in_col += input_height_stride;
          vi2_col_b = svld1_f32(col_b_pred, (const float*) in_col); in_col += input_height_stride;
          vi3_col_b = svld1_f32(col_b_pred, (const float*) in_col); in_col += input_height_stride;
          vi4_col_b = svld1_f32(col_b_pred, (const float*) in_col); in_col += input_height_stride;
          vi5_col_b = svld1_f32(col_b_pred, (const float*) in_col);

          vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_a, vk1);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_a, vk1);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_a, vk1);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_a, vk1);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk4);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk4);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk4);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk4);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk7);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk7);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk7);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk7);

          vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_b, vk2);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_b, vk2);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_b, vk2);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_b, vk2);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk5);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk5);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk5);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk5);
          vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk8);
          vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk8);
          vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk8);
          vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk8);

          vacc0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc0, vmin), vmax);
          vacc1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc1, vmin), vmax);
          vacc2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc2, vmin), vmax);
          vacc3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc3, vmin), vmax);

          uint8_t* out_col = out_ch_base;
          svst1_f32(channel_pred, (float*) out_col, vacc0);
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, vacc1);
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, vacc2);
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, vacc3);
          out_ch_base += output_pixel_stride;
        }
      }

      if (out_x < output_width) {
        svfloat32_t vacc0 = svmla_f32_x(channel_pred, vbias, vi0_col_a, vk0);
        svfloat32_t vacc1 = svmla_f32_x(channel_pred, vbias, vi1_col_a, vk0);
        svfloat32_t vacc2 = svmla_f32_x(channel_pred, vbias, vi2_col_a, vk0);
        svfloat32_t vacc3 = svmla_f32_x(channel_pred, vbias, vi3_col_a, vk0);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk3);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk3);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk3);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk3);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk6);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk6);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk6);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk6);

        const bool col_a_in_bounds = out_x + 2 - input_padding_left < input_width;
        const svbool_t col_a_pred = col_a_in_bounds ? channel_pred : svpfalse_b();
        const uint8_t* in_col = in_ch_base + (col_a_in_bounds ? (out_x + 2 - input_padding_left) * input_pixel_stride : 0);
        vi0_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
        vi1_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
        vi2_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
        vi3_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
        vi4_col_a = svld1_f32(col_a_pred, (const float*) in_col); in_col += input_height_stride;
        vi5_col_a = svld1_f32(col_a_pred, (const float*) in_col);

        vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_b, vk1);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_b, vk1);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_b, vk1);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_b, vk1);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk4);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk4);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk4);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk4);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk7);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk7);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk7);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk7);

        vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_a, vk2);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_a, vk2);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_a, vk2);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_a, vk2);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk5);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk5);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk5);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk5);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk8);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk8);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk8);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk8);

        vacc0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc0, vmin), vmax);
        vacc1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc1, vmin), vmax);
        vacc2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc2, vmin), vmax);
        vacc3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc3, vmin), vmax);

        uint8_t* out_col = out_ch_base;
        svst1_f32(channel_pred, (float*) out_col, vacc0);
        out_col += output_height_stride;
        svst1_f32(channel_pred, (float*) out_col, vacc1);
        out_col += output_height_stride;
        svst1_f32(channel_pred, (float*) out_col, vacc2);
        out_col += output_height_stride;
        svst1_f32(channel_pred, (float*) out_col, vacc3);
      }
    }
    return;
  }

  const bool row0_valid = 0 >= input_padding_top && 0 - input_padding_top < valid_input_rows;
  const bool row1_valid = 1 >= input_padding_top && 1 - input_padding_top < valid_input_rows;
  const bool row2_valid = 2 >= input_padding_top && 2 - input_padding_top < valid_input_rows;
  const bool row3_valid = 3 >= input_padding_top && 3 - input_padding_top < valid_input_rows;
  const bool row4_valid = 4 >= input_padding_top && 4 - input_padding_top < valid_input_rows;
  const bool row5_valid = 5 >= input_padding_top && 5 - input_padding_top < valid_input_rows;
  const size_t in_row0_offset = row0_valid ? (0 - input_padding_top) * input_height_stride : 0;
  const size_t in_row1_offset = row1_valid ? (1 - input_padding_top) * input_height_stride : 0;
  const size_t in_row2_offset = row2_valid ? (2 - input_padding_top) * input_height_stride : 0;
  const size_t in_row3_offset = row3_valid ? (3 - input_padding_top) * input_height_stride : 0;
  const size_t in_row4_offset = row4_valid ? (4 - input_padding_top) * input_height_stride : 0;
  const size_t in_row5_offset = row5_valid ? (5 - input_padding_top) * input_height_stride : 0;
  const size_t out_row1_offset = valid_output_rows > 1 ? output_height_stride : 0;
  const size_t out_row2_offset = valid_output_rows > 2 ? 2 * output_height_stride : 0;
  const size_t out_row3_offset = valid_output_rows > 3 ? 3 * output_height_stride : 0;

  for (size_t c = 0; c < channels; c += channel_tile) {
    const svbool_t channel_pred = svwhilelt_b32_u64(c, channels);
    const svbool_t row0_pred = row0_valid ? channel_pred : svpfalse_b();
    const svbool_t row1_pred = row1_valid ? channel_pred : svpfalse_b();
    const svbool_t row2_pred = row2_valid ? channel_pred : svpfalse_b();
    const svbool_t row3_pred = row3_valid ? channel_pred : svpfalse_b();
    const svbool_t row4_pred = row4_valid ? channel_pred : svpfalse_b();
    const svbool_t row5_pred = row5_valid ? channel_pred : svpfalse_b();

    const float* w = (const float*) packed_weights + c * 10;
    const svfloat32_t vbias = svld1_f32(channel_pred, w);
    const svfloat32_t vk0 = svld1_f32(channel_pred, w + channel_tile);
    const svfloat32_t vk1 = svld1_f32(channel_pred, w + 2 * channel_tile);
    const svfloat32_t vk2 = svld1_f32(channel_pred, w + 3 * channel_tile);
    const svfloat32_t vk3 = svld1_f32(channel_pred, w + 4 * channel_tile);
    const svfloat32_t vk4 = svld1_f32(channel_pred, w + 5 * channel_tile);
    const svfloat32_t vk5 = svld1_f32(channel_pred, w + 6 * channel_tile);
    const svfloat32_t vk6 = svld1_f32(channel_pred, w + 7 * channel_tile);
    const svfloat32_t vk7 = svld1_f32(channel_pred, w + 8 * channel_tile);
    const svfloat32_t vk8 = svld1_f32(channel_pred, w + 9 * channel_tile);

    const uint8_t* in_ch_base = (const uint8_t*) ((const float*) input + c);
    uint8_t* out_ch_base = (uint8_t*) ((float*) output + c);

    const bool col_a_init_in_bounds = 0 - input_padding_left < input_width;
    const uint8_t* in_col_a = in_ch_base + (col_a_init_in_bounds ? (0 - input_padding_left) * input_pixel_stride : 0);
    svfloat32_t vi0_col_a = svld1_f32(col_a_init_in_bounds ? row0_pred : svpfalse_b(), (const float*) (in_col_a + in_row0_offset));
    svfloat32_t vi1_col_a = svld1_f32(col_a_init_in_bounds ? row1_pred : svpfalse_b(), (const float*) (in_col_a + in_row1_offset));
    svfloat32_t vi2_col_a = svld1_f32(col_a_init_in_bounds ? row2_pred : svpfalse_b(), (const float*) (in_col_a + in_row2_offset));
    svfloat32_t vi3_col_a = svld1_f32(col_a_init_in_bounds ? row3_pred : svpfalse_b(), (const float*) (in_col_a + in_row3_offset));
    svfloat32_t vi4_col_a = svld1_f32(col_a_init_in_bounds ? row4_pred : svpfalse_b(), (const float*) (in_col_a + in_row4_offset));
    svfloat32_t vi5_col_a = svld1_f32(col_a_init_in_bounds ? row5_pred : svpfalse_b(), (const float*) (in_col_a + in_row5_offset));

    const bool col_b_init_in_bounds = 1 - input_padding_left < input_width;
    const uint8_t* in_col_b = in_ch_base + (col_b_init_in_bounds ? (1 - input_padding_left) * input_pixel_stride : 0);
    svfloat32_t vi0_col_b = svld1_f32(col_b_init_in_bounds ? row0_pred : svpfalse_b(), (const float*) (in_col_b + in_row0_offset));
    svfloat32_t vi1_col_b = svld1_f32(col_b_init_in_bounds ? row1_pred : svpfalse_b(), (const float*) (in_col_b + in_row1_offset));
    svfloat32_t vi2_col_b = svld1_f32(col_b_init_in_bounds ? row2_pred : svpfalse_b(), (const float*) (in_col_b + in_row2_offset));
    svfloat32_t vi3_col_b = svld1_f32(col_b_init_in_bounds ? row3_pred : svpfalse_b(), (const float*) (in_col_b + in_row3_offset));
    svfloat32_t vi4_col_b = svld1_f32(col_b_init_in_bounds ? row4_pred : svpfalse_b(), (const float*) (in_col_b + in_row4_offset));
    svfloat32_t vi5_col_b = svld1_f32(col_b_init_in_bounds ? row5_pred : svpfalse_b(), (const float*) (in_col_b + in_row5_offset));

    size_t out_x = 0;
    for (; out_x + 2 <= output_width; out_x += 2) {
      {
        svfloat32_t vacc0 = svmla_f32_x(channel_pred, vbias, vi0_col_a, vk0);
        svfloat32_t vacc1 = svmla_f32_x(channel_pred, vbias, vi1_col_a, vk0);
        svfloat32_t vacc2 = svmla_f32_x(channel_pred, vbias, vi2_col_a, vk0);
        svfloat32_t vacc3 = svmla_f32_x(channel_pred, vbias, vi3_col_a, vk0);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk3);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk3);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk3);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk3);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk6);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk6);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk6);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk6);

        const bool col_a_in_bounds = out_x + 2 - input_padding_left < input_width;
        const uint8_t* in_col = in_ch_base + (col_a_in_bounds ? (out_x + 2 - input_padding_left) * input_pixel_stride : 0);
        vi0_col_a = svld1_f32(col_a_in_bounds ? row0_pred : svpfalse_b(), (const float*) (in_col + in_row0_offset));
        vi1_col_a = svld1_f32(col_a_in_bounds ? row1_pred : svpfalse_b(), (const float*) (in_col + in_row1_offset));
        vi2_col_a = svld1_f32(col_a_in_bounds ? row2_pred : svpfalse_b(), (const float*) (in_col + in_row2_offset));
        vi3_col_a = svld1_f32(col_a_in_bounds ? row3_pred : svpfalse_b(), (const float*) (in_col + in_row3_offset));
        vi4_col_a = svld1_f32(col_a_in_bounds ? row4_pred : svpfalse_b(), (const float*) (in_col + in_row4_offset));
        vi5_col_a = svld1_f32(col_a_in_bounds ? row5_pred : svpfalse_b(), (const float*) (in_col + in_row5_offset));

        vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_b, vk1);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_b, vk1);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_b, vk1);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_b, vk1);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk4);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk4);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk4);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk4);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk7);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk7);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk7);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk7);

        vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_a, vk2);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_a, vk2);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_a, vk2);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_a, vk2);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk5);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk5);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk5);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk5);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk8);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk8);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk8);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk8);

        vacc0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc0, vmin), vmax);
        vacc1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc1, vmin), vmax);
        vacc2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc2, vmin), vmax);
        vacc3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc3, vmin), vmax);

        uint8_t* out_col = out_ch_base;
        svst1_f32(channel_pred, (float*) (out_col + out_row3_offset), vacc3);
        svst1_f32(channel_pred, (float*) (out_col + out_row2_offset), vacc2);
        svst1_f32(channel_pred, (float*) (out_col + out_row1_offset), vacc1);
        svst1_f32(channel_pred, (float*) out_col, vacc0);
        out_ch_base += output_pixel_stride;
      }
      {
        svfloat32_t vacc0 = svmla_f32_x(channel_pred, vbias, vi0_col_b, vk0);
        svfloat32_t vacc1 = svmla_f32_x(channel_pred, vbias, vi1_col_b, vk0);
        svfloat32_t vacc2 = svmla_f32_x(channel_pred, vbias, vi2_col_b, vk0);
        svfloat32_t vacc3 = svmla_f32_x(channel_pred, vbias, vi3_col_b, vk0);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk3);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk3);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk3);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk3);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk6);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk6);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk6);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk6);

        const bool col_b_in_bounds = out_x + 3 - input_padding_left < input_width;
        const uint8_t* in_col = in_ch_base + (col_b_in_bounds ? (out_x + 3 - input_padding_left) * input_pixel_stride : 0);
        vi0_col_b = svld1_f32(col_b_in_bounds ? row0_pred : svpfalse_b(), (const float*) (in_col + in_row0_offset));
        vi1_col_b = svld1_f32(col_b_in_bounds ? row1_pred : svpfalse_b(), (const float*) (in_col + in_row1_offset));
        vi2_col_b = svld1_f32(col_b_in_bounds ? row2_pred : svpfalse_b(), (const float*) (in_col + in_row2_offset));
        vi3_col_b = svld1_f32(col_b_in_bounds ? row3_pred : svpfalse_b(), (const float*) (in_col + in_row3_offset));
        vi4_col_b = svld1_f32(col_b_in_bounds ? row4_pred : svpfalse_b(), (const float*) (in_col + in_row4_offset));
        vi5_col_b = svld1_f32(col_b_in_bounds ? row5_pred : svpfalse_b(), (const float*) (in_col + in_row5_offset));

        vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_a, vk1);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_a, vk1);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_a, vk1);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_a, vk1);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk4);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk4);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk4);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk4);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk7);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk7);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk7);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk7);

        vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_b, vk2);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_b, vk2);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_b, vk2);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_b, vk2);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk5);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk5);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk5);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk5);
        vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk8);
        vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk8);
        vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk8);
        vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk8);

        vacc0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc0, vmin), vmax);
        vacc1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc1, vmin), vmax);
        vacc2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc2, vmin), vmax);
        vacc3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc3, vmin), vmax);

        uint8_t* out_col = out_ch_base;
        svst1_f32(channel_pred, (float*) (out_col + out_row3_offset), vacc3);
        svst1_f32(channel_pred, (float*) (out_col + out_row2_offset), vacc2);
        svst1_f32(channel_pred, (float*) (out_col + out_row1_offset), vacc1);
        svst1_f32(channel_pred, (float*) out_col, vacc0);
        out_ch_base += output_pixel_stride;
      }
    }

    if (out_x < output_width) {
      svfloat32_t vacc0 = svmla_f32_x(channel_pred, vbias, vi0_col_a, vk0);
      svfloat32_t vacc1 = svmla_f32_x(channel_pred, vbias, vi1_col_a, vk0);
      svfloat32_t vacc2 = svmla_f32_x(channel_pred, vbias, vi2_col_a, vk0);
      svfloat32_t vacc3 = svmla_f32_x(channel_pred, vbias, vi3_col_a, vk0);
      vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk3);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk3);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk3);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk3);
      vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk6);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk6);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk6);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk6);

      const bool col_a_in_bounds = out_x + 2 - input_padding_left < input_width;
      const uint8_t* in_col = in_ch_base + (col_a_in_bounds ? (out_x + 2 - input_padding_left) * input_pixel_stride : 0);
      vi0_col_a = svld1_f32(col_a_in_bounds ? row0_pred : svpfalse_b(), (const float*) (in_col + in_row0_offset));
      vi1_col_a = svld1_f32(col_a_in_bounds ? row1_pred : svpfalse_b(), (const float*) (in_col + in_row1_offset));
      vi2_col_a = svld1_f32(col_a_in_bounds ? row2_pred : svpfalse_b(), (const float*) (in_col + in_row2_offset));
      vi3_col_a = svld1_f32(col_a_in_bounds ? row3_pred : svpfalse_b(), (const float*) (in_col + in_row3_offset));
      vi4_col_a = svld1_f32(col_a_in_bounds ? row4_pred : svpfalse_b(), (const float*) (in_col + in_row4_offset));
      vi5_col_a = svld1_f32(col_a_in_bounds ? row5_pred : svpfalse_b(), (const float*) (in_col + in_row5_offset));

      vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_b, vk1);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_b, vk1);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_b, vk1);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_b, vk1);
      vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_b, vk4);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_b, vk4);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_b, vk4);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_b, vk4);
      vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_b, vk7);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_b, vk7);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_b, vk7);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_b, vk7);

      vacc0 = svmla_f32_x(channel_pred, vacc0, vi0_col_a, vk2);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi1_col_a, vk2);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi2_col_a, vk2);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi3_col_a, vk2);
      vacc0 = svmla_f32_x(channel_pred, vacc0, vi1_col_a, vk5);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi2_col_a, vk5);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi3_col_a, vk5);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi4_col_a, vk5);
      vacc0 = svmla_f32_x(channel_pred, vacc0, vi2_col_a, vk8);
      vacc1 = svmla_f32_x(channel_pred, vacc1, vi3_col_a, vk8);
      vacc2 = svmla_f32_x(channel_pred, vacc2, vi4_col_a, vk8);
      vacc3 = svmla_f32_x(channel_pred, vacc3, vi5_col_a, vk8);

      vacc0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc0, vmin), vmax);
      vacc1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc1, vmin), vmax);
      vacc2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc2, vmin), vmax);
      vacc3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc3, vmin), vmax);

      uint8_t* out_col = out_ch_base;
      svst1_f32(channel_pred, (float*) (out_col + out_row3_offset), vacc3);
      svst1_f32(channel_pred, (float*) (out_col + out_row2_offset), vacc2);
      svst1_f32(channel_pred, (float*) (out_col + out_row1_offset), vacc1);
      svst1_f32(channel_pred, (float*) out_col, vacc0);
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme(
    const void* input, const void* packed_weights, void* output,
    size_t input_height_stride, size_t input_pixel_stride,
    size_t output_height_stride, size_t output_pixel_stride,
    size_t valid_input_rows, size_t valid_output_rows,
    size_t input_padding_left, size_t input_padding_top, float padding_value,
    float output_min, float output_max)
{
  if XNN_UNLIKELY(valid_output_rows == 0 || output_height_stride == 0 || input_pixel_stride == 0) {
    return;
  }
  assert(padding_value == 0.0f);
  assert(valid_output_rows <= 4);
  assert(output_pixel_stride != 0);
  (void) padding_value;

  f32_dwconv_9p4xvc_sme_impl(
      input, packed_weights, output, input_height_stride, input_pixel_stride,
      output_height_stride, output_pixel_stride, valid_input_rows,
      valid_output_rows, input_padding_left, input_padding_top, output_min,
      output_max);
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
