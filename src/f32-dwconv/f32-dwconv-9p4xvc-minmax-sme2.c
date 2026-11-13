// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/dwconv.h"

size_t xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2_get_channel_tile(void) {
  return svcntsw();
}

size_t xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2_get_output_height_tile(void) {
  return 4;
}

__arm_locally_streaming __arm_new("za")
static void f32_dwconv_9p4xvc_sme2_two_stripes(
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
  const size_t stripe_width = (output_width + 1) / 2;
  const size_t in_stripe_stride = stripe_width * input_pixel_stride;
  const size_t out_stripe_stride = stripe_width * output_pixel_stride;

  if (input_padding_top == 0 && valid_input_rows >= 6 && valid_output_rows == 4) {
    for (size_t c = 0; c < channels; c += channel_tile) {
      const svbool_t channel_pred = svwhilelt_b32_u64(c, channels);
      const float* w = (const float*) packed_weights + c * 10;
      const svfloat32_t vbias = svld1_f32(channel_pred, w);
      svwrite_za32_f32_vg1x4(4, svcreate4_f32(vbias, vbias, vbias, vbias));
      const svfloat32x4_t vbias_x4 = svread_za32_f32_vg1x4(4);
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
      svwrite_za32_f32_vg1x4(5, vbias_x4);
      XNN_PRAGMA_CLANG("clang loop unroll_count(2)")
      for (size_t step = 0; step < stripe_width + 3; step++) {
        const uint32_t za_slice = (uint32_t) (step * 2);
        if (step >= 3) {
          const size_t out_x = step - 3;
          const uint32_t out_za_slice = za_slice - 2;
          const svfloat32x4_t vacc_r02 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(out_za_slice), vmin, vmax);
          const svfloat32x4_t vacc_r13 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(out_za_slice + 1), vmin, vmax);
          uint8_t* out_col_s0 = out_ch_base + out_x * output_pixel_stride;
          if (stripe_width + out_x < output_width) {
            uint8_t* out_col_s1 = out_col_s0 + out_stripe_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r02, 0));
            svst1_f32(channel_pred, (float*) out_col_s1, svget4_f32(vacc_r02, 1));
            out_col_s0 += output_height_stride;
            out_col_s1 += output_height_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r13, 0));
            svst1_f32(channel_pred, (float*) out_col_s1, svget4_f32(vacc_r13, 1));
            out_col_s0 += output_height_stride;
            out_col_s1 += output_height_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r02, 2));
            svst1_f32(channel_pred, (float*) out_col_s1, svget4_f32(vacc_r02, 3));
            out_col_s0 += output_height_stride;
            out_col_s1 += output_height_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r13, 2));
            svst1_f32(channel_pred, (float*) out_col_s1, svget4_f32(vacc_r13, 3));
          } else {
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r02, 0));
            out_col_s0 += output_height_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r13, 0));
            out_col_s0 += output_height_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r02, 2));
            out_col_s0 += output_height_stride;
            svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r13, 2));
          }
        }
        svwrite_za32_f32_vg1x4(za_slice + 6, vbias_x4);
        svwrite_za32_f32_vg1x4(za_slice + 7, vbias_x4);
        if (step >= input_padding_left && stripe_width + step - input_padding_left < input_width) {
          const uint8_t* i0 = in_ch_base + (step - input_padding_left) * input_pixel_stride;
          const uint8_t* i1 = i0 + input_height_stride;
          const uint8_t* i2 = i1 + input_height_stride;
          const uint8_t* i3 = i2 + input_height_stride;
          const uint8_t* i4 = i3 + input_height_stride;
          const uint8_t* i5 = i4 + input_height_stride;
          const svfloat32_t vi0_s0 = svld1_f32(channel_pred, (const float*) i0);
          const svfloat32_t vi0_s1 = svld1_f32(channel_pred, (const float*) (i0 + in_stripe_stride));
          const svfloat32_t vi2_s0 = svld1_f32(channel_pred, (const float*) i2);
          const svfloat32_t vi2_s1 = svld1_f32(channel_pred, (const float*) (i2 + in_stripe_stride));
          const svfloat32x4_t vi_r02 = svcreate4_f32(vi0_s0, vi0_s1, vi2_s0, vi2_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r02, vk0);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r02, vk1);
          svmla_single_za32_f32_vg1x4(za_slice, vi_r02, vk2);

          const svfloat32_t vi1_s0 = svld1_f32(channel_pred, (const float*) i1);
          const svfloat32_t vi1_s1 = svld1_f32(channel_pred, (const float*) (i1 + in_stripe_stride));
          const svfloat32_t vi3_s0 = svld1_f32(channel_pred, (const float*) i3);
          const svfloat32_t vi3_s1 = svld1_f32(channel_pred, (const float*) (i3 + in_stripe_stride));
          const svfloat32x4_t vi_r13 = svcreate4_f32(vi1_s0, vi1_s1, vi3_s0, vi3_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r13, vk0);
          svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r13, vk3);
          svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r13, vk1);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r13, vk4);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r13, vk2);
          svmla_single_za32_f32_vg1x4(za_slice, vi_r13, vk5);

          const svfloat32_t vi4_s0 = svld1_f32(channel_pred, (const float*) i4);
          const svfloat32_t vi4_s1 = svld1_f32(channel_pred, (const float*) (i4 + in_stripe_stride));
          const svfloat32x4_t vi_r24 = svcreate4_f32(vi2_s0, vi2_s1, vi4_s0, vi4_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r24, vk3);
          svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r24, vk6);
          svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r24, vk4);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r24, vk7);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r24, vk5);
          svmla_single_za32_f32_vg1x4(za_slice, vi_r24, vk8);

          const svfloat32_t vi5_s0 = svld1_f32(channel_pred, (const float*) i5);
          const svfloat32_t vi5_s1 = svld1_f32(channel_pred, (const float*) (i5 + in_stripe_stride));
          const svfloat32x4_t vi_r35 = svcreate4_f32(vi3_s0, vi3_s1, vi5_s0, vi5_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r35, vk6);
          svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r35, vk7);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r35, vk8);
        } else {
          const bool stripe0_in_bounds = step - input_padding_left < input_width;
          const bool stripe1_in_bounds = stripe_width + step - input_padding_left < input_width;
          const svbool_t s0_pred = stripe0_in_bounds ? channel_pred : svpfalse_b();
          const svbool_t s1_pred = stripe1_in_bounds ? channel_pred : svpfalse_b();
          const uint8_t* i0_s0 = in_ch_base + (stripe0_in_bounds ? (step - input_padding_left) * input_pixel_stride : 0);
          const uint8_t* i0_s1 = in_ch_base + (stripe1_in_bounds ? (stripe_width + step - input_padding_left) * input_pixel_stride : 0);
          const uint8_t* i1_s0 = i0_s0 + input_height_stride;
          const uint8_t* i1_s1 = i0_s1 + input_height_stride;
          const uint8_t* i2_s0 = i1_s0 + input_height_stride;
          const uint8_t* i2_s1 = i1_s1 + input_height_stride;
          const uint8_t* i3_s0 = i2_s0 + input_height_stride;
          const uint8_t* i3_s1 = i2_s1 + input_height_stride;
          const uint8_t* i4_s0 = i3_s0 + input_height_stride;
          const uint8_t* i4_s1 = i3_s1 + input_height_stride;
          const uint8_t* i5_s0 = i4_s0 + input_height_stride;
          const uint8_t* i5_s1 = i4_s1 + input_height_stride;
          const svfloat32_t vi0_s0 = svld1_f32(s0_pred, (const float*) i0_s0);
          const svfloat32_t vi0_s1 = svld1_f32(s1_pred, (const float*) i0_s1);
          const svfloat32_t vi2_s0 = svld1_f32(s0_pred, (const float*) i2_s0);
          const svfloat32_t vi2_s1 = svld1_f32(s1_pred, (const float*) i2_s1);
          const svfloat32x4_t vi_r02 = svcreate4_f32(vi0_s0, vi0_s1, vi2_s0, vi2_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r02, vk0);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r02, vk1);
          svmla_single_za32_f32_vg1x4(za_slice, vi_r02, vk2);
          const svfloat32_t vi1_s0 = svld1_f32(s0_pred, (const float*) i1_s0);
          const svfloat32_t vi1_s1 = svld1_f32(s1_pred, (const float*) i1_s1);
          const svfloat32_t vi3_s0 = svld1_f32(s0_pred, (const float*) i3_s0);
          const svfloat32_t vi3_s1 = svld1_f32(s1_pred, (const float*) i3_s1);
          const svfloat32x4_t vi_r13 = svcreate4_f32(vi1_s0, vi1_s1, vi3_s0, vi3_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r13, vk0);
          svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r13, vk3);
          svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r13, vk1);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r13, vk4);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r13, vk2);
          svmla_single_za32_f32_vg1x4(za_slice, vi_r13, vk5);
          const svfloat32_t vi4_s0 = svld1_f32(s0_pred, (const float*) i4_s0);
          const svfloat32_t vi4_s1 = svld1_f32(s1_pred, (const float*) i4_s1);
          const svfloat32x4_t vi_r24 = svcreate4_f32(vi2_s0, vi2_s1, vi4_s0, vi4_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r24, vk3);
          svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r24, vk6);
          svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r24, vk4);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r24, vk7);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r24, vk5);
          svmla_single_za32_f32_vg1x4(za_slice, vi_r24, vk8);
          const svfloat32_t vi5_s0 = svld1_f32(s0_pred, (const float*) i5_s0);
          const svfloat32_t vi5_s1 = svld1_f32(s1_pred, (const float*) i5_s1);
          const svfloat32x4_t vi_r35 = svcreate4_f32(vi3_s0, vi3_s1, vi5_s0, vi5_s1);
          svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r35, vk6);
          svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r35, vk7);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r35, vk8);
        }
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
    svwrite_za32_f32_vg1x4(4, svcreate4_f32(vbias, vbias, vbias, vbias));
    const svfloat32x4_t vbias_x4 = svread_za32_f32_vg1x4(4);
    const svfloat32_t vk0 = svld1_f32(channel_pred, w + channel_tile);
    const svfloat32_t vk1 = svld1_f32(channel_pred, w + 2 * channel_tile);
    const svfloat32_t vk2 = svld1_f32(channel_pred, w + 3 * channel_tile);
    const svfloat32_t vk3 = svld1_f32(channel_pred, w + 4 * channel_tile);
    const svfloat32_t vk4 = svld1_f32(channel_pred, w + 5 * channel_tile);
    const svfloat32_t vk5 = svld1_f32(channel_pred, w + 6 * channel_tile);
    const svfloat32_t vk6 = svld1_f32(channel_pred, w + 7 * channel_tile);
    const svfloat32_t vk7 = svld1_f32(channel_pred, w + 8 * channel_tile);
    const svfloat32_t vk8 = svld1_f32(channel_pred, w + 9 * channel_tile);

    const uintptr_t in_ch_base = (uintptr_t) ((const float*) input + c);
    const uintptr_t out_ch_base = (uintptr_t) ((float*) output + c);
    svwrite_za32_f32_vg1x4(5, vbias_x4);
    for (size_t step = 0; step < stripe_width + 3; step++) {
      const uint32_t za_slice = (uint32_t) (step * 2);
      if (step >= 3) {
        const size_t out_x = step - 3;
        const uint32_t out_za_slice = za_slice - 2;
        const svfloat32x4_t vacc_r02 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(out_za_slice), vmin, vmax);
        const svfloat32x4_t vacc_r13 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(out_za_slice + 1), vmin, vmax);
        const uintptr_t out_col_s0 = out_ch_base + out_x * output_pixel_stride;
        const uintptr_t out_col_s1 = out_col_s0 + out_stripe_stride;
        if (stripe_width + out_x < output_width) {
          svst1_f32(channel_pred, (float*) (out_col_s0 + out_row3_offset), svget4_f32(vacc_r13, 2));
          svst1_f32(channel_pred, (float*) (out_col_s1 + out_row3_offset), svget4_f32(vacc_r13, 3));
          svst1_f32(channel_pred, (float*) (out_col_s0 + out_row2_offset), svget4_f32(vacc_r02, 2));
          svst1_f32(channel_pred, (float*) (out_col_s1 + out_row2_offset), svget4_f32(vacc_r02, 3));
          svst1_f32(channel_pred, (float*) (out_col_s0 + out_row1_offset), svget4_f32(vacc_r13, 0));
          svst1_f32(channel_pred, (float*) (out_col_s1 + out_row1_offset), svget4_f32(vacc_r13, 1));
          svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r02, 0));
          svst1_f32(channel_pred, (float*) out_col_s1, svget4_f32(vacc_r02, 1));
        } else {
          svst1_f32(channel_pred, (float*) (out_col_s0 + out_row3_offset), svget4_f32(vacc_r13, 2));
          svst1_f32(channel_pred, (float*) (out_col_s0 + out_row2_offset), svget4_f32(vacc_r02, 2));
          svst1_f32(channel_pred, (float*) (out_col_s0 + out_row1_offset), svget4_f32(vacc_r13, 0));
          svst1_f32(channel_pred, (float*) out_col_s0, svget4_f32(vacc_r02, 0));
        }
      }
      svwrite_za32_f32_vg1x4(za_slice + 6, vbias_x4);
      svwrite_za32_f32_vg1x4(za_slice + 7, vbias_x4);
      if (step >= input_padding_left && stripe_width + step - input_padding_left < input_width) {
        const uintptr_t in_col_s0 = in_ch_base + (step - input_padding_left) * input_pixel_stride;
        const uintptr_t in_col_s1 = in_col_s0 + in_stripe_stride;
        const svfloat32_t vi0_s0 = svld1_f32(row0_pred, (const float*) (in_col_s0 + in_row0_offset));
        const svfloat32_t vi0_s1 = svld1_f32(row0_pred, (const float*) (in_col_s1 + in_row0_offset));
        const svfloat32_t vi2_s0 = svld1_f32(row2_pred, (const float*) (in_col_s0 + in_row2_offset));
        const svfloat32_t vi2_s1 = svld1_f32(row2_pred, (const float*) (in_col_s1 + in_row2_offset));
        const svfloat32x4_t vi_r02 = svcreate4_f32(vi0_s0, vi0_s1, vi2_s0, vi2_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r02, vk0);
        svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r02, vk1);
        svmla_single_za32_f32_vg1x4(za_slice, vi_r02, vk2);
        const svfloat32_t vi1_s0 = svld1_f32(row1_pred, (const float*) (in_col_s0 + in_row1_offset));
        const svfloat32_t vi1_s1 = svld1_f32(row1_pred, (const float*) (in_col_s1 + in_row1_offset));
        const svfloat32_t vi3_s0 = svld1_f32(row3_pred, (const float*) (in_col_s0 + in_row3_offset));
        const svfloat32_t vi3_s1 = svld1_f32(row3_pred, (const float*) (in_col_s1 + in_row3_offset));
        const svfloat32x4_t vi_r13 = svcreate4_f32(vi1_s0, vi1_s1, vi3_s0, vi3_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r13, vk0);
        svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r13, vk3);
        svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r13, vk1);
        svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r13, vk4);
        svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r13, vk2);
        svmla_single_za32_f32_vg1x4(za_slice, vi_r13, vk5);
        const svfloat32_t vi4_s0 = svld1_f32(row4_pred, (const float*) (in_col_s0 + in_row4_offset));
        const svfloat32_t vi4_s1 = svld1_f32(row4_pred, (const float*) (in_col_s1 + in_row4_offset));
        const svfloat32x4_t vi_r24 = svcreate4_f32(vi2_s0, vi2_s1, vi4_s0, vi4_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r24, vk3);
        svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r24, vk6);
        svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r24, vk4);
        svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r24, vk7);
        svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r24, vk5);
        svmla_single_za32_f32_vg1x4(za_slice, vi_r24, vk8);
        const svfloat32_t vi5_s0 = svld1_f32(row5_pred, (const float*) (in_col_s0 + in_row5_offset));
        const svfloat32_t vi5_s1 = svld1_f32(row5_pred, (const float*) (in_col_s1 + in_row5_offset));
        const svfloat32x4_t vi_r35 = svcreate4_f32(vi3_s0, vi3_s1, vi5_s0, vi5_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r35, vk6);
        svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r35, vk7);
        svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r35, vk8);
      } else {
        const bool stripe0_in_bounds = step - input_padding_left < input_width;
        const bool stripe1_in_bounds = stripe_width + step - input_padding_left < input_width;
        const uintptr_t in_col_s0 = in_ch_base + (stripe0_in_bounds ? (step - input_padding_left) * input_pixel_stride : 0);
        const uintptr_t in_col_s1 = in_ch_base + (stripe1_in_bounds ? (stripe_width + step - input_padding_left) * input_pixel_stride : 0);
        const svfloat32_t vi0_s0 = stripe0_in_bounds ? svld1_f32(row0_pred, (const float*) (in_col_s0 + in_row0_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi0_s1 = stripe1_in_bounds ? svld1_f32(row0_pred, (const float*) (in_col_s1 + in_row0_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi2_s0 = stripe0_in_bounds ? svld1_f32(row2_pred, (const float*) (in_col_s0 + in_row2_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi2_s1 = stripe1_in_bounds ? svld1_f32(row2_pred, (const float*) (in_col_s1 + in_row2_offset)) : svdup_n_f32(0.0f);
        const svfloat32x4_t vi_r02 = svcreate4_f32(vi0_s0, vi0_s1, vi2_s0, vi2_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r02, vk0);
        svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r02, vk1);
        svmla_single_za32_f32_vg1x4(za_slice, vi_r02, vk2);
        const svfloat32_t vi1_s0 = stripe0_in_bounds ? svld1_f32(row1_pred, (const float*) (in_col_s0 + in_row1_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi1_s1 = stripe1_in_bounds ? svld1_f32(row1_pred, (const float*) (in_col_s1 + in_row1_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi3_s0 = stripe0_in_bounds ? svld1_f32(row3_pred, (const float*) (in_col_s0 + in_row3_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi3_s1 = stripe1_in_bounds ? svld1_f32(row3_pred, (const float*) (in_col_s1 + in_row3_offset)) : svdup_n_f32(0.0f);
        const svfloat32x4_t vi_r13 = svcreate4_f32(vi1_s0, vi1_s1, vi3_s0, vi3_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r13, vk0);
        svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r13, vk3);
        svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r13, vk1);
        svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r13, vk4);
        svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r13, vk2);
        svmla_single_za32_f32_vg1x4(za_slice, vi_r13, vk5);
        const svfloat32_t vi4_s0 = stripe0_in_bounds ? svld1_f32(row4_pred, (const float*) (in_col_s0 + in_row4_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi4_s1 = stripe1_in_bounds ? svld1_f32(row4_pred, (const float*) (in_col_s1 + in_row4_offset)) : svdup_n_f32(0.0f);
        const svfloat32x4_t vi_r24 = svcreate4_f32(vi2_s0, vi2_s1, vi4_s0, vi4_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r24, vk3);
        svmla_single_za32_f32_vg1x4(za_slice + 4, vi_r24, vk6);
        svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r24, vk4);
        svmla_single_za32_f32_vg1x4(za_slice + 2, vi_r24, vk7);
        svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r24, vk5);
        svmla_single_za32_f32_vg1x4(za_slice, vi_r24, vk8);
        const svfloat32_t vi5_s0 = stripe0_in_bounds ? svld1_f32(row5_pred, (const float*) (in_col_s0 + in_row5_offset)) : svdup_n_f32(0.0f);
        const svfloat32_t vi5_s1 = stripe1_in_bounds ? svld1_f32(row5_pred, (const float*) (in_col_s1 + in_row5_offset)) : svdup_n_f32(0.0f);
        const svfloat32x4_t vi_r35 = svcreate4_f32(vi3_s0, vi3_s1, vi5_s0, vi5_s1);
        svmla_single_za32_f32_vg1x4(za_slice + 5, vi_r35, vk6);
        svmla_single_za32_f32_vg1x4(za_slice + 3, vi_r35, vk7);
        svmla_single_za32_f32_vg1x4(za_slice + 1, vi_r35, vk8);
      }
    }
  }
}

__arm_locally_streaming __arm_new("za")
static void f32_dwconv_9p4xvc_sme2_one_column(
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
      svwrite_za32_f32_vg1x4(2, svcreate4_f32(vbias, vbias, vbias, vbias));
      const svfloat32x4_t vbias_x4 = svread_za32_f32_vg1x4(2);
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
      XNN_PRAGMA_CLANG("clang loop unroll_count(2)")
      for (size_t step = 0; step < output_width + 3; step++) {
        const uint32_t za_slice = (uint32_t) step;
        if (step >= 3) {
          const svfloat32x4_t vacc = svclamp_single_f32_x4(svread_za32_f32_vg1x4(za_slice - 1), vmin, vmax);
          uint8_t* out_col = out_ch_base + (step - 3) * output_pixel_stride;
          svst1_f32(channel_pred, (float*) out_col, svget4_f32(vacc, 0));
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, svget4_f32(vacc, 1));
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, svget4_f32(vacc, 2));
          out_col += output_height_stride;
          svst1_f32(channel_pred, (float*) out_col, svget4_f32(vacc, 3));
        }
        svwrite_za32_f32_vg1x4(za_slice + 3, vbias_x4);
        if (step - input_padding_left < input_width) {
          const uint8_t* in_col = in_ch_base + (step - input_padding_left) * input_pixel_stride;
          const svfloat32_t vi0 = svld1_f32(channel_pred, (const float*) in_col);
          in_col += input_height_stride;
          const svfloat32_t vi1 = svld1_f32(channel_pred, (const float*) in_col);
          in_col += input_height_stride;
          const svfloat32_t vi2 = svld1_f32(channel_pred, (const float*) in_col);
          in_col += input_height_stride;
          const svfloat32_t vi3 = svld1_f32(channel_pred, (const float*) in_col);
          const svfloat32x4_t vi0123 = svcreate4_f32(vi0, vi1, vi2, vi3);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi0123, vk0);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi0123, vk1);
          svmla_single_za32_f32_vg1x4(za_slice, vi0123, vk2);
          in_col += input_height_stride;
          const svfloat32_t vi4 = svld1_f32(channel_pred, (const float*) in_col);
          const svfloat32x4_t vi1234 = svcreate4_f32(vi1, vi2, vi3, vi4);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi1234, vk3);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi1234, vk4);
          svmla_single_za32_f32_vg1x4(za_slice, vi1234, vk5);
          in_col += input_height_stride;
          const svfloat32_t vi5 = svld1_f32(channel_pred, (const float*) in_col);
          const svfloat32x4_t vi2345 = svcreate4_f32(vi2, vi3, vi4, vi5);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vi2345, vk6);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vi2345, vk7);
          svmla_single_za32_f32_vg1x4(za_slice, vi2345, vk8);
        } else {
          const svfloat32_t vzero = svdup_n_f32(0.0f);
          const svfloat32x4_t vzero_x4 = svcreate4_f32(vzero, vzero, vzero, vzero);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vzero_x4, vk0);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vzero_x4, vk1);
          svmla_single_za32_f32_vg1x4(za_slice, vzero_x4, vk2);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vzero_x4, vk3);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vzero_x4, vk4);
          svmla_single_za32_f32_vg1x4(za_slice, vzero_x4, vk5);
          svmla_single_za32_f32_vg1x4(za_slice + 2, vzero_x4, vk6);
          svmla_single_za32_f32_vg1x4(za_slice + 1, vzero_x4, vk7);
          svmla_single_za32_f32_vg1x4(za_slice, vzero_x4, vk8);
        }
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
    svwrite_za32_f32_vg1x4(2, svcreate4_f32(vbias, vbias, vbias, vbias));
    const svfloat32x4_t vbias_x4 = svread_za32_f32_vg1x4(2);
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
    for (size_t step = 0; step < output_width + 3; step++) {
      const uint32_t za_slice = (uint32_t) step;
      if (step >= 3) {
        const svfloat32x4_t vacc = svclamp_single_f32_x4(svread_za32_f32_vg1x4(za_slice - 1), vmin, vmax);
        uint8_t* out_col = out_ch_base + (step - 3) * output_pixel_stride;
        svst1_f32(channel_pred, (float*) (out_col + out_row3_offset), svget4_f32(vacc, 3));
        svst1_f32(channel_pred, (float*) (out_col + out_row2_offset), svget4_f32(vacc, 2));
        svst1_f32(channel_pred, (float*) (out_col + out_row1_offset), svget4_f32(vacc, 1));
        svst1_f32(channel_pred, (float*) out_col, svget4_f32(vacc, 0));
      }
      svwrite_za32_f32_vg1x4(za_slice + 3, vbias_x4);
      const bool col_in_bounds = step - input_padding_left < input_width;
      const uint8_t* in_col = in_ch_base + (col_in_bounds ? (step - input_padding_left) * input_pixel_stride : 0);
      const svfloat32_t vi0 = svld1_f32(col_in_bounds ? row0_pred : svpfalse_b(), (const float*) (in_col + in_row0_offset));
      const svfloat32_t vi1 = svld1_f32(col_in_bounds ? row1_pred : svpfalse_b(), (const float*) (in_col + in_row1_offset));
      const svfloat32_t vi2 = svld1_f32(col_in_bounds ? row2_pred : svpfalse_b(), (const float*) (in_col + in_row2_offset));
      const svfloat32_t vi3 = svld1_f32(col_in_bounds ? row3_pred : svpfalse_b(), (const float*) (in_col + in_row3_offset));
      const svfloat32x4_t vi0123 = svcreate4_f32(vi0, vi1, vi2, vi3);
      svmla_single_za32_f32_vg1x4(za_slice + 2, vi0123, vk0);
      svmla_single_za32_f32_vg1x4(za_slice + 1, vi0123, vk1);
      svmla_single_za32_f32_vg1x4(za_slice, vi0123, vk2);
      const svfloat32_t vi4 = svld1_f32(col_in_bounds ? row4_pred : svpfalse_b(), (const float*) (in_col + in_row4_offset));
      const svfloat32x4_t vi1234 = svcreate4_f32(vi1, vi2, vi3, vi4);
      svmla_single_za32_f32_vg1x4(za_slice + 2, vi1234, vk3);
      svmla_single_za32_f32_vg1x4(za_slice + 1, vi1234, vk4);
      svmla_single_za32_f32_vg1x4(za_slice, vi1234, vk5);
      const svfloat32_t vi5 = svld1_f32(col_in_bounds ? row5_pred : svpfalse_b(), (const float*) (in_col + in_row5_offset));
      const svfloat32x4_t vi2345 = svcreate4_f32(vi2, vi3, vi4, vi5);
      svmla_single_za32_f32_vg1x4(za_slice + 2, vi2345, vk6);
      svmla_single_za32_f32_vg1x4(za_slice + 1, vi2345, vk7);
      svmla_single_za32_f32_vg1x4(za_slice, vi2345, vk8);
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_9p4xvc__sme2(
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

  const size_t output_width = output_height_stride / output_pixel_stride;
  if (svcntsw() >= 8 && output_width >= 28 &&
      (((output_pixel_stride & 127) == 0 && output_pixel_stride <= 768) ||
       output_height_stride <= 8192)) {
    f32_dwconv_9p4xvc_sme2_two_stripes(
        input, packed_weights, output, input_height_stride, input_pixel_stride,
        output_height_stride, output_pixel_stride, valid_input_rows,
        valid_output_rows, input_padding_left, input_padding_top, output_min,
        output_max);
  } else {
    f32_dwconv_9p4xvc_sme2_one_column(
        input, packed_weights, output, input_height_stride, input_pixel_stride,
        output_height_stride, output_pixel_stride, valid_input_rows,
        valid_output_rows, input_padding_left, input_padding_top, output_min,
        output_max);
  }
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
