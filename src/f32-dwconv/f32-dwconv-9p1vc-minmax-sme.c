// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"

size_t xnn_f32_dwconv_minmax_ukernel_9p1vc__sme_get_channel_tile(void) {
  return svcntsw();
}

__arm_locally_streaming
static void f32_dwconv_9p1vc_sme_channel_outer(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    float output_min,
    float output_max)
{
  const size_t channel_tile = svcntw();
  const svfloat32_t vmin = svdup_n_f32(output_min);
  const svfloat32_t vmax = svdup_n_f32(output_max);
  const size_t output_pixel_stride = channels * sizeof(float) + output_increment;

  for (size_t c = 0; c < channels; c += channel_tile) {
    const svbool_t channel_pred = svwhilelt_b32_u64(c, channels);
    const float* w = weights + c * 10;
    const float* zero_ch = zero + c;
    const size_t in_ch_offset = input_offset + c * sizeof(float);
    uintptr_t indirection = (uintptr_t) input;
    uint8_t* out_ptr = (uint8_t*) (output + c);
    size_t out_x = 0;

    if (input_pixel_stride == 0 && output_width >= 4) {
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

      for (; out_x + 4 <= output_width; out_x += 4) {
        const float* const* indirection_p0 = (const float* const*) indirection;
        const float* const* indirection_p1 = (const float* const*) (indirection + (uintptr_t) input_stride);
        const float* const* indirection_p2 = (const float* const*) (indirection + (uintptr_t) (2 * input_stride));
        const float* const* indirection_p3 = (const float* const*) (indirection + (uintptr_t) (3 * input_stride));
        indirection += (uintptr_t) (4 * input_stride);

        svfloat32_t vacc_p0 = svmla_f32_x(channel_pred, vbias, svld1_f32(channel_pred, indirection_p0[0] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[0] + in_ch_offset)), vk0);
        svfloat32_t vacc_p1 = svmla_f32_x(channel_pred, vbias, svld1_f32(channel_pred, indirection_p1[0] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[0] + in_ch_offset)), vk0);
        svfloat32_t vacc_p2 = svmla_f32_x(channel_pred, vbias, svld1_f32(channel_pred, indirection_p2[0] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[0] + in_ch_offset)), vk0);
        svfloat32_t vacc_p3 = svmla_f32_x(channel_pred, vbias, svld1_f32(channel_pred, indirection_p3[0] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[0] + in_ch_offset)), vk0);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[1] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[1] + in_ch_offset)), vk1);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[1] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[1] + in_ch_offset)), vk1);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[1] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[1] + in_ch_offset)), vk1);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[1] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[1] + in_ch_offset)), vk1);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[2] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[2] + in_ch_offset)), vk2);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[2] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[2] + in_ch_offset)), vk2);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[2] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[2] + in_ch_offset)), vk2);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[2] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[2] + in_ch_offset)), vk2);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[3] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[3] + in_ch_offset)), vk3);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[3] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[3] + in_ch_offset)), vk3);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[3] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[3] + in_ch_offset)), vk3);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[3] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[3] + in_ch_offset)), vk3);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[4] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[4] + in_ch_offset)), vk4);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[4] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[4] + in_ch_offset)), vk4);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[4] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[4] + in_ch_offset)), vk4);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[4] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[4] + in_ch_offset)), vk4);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[5] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[5] + in_ch_offset)), vk5);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[5] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[5] + in_ch_offset)), vk5);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[5] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[5] + in_ch_offset)), vk5);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[5] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[5] + in_ch_offset)), vk5);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[6] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[6] + in_ch_offset)), vk6);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[6] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[6] + in_ch_offset)), vk6);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[6] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[6] + in_ch_offset)), vk6);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[6] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[6] + in_ch_offset)), vk6);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[7] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[7] + in_ch_offset)), vk7);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[7] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[7] + in_ch_offset)), vk7);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[7] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[7] + in_ch_offset)), vk7);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[7] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[7] + in_ch_offset)), vk7);

        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, indirection_p0[8] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p0[8] + in_ch_offset)), vk8);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, indirection_p1[8] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p1[8] + in_ch_offset)), vk8);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, indirection_p2[8] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p2[8] + in_ch_offset)), vk8);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, indirection_p3[8] == zero ? zero_ch : (const float*) ((uintptr_t) indirection_p3[8] + in_ch_offset)), vk8);

        vacc_p0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p0, vmin), vmax);
        vacc_p1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p1, vmin), vmax);
        vacc_p2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p2, vmin), vmax);
        vacc_p3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p3, vmin), vmax);

        svst1_f32(channel_pred, (float*) out_ptr, vacc_p0); out_ptr += output_pixel_stride;
        svst1_f32(channel_pred, (float*) out_ptr, vacc_p1); out_ptr += output_pixel_stride;
        svst1_f32(channel_pred, (float*) out_ptr, vacc_p2); out_ptr += output_pixel_stride;
        svst1_f32(channel_pred, (float*) out_ptr, vacc_p3); out_ptr += output_pixel_stride;
      }
    }

    size_t in_offset = in_ch_offset + out_x * input_pixel_stride;
    for (; out_x < output_width; out_x += 4) {
      const size_t rem_pixels = output_width - out_x;
      const float* const* indirection_p0 = (const float* const*) indirection;
      const float* const* indirection_p1 = rem_pixels > 1 ? (const float* const*) (indirection + (uintptr_t) input_stride) : indirection_p0;
      const float* const* indirection_p2 = rem_pixels > 2 ? (const float* const*) (indirection + (uintptr_t) (2 * input_stride)) : indirection_p1;
      const float* const* indirection_p3 = rem_pixels > 3 ? (const float* const*) (indirection + (uintptr_t) (3 * input_stride)) : indirection_p2;
      const size_t in_offset_p0 = in_offset;
      const size_t in_offset_p1 = rem_pixels > 1 ? in_offset_p0 + input_pixel_stride : in_offset_p0;
      const size_t in_offset_p2 = rem_pixels > 2 ? in_offset_p1 + input_pixel_stride : in_offset_p1;
      const size_t in_offset_p3 = rem_pixels > 3 ? in_offset_p2 + input_pixel_stride : in_offset_p2;
      indirection += (uintptr_t) (4 * input_stride);
      in_offset += 4 * input_pixel_stride;

      svfloat32_t vacc_p0 = svld1_f32(channel_pred, w);
      svfloat32_t vacc_p1 = vacc_p0;
      svfloat32_t vacc_p2 = vacc_p0;
      svfloat32_t vacc_p3 = vacc_p0;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t tap = 0; tap < 9; ++tap) {
        const svfloat32_t vk = svld1_f32(channel_pred, w + (1 + tap) * channel_tile);
        const float* i_p0 = indirection_p0[tap];
        const float* i_p1 = indirection_p1[tap];
        const float* i_p2 = indirection_p2[tap];
        const float* i_p3 = indirection_p3[tap];
        vacc_p0 = svmla_f32_x(channel_pred, vacc_p0, svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_offset_p0)), vk);
        vacc_p1 = svmla_f32_x(channel_pred, vacc_p1, svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_offset_p1)), vk);
        vacc_p2 = svmla_f32_x(channel_pred, vacc_p2, svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_offset_p2)), vk);
        vacc_p3 = svmla_f32_x(channel_pred, vacc_p3, svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_offset_p3)), vk);
      }

      vacc_p0 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p0, vmin), vmax);
      svst1_f32(channel_pred, (float*) out_ptr, vacc_p0);
      if (rem_pixels > 1) {
        out_ptr += output_pixel_stride;
        vacc_p1 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p1, vmin), vmax);
        svst1_f32(channel_pred, (float*) out_ptr, vacc_p1);
        if (rem_pixels > 2) {
          out_ptr += output_pixel_stride;
          vacc_p2 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p2, vmin), vmax);
          svst1_f32(channel_pred, (float*) out_ptr, vacc_p2);
          if (rem_pixels > 3) {
            out_ptr += output_pixel_stride;
            vacc_p3 = svminnm_f32_x(channel_pred, svmaxnm_f32_x(channel_pred, vacc_p3, vmin), vmax);
            svst1_f32(channel_pred, (float*) out_ptr, vacc_p3);
          }
        }
      }
      out_ptr += output_pixel_stride;
    }
  }
}

__arm_locally_streaming
static void f32_dwconv_9p1vc_sme_width_outer(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    float output_min,
    float output_max)
{
  const size_t channel_tile = svcntw();
  const svfloat32_t vmin = svdup_n_f32(output_min);
  const svfloat32_t vmax = svdup_n_f32(output_max);
  const size_t output_pixel_stride = channels * sizeof(float) + output_increment;

  uintptr_t indirection = (uintptr_t) input;
  uint8_t* out_base = (uint8_t*) output;
  size_t in_pixel_offset = input_offset;

  for (size_t out_x = 0; out_x < output_width; out_x += 4) {
    const size_t rem_pixels = output_width - out_x;
    const float* const* indirection_p0 = (const float* const*) indirection;
    const float* const* indirection_p1 = rem_pixels > 1 ? (const float* const*) (indirection + (uintptr_t) input_stride) : indirection_p0;
    const float* const* indirection_p2 = rem_pixels > 2 ? (const float* const*) (indirection + (uintptr_t) (2 * input_stride)) : indirection_p1;
    const float* const* indirection_p3 = rem_pixels > 3 ? (const float* const*) (indirection + (uintptr_t) (3 * input_stride)) : indirection_p2;
    const size_t in_offset_p0 = in_pixel_offset;
    const size_t in_offset_p1 = rem_pixels > 1 ? in_offset_p0 + input_pixel_stride : in_offset_p0;
    const size_t in_offset_p2 = rem_pixels > 2 ? in_offset_p1 + input_pixel_stride : in_offset_p1;
    const size_t in_offset_p3 = rem_pixels > 3 ? in_offset_p2 + input_pixel_stride : in_offset_p2;
    indirection += (uintptr_t) (4 * input_stride);
    in_pixel_offset += 4 * input_pixel_stride;
    const uintptr_t out_pixel0 = (uintptr_t) out_base;
    out_base += 4 * output_pixel_stride;

    for (size_t c = 0; c < channels; c += 2 * channel_tile) {
      const svbool_t pred_c0 = svwhilelt_b32_u64(c, channels);
      const svbool_t pred_c1 = svwhilelt_b32_u64(c + channel_tile, channels);
      const bool has_c1 = c + channel_tile < channels;
      const size_t ch_byte_offset0 = c * sizeof(float);
      const size_t ch_byte_offset1 = has_c1 ? (c + channel_tile) * sizeof(float) : ch_byte_offset0;
      const float* w_c0 = weights + c * 10;
      const float* w_c1 = has_c1 ? w_c0 + channel_tile * 10 : w_c0;
      const float* zero_c0 = zero + c;
      const float* zero_c1 = has_c1 ? zero + c + channel_tile : zero_c0;

      svfloat32_t vacc_p0_c0 = svld1_f32(pred_c0, w_c0);
      svfloat32_t vacc_p0_c1 = svld1_f32(pred_c1, w_c1);
      svfloat32_t vacc_p1_c0 = vacc_p0_c0, vacc_p1_c1 = vacc_p0_c1;
      svfloat32_t vacc_p2_c0 = vacc_p0_c0, vacc_p2_c1 = vacc_p0_c1;
      svfloat32_t vacc_p3_c0 = vacc_p0_c0, vacc_p3_c1 = vacc_p0_c1;

      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t tap = 0; tap < 9; ++tap) {
        const size_t tap_weight_offset = (1 + tap) * channel_tile;
        const svfloat32_t vk_c0 = svld1_f32(pred_c0, w_c0 + tap_weight_offset);
        const svfloat32_t vk_c1 = svld1_f32(pred_c1, w_c1 + tap_weight_offset);
        const float* i_p0 = indirection_p0[tap];
        const float* i_p1 = indirection_p1[tap];
        const float* i_p2 = indirection_p2[tap];
        const float* i_p3 = indirection_p3[tap];
        vacc_p0_c0 = svmla_f32_x(pred_c0, vacc_p0_c0, svld1_f32(pred_c0, i_p0 == zero ? zero_c0 : (const float*) ((uintptr_t) i_p0 + in_offset_p0 + ch_byte_offset0)), vk_c0);
        vacc_p0_c1 = svmla_f32_x(pred_c1, vacc_p0_c1, svld1_f32(pred_c1, i_p0 == zero ? zero_c1 : (const float*) ((uintptr_t) i_p0 + in_offset_p0 + ch_byte_offset1)), vk_c1);
        vacc_p1_c0 = svmla_f32_x(pred_c0, vacc_p1_c0, svld1_f32(pred_c0, i_p1 == zero ? zero_c0 : (const float*) ((uintptr_t) i_p1 + in_offset_p1 + ch_byte_offset0)), vk_c0);
        vacc_p1_c1 = svmla_f32_x(pred_c1, vacc_p1_c1, svld1_f32(pred_c1, i_p1 == zero ? zero_c1 : (const float*) ((uintptr_t) i_p1 + in_offset_p1 + ch_byte_offset1)), vk_c1);
        vacc_p2_c0 = svmla_f32_x(pred_c0, vacc_p2_c0, svld1_f32(pred_c0, i_p2 == zero ? zero_c0 : (const float*) ((uintptr_t) i_p2 + in_offset_p2 + ch_byte_offset0)), vk_c0);
        vacc_p2_c1 = svmla_f32_x(pred_c1, vacc_p2_c1, svld1_f32(pred_c1, i_p2 == zero ? zero_c1 : (const float*) ((uintptr_t) i_p2 + in_offset_p2 + ch_byte_offset1)), vk_c1);
        vacc_p3_c0 = svmla_f32_x(pred_c0, vacc_p3_c0, svld1_f32(pred_c0, i_p3 == zero ? zero_c0 : (const float*) ((uintptr_t) i_p3 + in_offset_p3 + ch_byte_offset0)), vk_c0);
        vacc_p3_c1 = svmla_f32_x(pred_c1, vacc_p3_c1, svld1_f32(pred_c1, i_p3 == zero ? zero_c1 : (const float*) ((uintptr_t) i_p3 + in_offset_p3 + ch_byte_offset1)), vk_c1);
      }

      vacc_p0_c0 = svminnm_f32_x(pred_c0, svmaxnm_f32_x(pred_c0, vacc_p0_c0, vmin), vmax);
      vacc_p0_c1 = svminnm_f32_x(pred_c1, svmaxnm_f32_x(pred_c1, vacc_p0_c1, vmin), vmax);
      svst1_f32(pred_c0, (float*) (out_pixel0 + ch_byte_offset0), vacc_p0_c0);
      svst1_f32(pred_c1, (float*) (out_pixel0 + ch_byte_offset1), vacc_p0_c1);
      if (rem_pixels > 1) {
        vacc_p1_c0 = svminnm_f32_x(pred_c0, svmaxnm_f32_x(pred_c0, vacc_p1_c0, vmin), vmax);
        vacc_p1_c1 = svminnm_f32_x(pred_c1, svmaxnm_f32_x(pred_c1, vacc_p1_c1, vmin), vmax);
        svst1_f32(pred_c0, (float*) (out_pixel0 + output_pixel_stride + ch_byte_offset0), vacc_p1_c0);
        svst1_f32(pred_c1, (float*) (out_pixel0 + output_pixel_stride + ch_byte_offset1), vacc_p1_c1);
        if (rem_pixels > 2) {
          vacc_p2_c0 = svminnm_f32_x(pred_c0, svmaxnm_f32_x(pred_c0, vacc_p2_c0, vmin), vmax);
          vacc_p2_c1 = svminnm_f32_x(pred_c1, svmaxnm_f32_x(pred_c1, vacc_p2_c1, vmin), vmax);
          svst1_f32(pred_c0, (float*) (out_pixel0 + 2 * output_pixel_stride + ch_byte_offset0), vacc_p2_c0);
          svst1_f32(pred_c1, (float*) (out_pixel0 + 2 * output_pixel_stride + ch_byte_offset1), vacc_p2_c1);
          if (rem_pixels > 3) {
            vacc_p3_c0 = svminnm_f32_x(pred_c0, svmaxnm_f32_x(pred_c0, vacc_p3_c0, vmin), vmax);
            vacc_p3_c1 = svminnm_f32_x(pred_c1, svmaxnm_f32_x(pred_c1, vacc_p3_c1, vmin), vmax);
            svst1_f32(pred_c0, (float*) (out_pixel0 + 3 * output_pixel_stride + ch_byte_offset0), vacc_p3_c0);
            svst1_f32(pred_c1, (float*) (out_pixel0 + 3 * output_pixel_stride + ch_byte_offset1), vacc_p3_c1);
          }
        }
      }
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_9p1vc__sme(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  if XNN_UNLIKELY(channels == 0 || output_width == 0) {
    return;
  }

  if (channels <= 2 * svcntsw() || output_width >= 32) {
    f32_dwconv_9p1vc_sme_channel_outer(
        channels, output_width, input, weights, output, input_stride,
        output_increment, input_offset, input_pixel_stride, zero,
        params->scalar.min, params->scalar.max);
  } else {
    f32_dwconv_9p1vc_sme_width_outer(
        channels, output_width, input, weights, output, input_stride,
        output_increment, input_offset, input_pixel_stride, zero,
        params->scalar.min, params->scalar.max);
  }
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
