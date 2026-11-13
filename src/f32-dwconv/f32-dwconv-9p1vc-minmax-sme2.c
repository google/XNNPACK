// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"

size_t xnn_f32_dwconv_minmax_ukernel_9p1vc__sme2_get_channel_tile(void) {
  return svcntsw();
}

__arm_locally_streaming __arm_new("za")
static void f32_dwconv_9p1vc_sme2_channel_outer(
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
    const svfloat32_t vbias = svld1_f32(channel_pred, w);
    svwrite_za32_f32_vg1x4(0, svcreate4_f32(vbias, vbias, vbias, vbias));
    const svfloat32x4_t vbias_x4 = svread_za32_f32_vg1x4(0);
    const float* zero_ch = zero + c;
    uintptr_t indirection = (uintptr_t) input;
    const size_t in_ch_offset = input_offset + c * sizeof(float);
    uint8_t* out_ptr = (uint8_t*) (output + c);

    size_t out_x = 0;
    if (input_pixel_stride == 0 && output_width >= 16) {
      const svfloat32_t vk0 = svld1_f32(channel_pred, w + channel_tile);
      const svfloat32_t vk1 = svld1_f32(channel_pred, w + 2 * channel_tile);
      const svfloat32_t vk2 = svld1_f32(channel_pred, w + 3 * channel_tile);
      const svfloat32_t vk3 = svld1_f32(channel_pred, w + 4 * channel_tile);
      const svfloat32_t vk4 = svld1_f32(channel_pred, w + 5 * channel_tile);
      const svfloat32_t vk5 = svld1_f32(channel_pred, w + 6 * channel_tile);
      const svfloat32_t vk6 = svld1_f32(channel_pred, w + 7 * channel_tile);
      const svfloat32_t vk7 = svld1_f32(channel_pred, w + 8 * channel_tile);
      const svfloat32_t vk8 = svld1_f32(channel_pred, w + 9 * channel_tile);
      for (; out_x + 16 <= output_width; out_x += 16) {
        svwrite_za32_f32_vg1x4(0, vbias_x4);
        svwrite_za32_f32_vg1x4(1, vbias_x4);
        svwrite_za32_f32_vg1x4(2, vbias_x4);
        svwrite_za32_f32_vg1x4(3, vbias_x4);
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[0];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk0);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[0];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk0);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[0];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk0);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[0]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[0];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk0);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[1];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk1);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[1];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk1);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[1];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk1);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[1]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[1];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk1);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[2];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk2);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[2];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk2);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[2];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk2);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[2]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[2];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk2);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[3];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk3);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[3];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk3);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[3];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk3);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[3]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[3];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk3);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[4];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk4);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[4];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk4);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[4];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk4);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[4]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[4];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk4);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[5];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk5);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[5];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk5);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[5];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk5);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[5]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[5];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk5);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[6];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk6);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[6];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk6);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[6];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk6);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[6]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[6];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk6);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[7];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk7);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[7];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk7);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[7];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk7);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[7]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[7];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk7);
        }
        {
          uintptr_t group_indirection = indirection;
          const float* i_p0 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[8];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk8);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (4 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[8];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(1, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk8);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (8 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[8];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(2, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk8);
        }
        {
          uintptr_t group_indirection = indirection + (uintptr_t) (12 * input_stride);
          const float* i_p0 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p1 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p2 = ((const float**) group_indirection)[8]; group_indirection += (uintptr_t) input_stride;
          const float* i_p3 = ((const float**) group_indirection)[8];
          const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_ch_offset));
          const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_ch_offset));
          const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_ch_offset));
          const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_ch_offset));
          svmla_single_za32_f32_vg1x4(3, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk8);
        }
        indirection += (uintptr_t) (16 * input_stride);
        for (uint32_t za_group = 0; za_group < 4; za_group++) {
          const svfloat32x4_t vacc = svclamp_single_f32_x4(svread_za32_f32_vg1x4(za_group), vmin, vmax);
          svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 0)); out_ptr += output_pixel_stride;
          svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 1)); out_ptr += output_pixel_stride;
          svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 2)); out_ptr += output_pixel_stride;
          svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 3)); out_ptr += output_pixel_stride;
        }
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

      svwrite_za32_f32_vg1x4(0, vbias_x4);
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t tap = 0; tap < 9; ++tap) {
        const svfloat32_t vk = svld1_f32(channel_pred, w + (1 + tap) * channel_tile);
        const float* i_p0 = indirection_p0[tap];
        const float* i_p1 = indirection_p1[tap];
        const float* i_p2 = indirection_p2[tap];
        const float* i_p3 = indirection_p3[tap];
        const svfloat32_t vi_p0 = svld1_f32(channel_pred, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_offset_p0));
        const svfloat32_t vi_p1 = svld1_f32(channel_pred, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_offset_p1));
        const svfloat32_t vi_p2 = svld1_f32(channel_pred, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_offset_p2));
        const svfloat32_t vi_p3 = svld1_f32(channel_pred, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_offset_p3));
        svmla_single_za32_f32_vg1x4(0, svcreate4_f32(vi_p0, vi_p1, vi_p2, vi_p3), vk);
      }
      const svfloat32x4_t vacc = svclamp_single_f32_x4(svread_za32_f32_vg1x4(0), vmin, vmax);
      svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 0));
      if (rem_pixels > 1) {
        out_ptr += output_pixel_stride;
        svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 1));
        if (rem_pixels > 2) {
          out_ptr += output_pixel_stride;
          svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 2));
          if (rem_pixels > 3) {
            out_ptr += output_pixel_stride;
            svst1_f32(channel_pred, (float*) out_ptr, svget4_f32(vacc, 3));
          }
        }
      }
      out_ptr += output_pixel_stride;
    }
  }
}

__arm_locally_streaming __arm_new("za")
static void f32_dwconv_9p1vc_sme2_width_outer(
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
  size_t in_pixel_offset = input_offset;
  uintptr_t out_base = (uintptr_t) output;

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

    const uintptr_t out_pixel0 = out_base;
    out_base += 4 * output_pixel_stride;

    uintptr_t weights_ptr = (uintptr_t) weights;
    for (size_t c = 0; c < channels; c += 4 * channel_tile) {
      const svcount_t channel_count_x4 = svwhilelt_c32_u64(c, channels, 4);
      const svbool_t pred_c0 = svwhilelt_b32_u64(c, channels);
      const svbool_t pred_c1 = svwhilelt_b32_u64(c + channel_tile, channels);
      const svbool_t pred_c2 = svwhilelt_b32_u64(c + 2 * channel_tile, channels);
      const svbool_t pred_c3 = svwhilelt_b32_u64(c + 3 * channel_tile, channels);
      const uintptr_t weight_tile_bytes = 10 * channel_tile * sizeof(float);
      const float* w_c0 = (const float*) weights_ptr;
      const float* w_c1 = c + channel_tile < channels ? (const float*) (weights_ptr + weight_tile_bytes) : w_c0;
      const float* w_c2 = c + 2 * channel_tile < channels ? (const float*) (weights_ptr + 2 * weight_tile_bytes) : w_c0;
      const float* w_c3 = c + 3 * channel_tile < channels ? (const float*) (weights_ptr + 3 * weight_tile_bytes) : w_c0;
      weights_ptr += 4 * weight_tile_bytes;

      const float* zero_ch = zero + c;
      const size_t ch_byte_offset = c * sizeof(float);
      const size_t in_byte_offset_p0 = in_offset_p0 + ch_byte_offset;
      const size_t in_byte_offset_p1 = in_offset_p1 + ch_byte_offset;
      const size_t in_byte_offset_p2 = in_offset_p2 + ch_byte_offset;
      const size_t in_byte_offset_p3 = in_offset_p3 + ch_byte_offset;

      const svfloat32x4_t vbias_x4 = svcreate4_f32(
          svld1_f32(pred_c0, w_c0), svld1_f32(pred_c1, w_c1),
          svld1_f32(pred_c2, w_c2), svld1_f32(pred_c3, w_c3));
      svwrite_za32_f32_vg1x4(0, vbias_x4);
      svwrite_za32_f32_vg1x4(1, vbias_x4);
      svwrite_za32_f32_vg1x4(2, vbias_x4);
      svwrite_za32_f32_vg1x4(3, vbias_x4);
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t tap = 0; tap < 9; ++tap) {
        const size_t tap_weight_offset = (1 + tap) * channel_tile;
        const svfloat32x4_t vk_x4 = svcreate4_f32(
            svld1_f32(pred_c0, w_c0 + tap_weight_offset), svld1_f32(pred_c1, w_c1 + tap_weight_offset),
            svld1_f32(pred_c2, w_c2 + tap_weight_offset), svld1_f32(pred_c3, w_c3 + tap_weight_offset));
        const float* i_p0 = indirection_p0[tap];
        const float* i_p1 = indirection_p1[tap];
        const float* i_p2 = indirection_p2[tap];
        const float* i_p3 = indirection_p3[tap];
        svmla_za32_f32_vg1x4(0, svld1_f32_x4(channel_count_x4, i_p0 == zero ? zero_ch : (const float*) ((uintptr_t) i_p0 + in_byte_offset_p0)), vk_x4);
        svmla_za32_f32_vg1x4(1, svld1_f32_x4(channel_count_x4, i_p1 == zero ? zero_ch : (const float*) ((uintptr_t) i_p1 + in_byte_offset_p1)), vk_x4);
        svmla_za32_f32_vg1x4(2, svld1_f32_x4(channel_count_x4, i_p2 == zero ? zero_ch : (const float*) ((uintptr_t) i_p2 + in_byte_offset_p2)), vk_x4);
        svmla_za32_f32_vg1x4(3, svld1_f32_x4(channel_count_x4, i_p3 == zero ? zero_ch : (const float*) ((uintptr_t) i_p3 + in_byte_offset_p3)), vk_x4);
      }

      const svfloat32x4_t vacc_p0 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(0), vmin, vmax);
      svst1_f32_x4(channel_count_x4, (float*) (out_pixel0 + ch_byte_offset), vacc_p0);
      if (rem_pixels > 1) {
        const svfloat32x4_t vacc_p1 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(1), vmin, vmax);
        svst1_f32_x4(channel_count_x4, (float*) (out_pixel0 + output_pixel_stride + ch_byte_offset), vacc_p1);
        if (rem_pixels > 2) {
          const svfloat32x4_t vacc_p2 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(2), vmin, vmax);
          svst1_f32_x4(channel_count_x4, (float*) (out_pixel0 + 2 * output_pixel_stride + ch_byte_offset), vacc_p2);
          if (rem_pixels > 3) {
            const svfloat32x4_t vacc_p3 = svclamp_single_f32_x4(svread_za32_f32_vg1x4(3), vmin, vmax);
            svst1_f32_x4(channel_count_x4, (float*) (out_pixel0 + 3 * output_pixel_stride + ch_byte_offset), vacc_p3);
          }
        }
      }
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_9p1vc__sme2(
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
    f32_dwconv_9p1vc_sme2_channel_outer(
        channels, output_width, input, weights, output, input_stride,
        output_increment, input_offset, input_pixel_stride, zero,
        params->scalar.min, params->scalar.max);
  } else {
    f32_dwconv_9p1vc_sme2_width_outer(
        channels, output_width, input, weights, output, input_stride,
        output_increment, input_offset, input_pixel_stride, zero,
        params->scalar.min, params->scalar.max);
  }
}

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
