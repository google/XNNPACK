// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>

#include "src/xnnpack/conv.h"
#include "src/xnnpack/math.h"


void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x2v__rvv_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  size_t vlmax = __riscv_vsetvlmax_e32m2();

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * vlmax - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const float voutput_max = params->scalar.max;
  const float voutput_min = params->scalar.min;
  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0 = output0;
    float* o1 = output1;

    do {
      size_t vl = __riscv_vsetvl_e32m2(c);

      // Left edge padding
      float vi00c0 = 0.0f;
      float vi00c1 = 0.0f;
      float vi00c2 = 0.0f;
      float vi10c0 = 0.0f;
      float vi10c1 = 0.0f;
      float vi10c2 = 0.0f;
      float vi20c0 = 0.0f;
      float vi20c1 = 0.0f;
      float vi20c2 = 0.0f;
      float vi30c0 = 0.0f;
      float vi30c1 = 0.0f;
      float vi30c2 = 0.0f;
      float vi40c0 = 0.0f;
      float vi40c1 = 0.0f;
      float vi40c2 = 0.0f;

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        const float* w2 = w;
        vfloat32m2_t vo0x0 = __riscv_vle32_v_f32m2(w2, vl);
        vfloat32m2_t vo1x0 = vo0x0;
        vfloat32m2_t vo0x1 = vo0x0;
        vfloat32m2_t vo1x1 = vo0x0;
        w2 += vlmax;

        const float vi02c0 = i0[3];
        const float vi12c0 = i1[3];
        const float vi22c0 = i2[3];
        const float vi32c0 = i3[3];
        const float vi42c0 = i4[3];

        vfloat32m2_t vk00c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi00c0, vk00c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi20c0, vk00c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi02c0, vk00c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi22c0, vk00c0, vl);

        vfloat32m2_t vk10c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi10c0, vk10c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi30c0, vk10c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi12c0, vk10c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi32c0, vk10c0, vl);

        vfloat32m2_t vk20c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi20c0, vk20c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi40c0, vk20c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi22c0, vk20c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi42c0, vk20c0, vl);

        const float vi02c1 = i0[4];
        const float vi12c1 = i1[4];
        const float vi22c1 = i2[4];
        const float vi32c1 = i3[4];
        const float vi42c1 = i4[4];

        vfloat32m2_t vk00c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi00c1, vk00c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi20c1, vk00c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi02c1, vk00c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi22c1, vk00c1, vl);

        vfloat32m2_t vk10c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi10c1, vk10c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi30c1, vk10c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi12c1, vk10c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi32c1, vk10c1, vl);

        vfloat32m2_t vk20c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi20c1, vk20c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi40c1, vk20c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi22c1, vk20c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi42c1, vk20c1, vl);

        const float vi02c2 = i0[5];
        const float vi12c2 = i1[5];
        const float vi22c2 = i2[5];
        const float vi32c2 = i3[5];
        const float vi42c2 = i4[5];

        vfloat32m2_t vk00c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi00c2, vk00c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi20c2, vk00c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi02c2, vk00c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi22c2, vk00c2, vl);

        vfloat32m2_t vk10c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi10c2, vk10c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi30c2, vk10c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi12c2, vk10c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi32c2, vk10c2, vl);

        vfloat32m2_t vk20c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi20c2, vk20c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi40c2, vk20c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi22c2, vk20c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi42c2, vk20c2, vl);

        const float vi01c0 = i0[0];
        const float vi11c0 = i1[0];
        const float vi21c0 = i2[0];
        const float vi31c0 = i3[0];
        const float vi41c0 = i4[0];

        const float vi03c0 = i0[6];
        const float vi13c0 = i1[6];
        const float vi23c0 = i2[6];
        const float vi33c0 = i3[6];
        const float vi43c0 = i4[6];

        vfloat32m2_t vk01c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi01c0, vk01c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi21c0, vk01c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi03c0, vk01c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi23c0, vk01c0, vl);

        vfloat32m2_t vk11c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi11c0, vk11c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi31c0, vk11c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi13c0, vk11c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi33c0, vk11c0, vl);

        vfloat32m2_t vk21c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi21c0, vk21c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi41c0, vk21c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi23c0, vk21c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi43c0, vk21c0, vl);

        const float vi01c1 = i0[1];
        const float vi11c1 = i1[1];
        const float vi21c1 = i2[1];
        const float vi31c1 = i3[1];
        const float vi41c1 = i4[1];

        const float vi03c1 = i0[7];
        const float vi13c1 = i1[7];
        const float vi23c1 = i2[7];
        const float vi33c1 = i3[7];
        const float vi43c1 = i4[7];

        vfloat32m2_t vk01c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi01c1, vk01c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi21c1, vk01c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi03c1, vk01c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi23c1, vk01c1, vl);

        vfloat32m2_t vk11c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi11c1, vk11c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi31c1, vk11c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi13c1, vk11c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi33c1, vk11c1, vl);

        vfloat32m2_t vk21c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi21c1, vk21c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi41c1, vk21c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi23c1, vk21c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi43c1, vk21c1, vl);

        const float vi01c2 = i0[2];
        const float vi11c2 = i1[2];
        const float vi21c2 = i2[2];
        const float vi31c2 = i3[2];
        const float vi41c2 = i4[2];

        const float vi03c2 = i0[8];
        const float vi13c2 = i1[8];
        const float vi23c2 = i2[8];
        const float vi33c2 = i3[8];
        const float vi43c2 = i4[8];

        vfloat32m2_t vk01c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi01c2, vk01c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi21c2, vk01c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi03c2, vk01c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi23c2, vk01c2, vl);

        vfloat32m2_t vk11c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi11c2, vk11c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi31c2, vk11c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi13c2, vk11c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi33c2, vk11c2, vl);

        vfloat32m2_t vk21c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi21c2, vk21c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi41c2, vk21c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi23c2, vk21c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi43c2, vk21c2, vl);

        const float vi04c0 = i0[9];
        const float vi14c0 = i1[9];
        const float vi24c0 = i2[9];
        const float vi34c0 = i3[9];
        const float vi44c0 = i4[9];

        vfloat32m2_t vk02c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi02c0, vk02c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi22c0, vk02c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi04c0, vk02c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi24c0, vk02c0, vl);

        vfloat32m2_t vk12c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi12c0, vk12c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi32c0, vk12c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi14c0, vk12c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi34c0, vk12c0, vl);

        vfloat32m2_t vk22c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi22c0, vk22c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi42c0, vk22c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi24c0, vk22c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi44c0, vk22c0, vl);

        vi00c0 = vi04c0;
        vi10c0 = vi14c0;
        vi20c0 = vi24c0;
        vi30c0 = vi34c0;
        vi40c0 = vi44c0;

        const float vi04c1 = i0[10];
        const float vi14c1 = i1[10];
        const float vi24c1 = i2[10];
        const float vi34c1 = i3[10];
        const float vi44c1 = i4[10];

        vfloat32m2_t vk02c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi02c1, vk02c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi22c1, vk02c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi04c1, vk02c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi24c1, vk02c1, vl);

        vfloat32m2_t vk12c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi12c1, vk12c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi32c1, vk12c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi14c1, vk12c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi34c1, vk12c1, vl);

        vfloat32m2_t vk22c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi22c1, vk22c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi42c1, vk22c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi24c1, vk22c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi44c1, vk22c1, vl);

        vi00c1 = vi04c1;
        vi10c1 = vi14c1;
        vi20c1 = vi24c1;
        vi30c1 = vi34c1;
        vi40c1 = vi44c1;

        const float vi04c2 = i0[11];
        const float vi14c2 = i1[11];
        const float vi24c2 = i2[11];
        const float vi34c2 = i3[11];
        const float vi44c2 = i4[11];

        vfloat32m2_t vk02c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi02c2, vk02c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi22c2, vk02c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi04c2, vk02c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi24c2, vk02c2, vl);

        vfloat32m2_t vk12c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi12c2, vk12c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi32c2, vk12c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi14c2, vk12c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi34c2, vk12c2, vl);

        vfloat32m2_t vk22c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi22c2, vk22c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi42c2, vk22c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi24c2, vk22c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi44c2, vk22c2, vl);

        vi00c2 = vi04c2;
        vi10c2 = vi14c2;
        vi20c2 = vi24c2;
        vi30c2 = vi34c2;
        vi40c2 = vi44c2;

        vo0x0 = __riscv_vfmin_vf_f32m2(vo0x0, voutput_max, vl);
        vo1x0 = __riscv_vfmin_vf_f32m2(vo1x0, voutput_max, vl);
        vo0x1 = __riscv_vfmin_vf_f32m2(vo0x1, voutput_max, vl);
        vo1x1 = __riscv_vfmin_vf_f32m2(vo1x1, voutput_max, vl);
        vo0x0 = __riscv_vfmax_vf_f32m2(vo0x0, voutput_min, vl);
        vo1x0 = __riscv_vfmax_vf_f32m2(vo1x0, voutput_min, vl);
        vo0x1 = __riscv_vfmax_vf_f32m2(vo0x1, voutput_min, vl);
        vo1x1 = __riscv_vfmax_vf_f32m2(vo1x1, voutput_min, vl);

        __riscv_vsse32_v_f32m2(o1, output_channel_stride, vo1x0, vl);
        __riscv_vsse32_v_f32m2(o0, output_channel_stride, vo0x0, vl);
        o0++;
        o1++;

        __riscv_vsse32_v_f32m2(o1, output_channel_stride, vo1x1, vl);
        __riscv_vsse32_v_f32m2(o0, output_channel_stride, vo0x1, vl);
        o0++;
        o1++;

        i0 += 12;
        i1 += 12;
        i2 += 12;
        i3 += 12;
        i4 += 12;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        const float* w2 = w;
        vfloat32m2_t vo0x0 = __riscv_vle32_v_f32m2(w2, vl);
        vfloat32m2_t vo1x0 = vo0x0;
        vfloat32m2_t vo0x1 = vo0x0;
        vfloat32m2_t vo1x1 = vo0x0;
        w2 += vlmax;

        float vi02c0 = 0.0f;
        float vi12c0 = 0.0f;
        float vi22c0 = 0.0f;
        float vi32c0 = 0.0f;
        float vi42c0 = 0.0f;
        if (iw >= 2) {
          vi02c0 = i0[3];
          vi12c0 = i1[3];
          vi22c0 = i2[3];
          vi32c0 = i3[3];
          vi42c0 = i4[3];
        }

        vfloat32m2_t vk00c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi00c0, vk00c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi20c0, vk00c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi02c0, vk00c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi22c0, vk00c0, vl);

        vfloat32m2_t vk10c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi10c0, vk10c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi30c0, vk10c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi12c0, vk10c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi32c0, vk10c0, vl);

        vfloat32m2_t vk20c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi20c0, vk20c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi40c0, vk20c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi22c0, vk20c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi42c0, vk20c0, vl);

        float vi02c1 = 0.0f;
        float vi12c1 = 0.0f;
        float vi22c1 = 0.0f;
        float vi32c1 = 0.0f;
        float vi42c1 = 0.0f;
        if (iw >= 2) {
          vi02c1 = i0[4];
          vi12c1 = i1[4];
          vi22c1 = i2[4];
          vi32c1 = i3[4];
          vi42c1 = i4[4];
        }

        vfloat32m2_t vk00c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi00c1, vk00c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi20c1, vk00c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi02c1, vk00c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi22c1, vk00c1, vl);

        vfloat32m2_t vk10c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi10c1, vk10c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi30c1, vk10c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi12c1, vk10c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi32c1, vk10c1, vl);

        vfloat32m2_t vk20c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi20c1, vk20c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi40c1, vk20c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi22c1, vk20c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi42c1, vk20c1, vl);

        float vi02c2 = 0.0f;
        float vi12c2 = 0.0f;
        float vi22c2 = 0.0f;
        float vi32c2 = 0.0f;
        float vi42c2 = 0.0f;
        if (iw >= 2) {
          vi02c2 = i0[5];
          vi12c2 = i1[5];
          vi22c2 = i2[5];
          vi32c2 = i3[5];
          vi42c2 = i4[5];
        }

        vfloat32m2_t vk00c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi00c2, vk00c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi20c2, vk00c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi02c2, vk00c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi22c2, vk00c2, vl);

        vfloat32m2_t vk10c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi10c2, vk10c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi30c2, vk10c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi12c2, vk10c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi32c2, vk10c2, vl);

        vfloat32m2_t vk20c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi20c2, vk20c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi40c2, vk20c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi22c2, vk20c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi42c2, vk20c2, vl);

        const float vi01c0 = i0[0];
        const float vi11c0 = i1[0];
        const float vi21c0 = i2[0];
        const float vi31c0 = i3[0];
        const float vi41c0 = i4[0];

        float vi03c0 = 0.0f;
        float vi13c0 = 0.0f;
        float vi23c0 = 0.0f;
        float vi33c0 = 0.0f;
        float vi43c0 = 0.0f;
        if (iw > 2) {
          vi03c0 = i0[6];
          vi13c0 = i1[6];
          vi23c0 = i2[6];
          vi33c0 = i3[6];
          vi43c0 = i4[6];
        }

        vfloat32m2_t vk01c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi01c0, vk01c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi21c0, vk01c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi03c0, vk01c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi23c0, vk01c0, vl);

        vfloat32m2_t vk11c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi11c0, vk11c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi31c0, vk11c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi13c0, vk11c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi33c0, vk11c0, vl);

        vfloat32m2_t vk21c0 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi21c0, vk21c0, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi41c0, vk21c0, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi23c0, vk21c0, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi43c0, vk21c0, vl);

        const float vi01c1 = i0[1];
        const float vi11c1 = i1[1];
        const float vi21c1 = i2[1];
        const float vi31c1 = i3[1];
        const float vi41c1 = i4[1];

        float vi03c1 = 0.0f;
        float vi13c1 = 0.0f;
        float vi23c1 = 0.0f;
        float vi33c1 = 0.0f;
        float vi43c1 = 0.0f;
        if (iw > 2) {
          vi03c1 = i0[7];
          vi13c1 = i1[7];
          vi23c1 = i2[7];
          vi33c1 = i3[7];
          vi43c1 = i4[7];
        }

        vfloat32m2_t vk01c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi01c1, vk01c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi21c1, vk01c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi03c1, vk01c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi23c1, vk01c1, vl);

        vfloat32m2_t vk11c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi11c1, vk11c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi31c1, vk11c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi13c1, vk11c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi33c1, vk11c1, vl);

        vfloat32m2_t vk21c1 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi21c1, vk21c1, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi41c1, vk21c1, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi23c1, vk21c1, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi43c1, vk21c1, vl);

        const float vi01c2 = i0[2];
        const float vi11c2 = i1[2];
        const float vi21c2 = i2[2];
        const float vi31c2 = i3[2];
        const float vi41c2 = i4[2];

        float vi03c2 = 0.0f;
        float vi13c2 = 0.0f;
        float vi23c2 = 0.0f;
        float vi33c2 = 0.0f;
        float vi43c2 = 0.0f;
        if (iw > 2) {
          vi03c2 = i0[8];
          vi13c2 = i1[8];
          vi23c2 = i2[8];
          vi33c2 = i3[8];
          vi43c2 = i4[8];
        }

        vfloat32m2_t vk01c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi01c2, vk01c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi21c2, vk01c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi03c2, vk01c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi23c2, vk01c2, vl);

        vfloat32m2_t vk11c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi11c2, vk11c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi31c2, vk11c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi13c2, vk11c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi33c2, vk11c2, vl);

        vfloat32m2_t vk21c2 = __riscv_vle32_v_f32m2(w2, vl);
        w2 += vlmax;
        vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi21c2, vk21c2, vl);
        vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi41c2, vk21c2, vl);
        vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi23c2, vk21c2, vl);
        vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi43c2, vk21c2, vl);

        if (iw >= 2) {
          const float vi04c0 = 0.0f;
          const float vi14c0 = 0.0f;
          const float vi24c0 = 0.0f;
          const float vi34c0 = 0.0f;
          const float vi44c0 = 0.0f;

          vfloat32m2_t vk02c0 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi02c0, vk02c0, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi22c0, vk02c0, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi04c0, vk02c0, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi24c0, vk02c0, vl);

          vfloat32m2_t vk12c0 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi12c0, vk12c0, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi32c0, vk12c0, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi14c0, vk12c0, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi34c0, vk12c0, vl);

          vfloat32m2_t vk22c0 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi22c0, vk22c0, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi42c0, vk22c0, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi24c0, vk22c0, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi44c0, vk22c0, vl);

          const float vi04c1 = 0.0f;
          const float vi14c1 = 0.0f;
          const float vi24c1 = 0.0f;
          const float vi34c1 = 0.0f;
          const float vi44c1 = 0.0f;

          vfloat32m2_t vk02c1 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi02c1, vk02c1, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi22c1, vk02c1, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi04c1, vk02c1, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi24c1, vk02c1, vl);

          vfloat32m2_t vk12c1 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi12c1, vk12c1, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi32c1, vk12c1, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi14c1, vk12c1, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi34c1, vk12c1, vl);

          vfloat32m2_t vk22c1 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi22c1, vk22c1, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi42c1, vk22c1, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi24c1, vk22c1, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi44c1, vk22c1, vl);

          const float vi04c2 = 0.0f;
          const float vi14c2 = 0.0f;
          const float vi24c2 = 0.0f;
          const float vi34c2 = 0.0f;
          const float vi44c2 = 0.0f;

          vfloat32m2_t vk02c2 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi02c2, vk02c2, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi22c2, vk02c2, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi04c2, vk02c2, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi24c2, vk02c2, vl);

          vfloat32m2_t vk12c2 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi12c2, vk12c2, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi32c2, vk12c2, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi14c2, vk12c2, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi34c2, vk12c2, vl);

          vfloat32m2_t vk22c2 = __riscv_vle32_v_f32m2(w2, vl);
          w2 += vlmax;
          vo0x0 = __riscv_vfmacc_vf_f32m2(vo0x0, vi22c2, vk22c2, vl);
          vo1x0 = __riscv_vfmacc_vf_f32m2(vo1x0, vi42c2, vk22c2, vl);
          vo0x1 = __riscv_vfmacc_vf_f32m2(vo0x1, vi24c2, vk22c2, vl);
          vo1x1 = __riscv_vfmacc_vf_f32m2(vo1x1, vi44c2, vk22c2, vl);
        }

        vo0x0 = __riscv_vfmin_vf_f32m2(vo0x0, voutput_max, vl);
        vo1x0 = __riscv_vfmin_vf_f32m2(vo1x0, voutput_max, vl);
        vo0x1 = __riscv_vfmin_vf_f32m2(vo0x1, voutput_max, vl);
        vo1x1 = __riscv_vfmin_vf_f32m2(vo1x1, voutput_max, vl);
        vo0x0 = __riscv_vfmax_vf_f32m2(vo0x0, voutput_min, vl);
        vo1x0 = __riscv_vfmax_vf_f32m2(vo1x0, voutput_min, vl);
        vo0x1 = __riscv_vfmax_vf_f32m2(vo0x1, voutput_min, vl);
        vo1x1 = __riscv_vfmax_vf_f32m2(vo1x1, voutput_min, vl);

        if (iw == 3)  {
          __riscv_vsse32_v_f32m2(o1, output_channel_stride, vo1x0, vl);
          __riscv_vsse32_v_f32m2(o0, output_channel_stride, vo0x0, vl);
          o0++;
          o1++;

          __riscv_vsse32_v_f32m2(o1, output_channel_stride, vo1x1, vl);
          __riscv_vsse32_v_f32m2(o0, output_channel_stride, vo0x1, vl);
          o0++;
          o1++;
        } else {
          __riscv_vsse32_v_f32m2(o1, output_channel_stride, vo1x0, vl);
          __riscv_vsse32_v_f32m2(o0, output_channel_stride, vo0x0, vl);
          o0++;
          o1++;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0 = (float*) ((uintptr_t) o0 + output_channel_increment);
      o1 = (float*) ((uintptr_t) o1 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const float*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const float*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next vlmax output channels
      w += 28 * vlmax;
      c = doz(c, vlmax);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}
