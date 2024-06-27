// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5s2p2-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_3x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const uint32x4_t vmask_even = vld1q_u32(params->neon_stride2.mask_even);
  const uint32x4_t vmask_odd = vld1q_u32(params->neon_stride2.mask_odd);
  const float32x4_t vmax = vld1q_dup_f32(&params->neon_stride2.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->neon_stride2.min);

  const float32x4_t vw0123 = vld1q_f32(weights);
  const float32x4_t vw4567 = vld1q_f32(weights + 4);
  const float32x4_t vw89AB = vld1q_f32(weights + 8);
  const float32x4_t vwCDEF = vld1q_f32(weights + 12);
  const float32x4_t vwGHIJ = vld1q_f32(weights + 16);
  const float32x4_t vwKLMN = vld1q_f32(weights + 20);
  const float32x2_t vwOP   = vld1_f32(weights + 24);

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);
  float* o2 = (float*) ((uintptr_t) o1 + output_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i6 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i7 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 11) {
      i8 = zero;
    }

    float32x4_t vi0x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi1x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi2x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi3x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi4x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi5x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi6x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi7x0246 = vmovq_n_f32(0.0f);
    float32x4_t vi8x0246 = vmovq_n_f32(0.0f);

    float32x4_t vi0x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi1x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi2x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi3x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi4x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi5x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi6x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi7x1357 = vmovq_n_f32(0.0f);
    float32x4_t vi8x1357 = vmovq_n_f32(0.0f);

    float32x4x2_t vi0x8ACE9BDF = vld2q_f32(i0); i0 += 8;
    float32x4x2_t vi1x8ACE9BDF = vld2q_f32(i1); i1 += 8;
    float32x4x2_t vi2x8ACE9BDF = vld2q_f32(i2); i2 += 8;
    float32x4x2_t vi3x8ACE9BDF = vld2q_f32(i3); i3 += 8;
    float32x4x2_t vi4x8ACE9BDF = vld2q_f32(i4); i4 += 8;
    float32x4x2_t vi5x8ACE9BDF = vld2q_f32(i5); i5 += 8;
    float32x4x2_t vi6x8ACE9BDF = vld2q_f32(i6); i6 += 8;
    float32x4x2_t vi7x8ACE9BDF = vld2q_f32(i7); i7 += 8;
    float32x4x2_t vi8x8ACE9BDF = vld2q_f32(i8); i8 += 8;

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x8ACE9BDF.val[0], vget_high_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x8ACE9BDF.val[0], vget_high_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x8ACE9BDF.val[0], vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x8ACE9BDF.val[0], vget_low_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x8ACE9BDF.val[0], vget_low_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x8ACE9BDF.val[0], vget_low_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x8ACE9BDF.val[0], vget_low_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x8ACE9BDF.val[0], vget_low_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x8ACE9BDF.val[0], vget_low_f32(vwCDEF), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x8ACE9BDF.val[0], vget_high_f32(vwGHIJ), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x8ACE9BDF.val[0], vget_high_f32(vwGHIJ), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x8ACE9BDF.val[0], vget_high_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x8ACE9BDF.val[0], vget_high_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x8ACE9BDF.val[0], vget_high_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x8ACE9BDF.val[0], vget_high_f32(vwKLMN), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x8ACE9BDF.val[1], vget_low_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x8ACE9BDF.val[1], vget_low_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x8ACE9BDF.val[1], vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x8ACE9BDF.val[1], vget_low_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x8ACE9BDF.val[1], vget_low_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x8ACE9BDF.val[1], vget_low_f32(vw89AB), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x8ACE9BDF.val[1], vget_high_f32(vwCDEF), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x8ACE9BDF.val[1], vget_high_f32(vwCDEF), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x8ACE9BDF.val[1], vget_high_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x8ACE9BDF.val[1], vget_high_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x8ACE9BDF.val[1], vget_high_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x8ACE9BDF.val[1], vget_high_f32(vwGHIJ), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x8ACE9BDF.val[1], vwOP, 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x8ACE9BDF.val[1], vwOP, 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x8ACE9BDF.val[1], vwOP, 0);

      const float32x4_t vi0x68AC = vextq_f32(vi0x0246, vi0x8ACE9BDF.val[0], 3);
      vi0x0246 = vi0x8ACE9BDF.val[0];
      const float32x4_t vi1x68AC = vextq_f32(vi1x0246, vi1x8ACE9BDF.val[0], 3);
      vi1x0246 = vi1x8ACE9BDF.val[0];
      const float32x4_t vi2x68AC = vextq_f32(vi2x0246, vi2x8ACE9BDF.val[0], 3);
      vi2x0246 = vi2x8ACE9BDF.val[0];
      const float32x4_t vi3x68AC = vextq_f32(vi3x0246, vi3x8ACE9BDF.val[0], 3);
      vi3x0246 = vi3x8ACE9BDF.val[0];
      const float32x4_t vi4x68AC = vextq_f32(vi4x0246, vi4x8ACE9BDF.val[0], 3);
      vi4x0246 = vi4x8ACE9BDF.val[0];
      const float32x4_t vi5x68AC = vextq_f32(vi5x0246, vi5x8ACE9BDF.val[0], 3);
      vi5x0246 = vi5x8ACE9BDF.val[0];
      const float32x4_t vi6x68AC = vextq_f32(vi6x0246, vi6x8ACE9BDF.val[0], 3);
      vi6x0246 = vi6x8ACE9BDF.val[0];
      const float32x4_t vi7x68AC = vextq_f32(vi7x0246, vi7x8ACE9BDF.val[0], 3);
      vi7x0246 = vi7x8ACE9BDF.val[0];
      const float32x4_t vi8x68AC = vextq_f32(vi8x0246, vi8x8ACE9BDF.val[0], 3);
      vi8x0246 = vi8x8ACE9BDF.val[0];

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x68AC, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x68AC, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x68AC, vget_low_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x68AC, vget_high_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x68AC, vget_high_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x68AC, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x68AC, vget_high_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x68AC, vget_high_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x68AC, vget_high_f32(vw89AB), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x68AC, vget_low_f32(vwGHIJ), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x68AC, vget_low_f32(vwGHIJ), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x68AC, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x68AC, vget_low_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x68AC, vget_low_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x68AC, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x79BD = vextq_f32(vi0x1357, vi0x8ACE9BDF.val[1], 3);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float32x4_t vi1x79BD = vextq_f32(vi1x1357, vi1x8ACE9BDF.val[1], 3);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float32x4_t vi2x79BD = vextq_f32(vi2x1357, vi2x8ACE9BDF.val[1], 3);
      vi2x1357 = vi2x8ACE9BDF.val[1];
      const float32x4_t vi3x79BD = vextq_f32(vi3x1357, vi3x8ACE9BDF.val[1], 3);
      vi3x1357 = vi3x8ACE9BDF.val[1];
      const float32x4_t vi4x79BD = vextq_f32(vi4x1357, vi4x8ACE9BDF.val[1], 3);
      vi4x1357 = vi4x8ACE9BDF.val[1];
      const float32x4_t vi5x79BD = vextq_f32(vi5x1357, vi5x8ACE9BDF.val[1], 3);
      vi5x1357 = vi5x8ACE9BDF.val[1];
      const float32x4_t vi6x79BD = vextq_f32(vi6x1357, vi6x8ACE9BDF.val[1], 3);
      vi6x1357 = vi6x8ACE9BDF.val[1];
      const float32x4_t vi7x79BD = vextq_f32(vi7x1357, vi7x8ACE9BDF.val[1], 3);
      vi7x1357 = vi7x8ACE9BDF.val[1];
      const float32x4_t vi8x79BD = vextq_f32(vi8x1357, vi8x8ACE9BDF.val[1], 3);
      vi8x1357 = vi8x8ACE9BDF.val[1];

      const float32x4x2_t vi0xGIKMHJLN = vld2q_f32(i0); i0 += 8;
      const float32x4x2_t vi1xGIKMHJLN = vld2q_f32(i1); i1 += 8;
      const float32x4x2_t vi2xGIKMHJLN = vld2q_f32(i2); i2 += 8;
      const float32x4x2_t vi3xGIKMHJLN = vld2q_f32(i3); i3 += 8;
      const float32x4x2_t vi4xGIKMHJLN = vld2q_f32(i4); i4 += 8;
      const float32x4x2_t vi5xGIKMHJLN = vld2q_f32(i5); i5 += 8;
      const float32x4x2_t vi6xGIKMHJLN = vld2q_f32(i6); i6 += 8;
      const float32x4x2_t vi7xGIKMHJLN = vld2q_f32(i7); i7 += 8;
      const float32x4x2_t vi8xGIKMHJLN = vld2q_f32(i8); i8 += 8;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x79BD, vget_high_f32(vw0123), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x79BD, vget_high_f32(vw0123), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x79BD, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x79BD, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x79BD, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x79BD, vget_high_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x79BD, vget_low_f32(vwCDEF), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x79BD, vget_low_f32(vwCDEF), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x79BD, vget_low_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x79BD, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x79BD, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x79BD, vget_low_f32(vwGHIJ), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x79BD, vget_high_f32(vwKLMN), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x79BD, vget_high_f32(vwKLMN), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x79BD, vget_high_f32(vwKLMN), 0);

      const float32x4_t vi0xACEG = vextq_f32(vi0x8ACE9BDF.val[0], vi0xGIKMHJLN.val[0], 1);
      vi0x8ACE9BDF = vi0xGIKMHJLN;
      const float32x4_t vi1xACEG = vextq_f32(vi1x8ACE9BDF.val[0], vi1xGIKMHJLN.val[0], 1);
      vi1x8ACE9BDF = vi1xGIKMHJLN;
      const float32x4_t vi2xACEG = vextq_f32(vi2x8ACE9BDF.val[0], vi2xGIKMHJLN.val[0], 1);
      vi2x8ACE9BDF = vi2xGIKMHJLN;
      const float32x4_t vi3xACEG = vextq_f32(vi3x8ACE9BDF.val[0], vi3xGIKMHJLN.val[0], 1);
      vi3x8ACE9BDF = vi3xGIKMHJLN;
      const float32x4_t vi4xACEG = vextq_f32(vi4x8ACE9BDF.val[0], vi4xGIKMHJLN.val[0], 1);
      vi4x8ACE9BDF = vi4xGIKMHJLN;
      const float32x4_t vi5xACEG = vextq_f32(vi5x8ACE9BDF.val[0], vi5xGIKMHJLN.val[0], 1);
      vi5x8ACE9BDF = vi5xGIKMHJLN;
      const float32x4_t vi6xACEG = vextq_f32(vi6x8ACE9BDF.val[0], vi6xGIKMHJLN.val[0], 1);
      vi6x8ACE9BDF = vi6xGIKMHJLN;
      const float32x4_t vi7xACEG = vextq_f32(vi7x8ACE9BDF.val[0], vi7xGIKMHJLN.val[0], 1);
      vi7x8ACE9BDF = vi7xGIKMHJLN;
      const float32x4_t vi8xACEG = vextq_f32(vi8x8ACE9BDF.val[0], vi8xGIKMHJLN.val[0], 1);
      vi8x8ACE9BDF = vi8xGIKMHJLN;

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0xACEG, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2xACEG, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4xACEG, vget_low_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1xACEG, vget_high_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3xACEG, vget_high_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5xACEG, vget_high_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2xACEG, vget_high_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4xACEG, vget_high_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6xACEG, vget_high_f32(vwCDEF), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3xACEG, vget_low_f32(vwKLMN), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5xACEG, vget_low_f32(vwKLMN), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7xACEG, vget_low_f32(vwKLMN), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4xACEG, vwOP, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6xACEG, vwOP, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8xACEG, vwOP, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      vst1q_f32(o2, vo2); o2 += 4;
      vst1q_f32(o1, vo1); o1 += 4;
      vst1q_f32(o0, vo0); o0 += 4;
    }
    // Last block has 1-8 pixels to process.
    assert(w <= 8 * sizeof(float));
    assert(w >= 1 * sizeof(float));
    {
      float32x4_t vo0p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo1p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);
      float32x4_t vo2p0 = vdupq_lane_f32(vget_low_f32(vw0123), 0);

      const float32x4_t vi0x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi0x8ACE9BDF.val[0])));
      const float32x4_t vi1x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi1x8ACE9BDF.val[0])));
      const float32x4_t vi2x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi2x8ACE9BDF.val[0])));
      const float32x4_t vi3x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi3x8ACE9BDF.val[0])));
      const float32x4_t vi4x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi4x8ACE9BDF.val[0])));
      const float32x4_t vi5x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi5x8ACE9BDF.val[0])));
      const float32x4_t vi6x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi6x8ACE9BDF.val[0])));
      const float32x4_t vi7x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi7x8ACE9BDF.val[0])));
      const float32x4_t vi8x8ACE = vreinterpretq_f32_u32(vandq_u32(vmask_even, vreinterpretq_u32_f32(vi8x8ACE9BDF.val[0])));

      const float32x4_t vi0x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi0x8ACE9BDF.val[1])));
      const float32x4_t vi1x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi1x8ACE9BDF.val[1])));
      const float32x4_t vi2x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi2x8ACE9BDF.val[1])));
      const float32x4_t vi3x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi3x8ACE9BDF.val[1])));
      const float32x4_t vi4x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi4x8ACE9BDF.val[1])));
      const float32x4_t vi5x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi5x8ACE9BDF.val[1])));
      const float32x4_t vi6x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi6x8ACE9BDF.val[1])));
      const float32x4_t vi7x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi7x8ACE9BDF.val[1])));
      const float32x4_t vi8x9BDF = vreinterpretq_f32_u32(vandq_u32(vmask_odd, vreinterpretq_u32_f32(vi8x8ACE9BDF.val[1])));

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x8ACE, vget_high_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x8ACE, vget_high_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x8ACE, vget_high_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x8ACE, vget_low_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x8ACE, vget_low_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x8ACE, vget_low_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x8ACE, vget_low_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x8ACE, vget_low_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x8ACE, vget_low_f32(vwCDEF), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x8ACE, vget_high_f32(vwGHIJ), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x8ACE, vget_high_f32(vwGHIJ), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x8ACE, vget_high_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x8ACE, vget_high_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x8ACE, vget_high_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x8ACE, vget_high_f32(vwKLMN), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x9BDF, vget_low_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x9BDF, vget_low_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x9BDF, vget_low_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x9BDF, vget_low_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x9BDF, vget_low_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x9BDF, vget_low_f32(vw89AB), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x9BDF, vget_high_f32(vwCDEF), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x9BDF, vget_high_f32(vwCDEF), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x9BDF, vget_high_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x9BDF, vget_high_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x9BDF, vget_high_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x9BDF, vget_high_f32(vwGHIJ), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x9BDF, vwOP, 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x9BDF, vwOP, 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x9BDF, vwOP, 0);

      const float32x4_t vi0x68AC = vextq_f32(vi0x0246, vi0x8ACE, 3);
      const float32x4_t vi1x68AC = vextq_f32(vi1x0246, vi1x8ACE, 3);
      const float32x4_t vi2x68AC = vextq_f32(vi2x0246, vi2x8ACE, 3);
      const float32x4_t vi3x68AC = vextq_f32(vi3x0246, vi3x8ACE, 3);
      const float32x4_t vi4x68AC = vextq_f32(vi4x0246, vi4x8ACE, 3);
      const float32x4_t vi5x68AC = vextq_f32(vi5x0246, vi5x8ACE, 3);
      const float32x4_t vi6x68AC = vextq_f32(vi6x0246, vi6x8ACE, 3);
      const float32x4_t vi7x68AC = vextq_f32(vi7x0246, vi7x8ACE, 3);
      const float32x4_t vi8x68AC = vextq_f32(vi8x0246, vi8x8ACE, 3);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x68AC, vget_low_f32(vw0123), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x68AC, vget_low_f32(vw0123), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x68AC, vget_low_f32(vw0123), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x68AC, vget_high_f32(vw4567), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x68AC, vget_high_f32(vw4567), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x68AC, vget_high_f32(vw4567), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x68AC, vget_high_f32(vw89AB), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x68AC, vget_high_f32(vw89AB), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x68AC, vget_high_f32(vw89AB), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x68AC, vget_low_f32(vwGHIJ), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x68AC, vget_low_f32(vwGHIJ), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x68AC, vget_low_f32(vwGHIJ), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x68AC, vget_low_f32(vwKLMN), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x68AC, vget_low_f32(vwKLMN), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x68AC, vget_low_f32(vwKLMN), 1);

      const float32x4_t vi0x79BD = vextq_f32(vi0x1357, vi0x9BDF, 3);
      const float32x4_t vi1x79BD = vextq_f32(vi1x1357, vi1x9BDF, 3);
      const float32x4_t vi2x79BD = vextq_f32(vi2x1357, vi2x9BDF, 3);
      const float32x4_t vi3x79BD = vextq_f32(vi3x1357, vi3x9BDF, 3);
      const float32x4_t vi4x79BD = vextq_f32(vi4x1357, vi4x9BDF, 3);
      const float32x4_t vi5x79BD = vextq_f32(vi5x1357, vi5x9BDF, 3);
      const float32x4_t vi6x79BD = vextq_f32(vi6x1357, vi6x9BDF, 3);
      const float32x4_t vi7x79BD = vextq_f32(vi7x1357, vi7x9BDF, 3);
      const float32x4_t vi8x79BD = vextq_f32(vi8x1357, vi8x9BDF, 3);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0x79BD, vget_high_f32(vw0123), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2x79BD, vget_high_f32(vw0123), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4x79BD, vget_high_f32(vw0123), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1x79BD, vget_high_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3x79BD, vget_high_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5x79BD, vget_high_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2x79BD, vget_low_f32(vwCDEF), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4x79BD, vget_low_f32(vwCDEF), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6x79BD, vget_low_f32(vwCDEF), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3x79BD, vget_low_f32(vwGHIJ), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5x79BD, vget_low_f32(vwGHIJ), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7x79BD, vget_low_f32(vwGHIJ), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4x79BD, vget_high_f32(vwKLMN), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6x79BD, vget_high_f32(vwKLMN), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8x79BD, vget_high_f32(vwKLMN), 0);

      const float32x4_t vzero = vmovq_n_f32(0.0f);
      const float32x4_t vi0xACEG = vextq_f32(vi0x8ACE, vzero, 1);
      const float32x4_t vi1xACEG = vextq_f32(vi1x8ACE, vzero, 1);
      const float32x4_t vi2xACEG = vextq_f32(vi2x8ACE, vzero, 1);
      const float32x4_t vi3xACEG = vextq_f32(vi3x8ACE, vzero, 1);
      const float32x4_t vi4xACEG = vextq_f32(vi4x8ACE, vzero, 1);
      const float32x4_t vi5xACEG = vextq_f32(vi5x8ACE, vzero, 1);
      const float32x4_t vi6xACEG = vextq_f32(vi6x8ACE, vzero, 1);
      const float32x4_t vi7xACEG = vextq_f32(vi7x8ACE, vzero, 1);
      const float32x4_t vi8xACEG = vextq_f32(vi8x8ACE, vzero, 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi0xACEG, vget_low_f32(vw4567), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi2xACEG, vget_low_f32(vw4567), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi4xACEG, vget_low_f32(vw4567), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi1xACEG, vget_high_f32(vw89AB), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi3xACEG, vget_high_f32(vw89AB), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi5xACEG, vget_high_f32(vw89AB), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi2xACEG, vget_high_f32(vwCDEF), 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi4xACEG, vget_high_f32(vwCDEF), 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi6xACEG, vget_high_f32(vwCDEF), 1);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi3xACEG, vget_low_f32(vwKLMN), 0);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi5xACEG, vget_low_f32(vwKLMN), 0);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi7xACEG, vget_low_f32(vwKLMN), 0);

      vo0p0 = vmlaq_lane_f32(vo0p0, vi4xACEG, vwOP, 1);
      vo1p0 = vmlaq_lane_f32(vo1p0, vi6xACEG, vwOP, 1);
      vo2p0 = vmlaq_lane_f32(vo2p0, vi8xACEG, vwOP, 1);


      float32x4_t vo0 = vmaxq_f32(vo0p0, vmin);
      float32x4_t vo1 = vmaxq_f32(vo1p0, vmin);
      float32x4_t vo2 = vmaxq_f32(vo2p0, vmin);

      vo0 = vminq_f32(vo0, vmax);
      vo1 = vminq_f32(vo1, vmax);
      vo2 = vminq_f32(vo2, vmax);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        vst1q_f32(o2, vo2); o2 += 4;
        vst1q_f32(o1, vo1); o1 += 4;
        vst1q_f32(o0, vo0); o0 += 4;
      } else {
        float32x2_t vo0_lo = vget_low_f32(vo0);
        float32x2_t vo1_lo = vget_low_f32(vo1);
        float32x2_t vo2_lo = vget_low_f32(vo2);
        if (w_tmp & 2) {
          vst1_f32(o2, vo2_lo); o2 += 2;
          vst1_f32(o1, vo1_lo); o1 += 2;
          vst1_f32(o0, vo0_lo); o0 += 2;

          vo0_lo = vget_high_f32(vo0);
          vo1_lo = vget_high_f32(vo1);
          vo2_lo = vget_high_f32(vo2);
        }
        if (w_tmp & 1) {
          vst1_lane_f32(o2, vo2_lo, 0); o2 += 1;
          vst1_lane_f32(o1, vo1_lo, 0); o1 += 1;
          vst1_lane_f32(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i6 - input_decrement);
    i1 = (const float*) ((uintptr_t) i7 - input_decrement);
    i2 = (const float*) ((uintptr_t) i8 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + output_width);
    o2 = (float*) ((uintptr_t) o1 + output_width);

    output_height = doz(output_height, 3);
    padded_input_height = doz(padded_input_height, 6);
  } while (output_height != 0);
}
