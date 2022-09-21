// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_3x4_acc2(
    size_t input_height,
    size_t input_width,
    const void* input,
    const void* weights,
    const void* zero,
    void* output,
    uint32_t padding_top,
    const union xnn_f16_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(__fp16) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const uint16x4_t vmask_even = vld1_u16(params->neonfp16arith.maskx4_even);
  const uint16x4_t vmask_odd = vld1_u16(params->neonfp16arith.maskx4_odd);
  const float16x4_t vmax = vld1_dup_f16(&params->neonfp16arith.max);
  const float16x4_t vmin = vld1_dup_f16(&params->neonfp16arith.min);

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x8_t vw89ABCDEF = vld1q_f16(w0 + 8);
  const float16x8_t vwGHIJKLMN = vld1q_f16(w0 + 16);
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 24)));

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(__fp16));

  const __fp16* i0 = zero;
  const __fp16* i1 = (const __fp16*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_width);
  const __fp16* i6 = (const __fp16*) ((uintptr_t) i5 + input_width);
  const __fp16* i7 = (const __fp16*) ((uintptr_t) i6 + input_width);
  const __fp16* i8 = (const __fp16*) ((uintptr_t) i7 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(__fp16)) / 2, sizeof(__fp16));

  __fp16* o0 = output;
  __fp16* o1 = (__fp16*) ((uintptr_t) o0 + output_width);
  __fp16* o2 = (__fp16*) ((uintptr_t) o1 + output_width);

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

    float16x4_t vi0x0246 = vmov_n_f16(0.0f);
    float16x4_t vi1x0246 = vmov_n_f16(0.0f);
    float16x4_t vi2x0246 = vmov_n_f16(0.0f);
    float16x4_t vi3x0246 = vmov_n_f16(0.0f);
    float16x4_t vi4x0246 = vmov_n_f16(0.0f);
    float16x4_t vi5x0246 = vmov_n_f16(0.0f);
    float16x4_t vi6x0246 = vmov_n_f16(0.0f);
    float16x4_t vi7x0246 = vmov_n_f16(0.0f);
    float16x4_t vi8x0246 = vmov_n_f16(0.0f);

    float16x4_t vi0x1357 = vmov_n_f16(0.0f);
    float16x4_t vi1x1357 = vmov_n_f16(0.0f);
    float16x4_t vi2x1357 = vmov_n_f16(0.0f);
    float16x4_t vi3x1357 = vmov_n_f16(0.0f);
    float16x4_t vi4x1357 = vmov_n_f16(0.0f);
    float16x4_t vi5x1357 = vmov_n_f16(0.0f);
    float16x4_t vi6x1357 = vmov_n_f16(0.0f);
    float16x4_t vi7x1357 = vmov_n_f16(0.0f);
    float16x4_t vi8x1357 = vmov_n_f16(0.0f);

    float16x4x2_t vi0x8ACE9BDF = vld2_f16(i0); i0 += 8;
    float16x4x2_t vi1x8ACE9BDF = vld2_f16(i1); i1 += 8;
    float16x4x2_t vi2x8ACE9BDF = vld2_f16(i2); i2 += 8;
    float16x4x2_t vi3x8ACE9BDF = vld2_f16(i3); i3 += 8;
    float16x4x2_t vi4x8ACE9BDF = vld2_f16(i4); i4 += 8;
    float16x4x2_t vi5x8ACE9BDF = vld2_f16(i5); i5 += 8;
    float16x4x2_t vi6x8ACE9BDF = vld2_f16(i6); i6 += 8;
    float16x4x2_t vi7x8ACE9BDF = vld2_f16(i7); i7 += 8;
    float16x4x2_t vi8x8ACE9BDF = vld2_f16(i8); i8 += 8;

    size_t w = input_width;
    for (; w > 8 * sizeof(__fp16); w -= 8 * sizeof(__fp16)) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);

      // Center column
      float16x4_t vo0p1 = vmul_laneq_f16(vi0x8ACE9BDF.val[0], vw01234567, 3);
      float16x4_t vo1p1 = vmul_laneq_f16(vi2x8ACE9BDF.val[0], vw01234567, 3);
      float16x4_t vo2p1 = vmul_laneq_f16(vi4x8ACE9BDF.val[0], vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x8ACE9BDF.val[0], vw89ABCDEF, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x8ACE9BDF.val[0], vw89ABCDEF, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x8ACE9BDF.val[0], vw89ABCDEF, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x8ACE9BDF.val[0], vw89ABCDEF, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x8ACE9BDF.val[0], vw89ABCDEF, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x8ACE9BDF.val[0], vw89ABCDEF, 5);

      vo0p1 = vfma_laneq_f16(vo0p1, vi3x8ACE9BDF.val[0], vwGHIJKLMN, 2);
      vo1p1 = vfma_laneq_f16(vo1p1, vi5x8ACE9BDF.val[0], vwGHIJKLMN, 2);
      vo2p1 = vfma_laneq_f16(vo2p1, vi7x8ACE9BDF.val[0], vwGHIJKLMN, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x8ACE9BDF.val[0], vwGHIJKLMN, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi6x8ACE9BDF.val[0], vwGHIJKLMN, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi8x8ACE9BDF.val[0], vwGHIJKLMN, 7);

      // Right by 1 column
      vo0p1 = vfma_laneq_f16(vo0p1, vi0x8ACE9BDF.val[1], vw01234567, 4);
      vo1p1 = vfma_laneq_f16(vo1p1, vi2x8ACE9BDF.val[1], vw01234567, 4);
      vo2p1 = vfma_laneq_f16(vo2p1, vi4x8ACE9BDF.val[1], vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x8ACE9BDF.val[1], vw89ABCDEF, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x8ACE9BDF.val[1], vw89ABCDEF, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x8ACE9BDF.val[1], vw89ABCDEF, 1);

      vo0p1 = vfma_laneq_f16(vo0p1, vi2x8ACE9BDF.val[1], vw89ABCDEF, 6);
      vo1p1 = vfma_laneq_f16(vo1p1, vi4x8ACE9BDF.val[1], vw89ABCDEF, 6);
      vo2p1 = vfma_laneq_f16(vo2p1, vi6x8ACE9BDF.val[1], vw89ABCDEF, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x8ACE9BDF.val[1], vwGHIJKLMN, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x8ACE9BDF.val[1], vwGHIJKLMN, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi7x8ACE9BDF.val[1], vwGHIJKLMN, 3);

      vo0p1 = vfma_lane_f16(vo0p1, vi4x8ACE9BDF.val[1], vwOP, 0);
      vo1p1 = vfma_lane_f16(vo1p1, vi6x8ACE9BDF.val[1], vwOP, 0);
      vo2p1 = vfma_lane_f16(vo2p1, vi8x8ACE9BDF.val[1], vwOP, 0);

      // Left by 2 column
      const float16x4_t vi0x68AC = vext_f16(vi0x0246, vi0x8ACE9BDF.val[0], 3);
      vi0x0246 = vi0x8ACE9BDF.val[0];
      const float16x4_t vi1x68AC = vext_f16(vi1x0246, vi1x8ACE9BDF.val[0], 3);
      vi1x0246 = vi1x8ACE9BDF.val[0];
      const float16x4_t vi2x68AC = vext_f16(vi2x0246, vi2x8ACE9BDF.val[0], 3);
      vi2x0246 = vi2x8ACE9BDF.val[0];
      const float16x4_t vi3x68AC = vext_f16(vi3x0246, vi3x8ACE9BDF.val[0], 3);
      vi3x0246 = vi3x8ACE9BDF.val[0];
      const float16x4_t vi4x68AC = vext_f16(vi4x0246, vi4x8ACE9BDF.val[0], 3);
      vi4x0246 = vi4x8ACE9BDF.val[0];
      const float16x4_t vi5x68AC = vext_f16(vi5x0246, vi5x8ACE9BDF.val[0], 3);
      vi5x0246 = vi5x8ACE9BDF.val[0];
      const float16x4_t vi6x68AC = vext_f16(vi6x0246, vi6x8ACE9BDF.val[0], 3);
      vi6x0246 = vi6x8ACE9BDF.val[0];
      const float16x4_t vi7x68AC = vext_f16(vi7x0246, vi7x8ACE9BDF.val[0], 3);
      vi7x0246 = vi7x8ACE9BDF.val[0];
      const float16x4_t vi8x68AC = vext_f16(vi8x0246, vi8x8ACE9BDF.val[0], 3);
      vi8x0246 = vi8x8ACE9BDF.val[0];

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x68AC, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x68AC, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x68AC, vw01234567, 1);

      vo0p1 = vfma_laneq_f16(vo0p1, vi1x68AC, vw01234567, 6);
      vo1p1 = vfma_laneq_f16(vo1p1, vi3x68AC, vw01234567, 6);
      vo2p1 = vfma_laneq_f16(vo2p1, vi5x68AC, vw01234567, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x68AC, vw89ABCDEF, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x68AC, vw89ABCDEF, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x68AC, vw89ABCDEF, 3);

      vo0p1 = vfma_laneq_f16(vo0p1, vi3x68AC, vwGHIJKLMN, 0);
      vo1p1 = vfma_laneq_f16(vo1p1, vi5x68AC, vwGHIJKLMN, 0);
      vo2p1 = vfma_laneq_f16(vo2p1, vi7x68AC, vwGHIJKLMN, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x68AC, vwGHIJKLMN, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi6x68AC, vwGHIJKLMN, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi8x68AC, vwGHIJKLMN, 5);

      // Left by 1 column
      const float16x4_t vi0x79BD = vext_f16(vi0x1357, vi0x8ACE9BDF.val[1], 3);
      vi0x1357 = vi0x8ACE9BDF.val[1];
      const float16x4_t vi1x79BD = vext_f16(vi1x1357, vi1x8ACE9BDF.val[1], 3);
      vi1x1357 = vi1x8ACE9BDF.val[1];
      const float16x4_t vi2x79BD = vext_f16(vi2x1357, vi2x8ACE9BDF.val[1], 3);
      vi2x1357 = vi2x8ACE9BDF.val[1];
      const float16x4_t vi3x79BD = vext_f16(vi3x1357, vi3x8ACE9BDF.val[1], 3);
      vi3x1357 = vi3x8ACE9BDF.val[1];
      const float16x4_t vi4x79BD = vext_f16(vi4x1357, vi4x8ACE9BDF.val[1], 3);
      vi4x1357 = vi4x8ACE9BDF.val[1];
      const float16x4_t vi5x79BD = vext_f16(vi5x1357, vi5x8ACE9BDF.val[1], 3);
      vi5x1357 = vi5x8ACE9BDF.val[1];
      const float16x4_t vi6x79BD = vext_f16(vi6x1357, vi6x8ACE9BDF.val[1], 3);
      vi6x1357 = vi6x8ACE9BDF.val[1];
      const float16x4_t vi7x79BD = vext_f16(vi7x1357, vi7x8ACE9BDF.val[1], 3);
      vi7x1357 = vi7x8ACE9BDF.val[1];
      const float16x4_t vi8x79BD = vext_f16(vi8x1357, vi8x8ACE9BDF.val[1], 3);
      vi8x1357 = vi8x8ACE9BDF.val[1];

      const float16x4x2_t vi0xGIKMHJLN = vld2_f16(i0); i0 += 8;
      const float16x4x2_t vi1xGIKMHJLN = vld2_f16(i1); i1 += 8;
      const float16x4x2_t vi2xGIKMHJLN = vld2_f16(i2); i2 += 8;
      const float16x4x2_t vi3xGIKMHJLN = vld2_f16(i3); i3 += 8;
      const float16x4x2_t vi4xGIKMHJLN = vld2_f16(i4); i4 += 8;
      const float16x4x2_t vi5xGIKMHJLN = vld2_f16(i5); i5 += 8;
      const float16x4x2_t vi6xGIKMHJLN = vld2_f16(i6); i6 += 8;
      const float16x4x2_t vi7xGIKMHJLN = vld2_f16(i7); i7 += 8;
      const float16x4x2_t vi8xGIKMHJLN = vld2_f16(i8); i8 += 8;

      vo0p1 = vfma_laneq_f16(vo0p1, vi0x79BD, vw01234567, 2);
      vo1p1 = vfma_laneq_f16(vo1p1, vi2x79BD, vw01234567, 2);
      vo2p1 = vfma_laneq_f16(vo2p1, vi4x79BD, vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x79BD, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x79BD, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x79BD, vw01234567, 7);

      vo0p1 = vfma_laneq_f16(vo0p1, vi2x79BD, vw89ABCDEF, 4);
      vo1p1 = vfma_laneq_f16(vo1p1, vi4x79BD, vw89ABCDEF, 4);
      vo2p1 = vfma_laneq_f16(vo2p1, vi6x79BD, vw89ABCDEF, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x79BD, vwGHIJKLMN, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x79BD, vwGHIJKLMN, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi7x79BD, vwGHIJKLMN, 1);

      vo0p1 = vfma_laneq_f16(vo0p1, vi4x79BD, vwGHIJKLMN, 6);
      vo1p1 = vfma_laneq_f16(vo1p1, vi6x79BD, vwGHIJKLMN, 6);
      vo2p1 = vfma_laneq_f16(vo2p1, vi8x79BD, vwGHIJKLMN, 6);

      // Right by 2 column
      const float16x4_t vi0xACEG = vext_f16(vi0x8ACE9BDF.val[0], vi0xGIKMHJLN.val[0], 1);
      vi0x8ACE9BDF = vi0xGIKMHJLN;
      const float16x4_t vi1xACEG = vext_f16(vi1x8ACE9BDF.val[0], vi1xGIKMHJLN.val[0], 1);
      vi1x8ACE9BDF = vi1xGIKMHJLN;
      const float16x4_t vi2xACEG = vext_f16(vi2x8ACE9BDF.val[0], vi2xGIKMHJLN.val[0], 1);
      vi2x8ACE9BDF = vi2xGIKMHJLN;
      const float16x4_t vi3xACEG = vext_f16(vi3x8ACE9BDF.val[0], vi3xGIKMHJLN.val[0], 1);
      vi3x8ACE9BDF = vi3xGIKMHJLN;
      const float16x4_t vi4xACEG = vext_f16(vi4x8ACE9BDF.val[0], vi4xGIKMHJLN.val[0], 1);
      vi4x8ACE9BDF = vi4xGIKMHJLN;
      const float16x4_t vi5xACEG = vext_f16(vi5x8ACE9BDF.val[0], vi5xGIKMHJLN.val[0], 1);
      vi5x8ACE9BDF = vi5xGIKMHJLN;
      const float16x4_t vi6xACEG = vext_f16(vi6x8ACE9BDF.val[0], vi6xGIKMHJLN.val[0], 1);
      vi6x8ACE9BDF = vi6xGIKMHJLN;
      const float16x4_t vi7xACEG = vext_f16(vi7x8ACE9BDF.val[0], vi7xGIKMHJLN.val[0], 1);
      vi7x8ACE9BDF = vi7xGIKMHJLN;
      const float16x4_t vi8xACEG = vext_f16(vi8x8ACE9BDF.val[0], vi8xGIKMHJLN.val[0], 1);
      vi8x8ACE9BDF = vi8xGIKMHJLN;

      vo0p0 = vfma_laneq_f16(vo0p0, vi0xACEG, vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2xACEG, vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4xACEG, vw01234567, 5);

      vo0p1 = vfma_laneq_f16(vo0p1, vi1xACEG, vw89ABCDEF, 2);
      vo1p1 = vfma_laneq_f16(vo1p1, vi3xACEG, vw89ABCDEF, 2);
      vo2p1 = vfma_laneq_f16(vo2p1, vi5xACEG, vw89ABCDEF, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2xACEG, vw89ABCDEF, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4xACEG, vw89ABCDEF, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6xACEG, vw89ABCDEF, 7);

      vo0p1 = vfma_laneq_f16(vo0p1, vi3xACEG, vwGHIJKLMN, 4);
      vo1p1 = vfma_laneq_f16(vo1p1, vi5xACEG, vwGHIJKLMN, 4);
      vo2p1 = vfma_laneq_f16(vo2p1, vi7xACEG, vwGHIJKLMN, 4);

      vo0p0 = vfma_lane_f16(vo0p0, vi4xACEG, vwOP, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi6xACEG, vwOP, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi8xACEG, vwOP, 1);

      vo0p0 = vadd_f16(vo0p0, vo0p1);
      vo1p0 = vadd_f16(vo1p0, vo1p1);
      vo2p0 = vadd_f16(vo2p0, vo2p1);

      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);

      vst1_f16(o2, vo2); o2 += 4;
      vst1_f16(o1, vo1); o1 += 4;
      vst1_f16(o0, vo0); o0 += 4;
    }

    // Last block has 1-8 pixels to process.
    assert(w <= 8 * sizeof(__fp16));
    assert(w >= 1 * sizeof(__fp16));
    {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);

      const float16x4_t vi0x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi0x8ACE9BDF.val[0])));
      const float16x4_t vi1x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi1x8ACE9BDF.val[0])));
      const float16x4_t vi2x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi2x8ACE9BDF.val[0])));
      const float16x4_t vi3x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi3x8ACE9BDF.val[0])));
      const float16x4_t vi4x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi4x8ACE9BDF.val[0])));
      const float16x4_t vi5x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi5x8ACE9BDF.val[0])));
      const float16x4_t vi6x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi6x8ACE9BDF.val[0])));
      const float16x4_t vi7x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi7x8ACE9BDF.val[0])));
      const float16x4_t vi8x8ACE = vreinterpret_f16_u16(vand_u16(vmask_even, vreinterpret_u16_f16(vi8x8ACE9BDF.val[0])));

      const float16x4_t vi0x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi0x8ACE9BDF.val[1])));
      const float16x4_t vi1x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi1x8ACE9BDF.val[1])));
      const float16x4_t vi2x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi2x8ACE9BDF.val[1])));
      const float16x4_t vi3x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi3x8ACE9BDF.val[1])));
      const float16x4_t vi4x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi4x8ACE9BDF.val[1])));
      const float16x4_t vi5x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi5x8ACE9BDF.val[1])));
      const float16x4_t vi6x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi6x8ACE9BDF.val[1])));
      const float16x4_t vi7x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi7x8ACE9BDF.val[1])));
      const float16x4_t vi8x9BDF = vreinterpret_f16_u16(vand_u16(vmask_odd, vreinterpret_u16_f16(vi8x8ACE9BDF.val[1])));

      // Center column
      float16x4_t vo0p1 = vmul_laneq_f16(vi0x8ACE, vw01234567, 3);
      float16x4_t vo1p1 = vmul_laneq_f16(vi2x8ACE, vw01234567, 3);
      float16x4_t vo2p1 = vmul_laneq_f16(vi4x8ACE, vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x8ACE, vw89ABCDEF, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x8ACE, vw89ABCDEF, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x8ACE, vw89ABCDEF, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x8ACE, vw89ABCDEF, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x8ACE, vw89ABCDEF, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x8ACE, vw89ABCDEF, 5);

      vo0p1 = vfma_laneq_f16(vo0p1, vi3x8ACE, vwGHIJKLMN, 2);
      vo1p1 = vfma_laneq_f16(vo1p1, vi5x8ACE, vwGHIJKLMN, 2);
      vo2p1 = vfma_laneq_f16(vo2p1, vi7x8ACE, vwGHIJKLMN, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x8ACE, vwGHIJKLMN, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi6x8ACE, vwGHIJKLMN, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi8x8ACE, vwGHIJKLMN, 7);

      // Right by 1 column
      vo0p1 = vfma_laneq_f16(vo0p1, vi0x9BDF, vw01234567, 4);
      vo1p1 = vfma_laneq_f16(vo1p1, vi2x9BDF, vw01234567, 4);
      vo2p1 = vfma_laneq_f16(vo2p1, vi4x9BDF, vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x9BDF, vw89ABCDEF, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x9BDF, vw89ABCDEF, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x9BDF, vw89ABCDEF, 1);

      vo0p1 = vfma_laneq_f16(vo0p1, vi2x9BDF, vw89ABCDEF, 6);
      vo1p1 = vfma_laneq_f16(vo1p1, vi4x9BDF, vw89ABCDEF, 6);
      vo2p1 = vfma_laneq_f16(vo2p1, vi6x9BDF, vw89ABCDEF, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x9BDF, vwGHIJKLMN, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x9BDF, vwGHIJKLMN, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi7x9BDF, vwGHIJKLMN, 3);

      vo0p1 = vfma_lane_f16(vo0p1, vi4x9BDF, vwOP, 0);
      vo1p1 = vfma_lane_f16(vo1p1, vi6x9BDF, vwOP, 0);
      vo2p1 = vfma_lane_f16(vo2p1, vi8x9BDF, vwOP, 0);

      // Left by 2 column
      const float16x4_t vi0x68AC = vext_f16(vi0x0246, vi0x8ACE, 3);
      const float16x4_t vi1x68AC = vext_f16(vi1x0246, vi1x8ACE, 3);
      const float16x4_t vi2x68AC = vext_f16(vi2x0246, vi2x8ACE, 3);
      const float16x4_t vi3x68AC = vext_f16(vi3x0246, vi3x8ACE, 3);
      const float16x4_t vi4x68AC = vext_f16(vi4x0246, vi4x8ACE, 3);
      const float16x4_t vi5x68AC = vext_f16(vi5x0246, vi5x8ACE, 3);
      const float16x4_t vi6x68AC = vext_f16(vi6x0246, vi6x8ACE, 3);
      const float16x4_t vi7x68AC = vext_f16(vi7x0246, vi7x8ACE, 3);
      const float16x4_t vi8x68AC = vext_f16(vi8x0246, vi8x8ACE, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x68AC, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x68AC, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x68AC, vw01234567, 1);

      vo0p1 = vfma_laneq_f16(vo0p1, vi1x68AC, vw01234567, 6);
      vo1p1 = vfma_laneq_f16(vo1p1, vi3x68AC, vw01234567, 6);
      vo2p1 = vfma_laneq_f16(vo2p1, vi5x68AC, vw01234567, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x68AC, vw89ABCDEF, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x68AC, vw89ABCDEF, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x68AC, vw89ABCDEF, 3);

      vo0p1 = vfma_laneq_f16(vo0p1, vi3x68AC, vwGHIJKLMN, 0);
      vo1p1 = vfma_laneq_f16(vo1p1, vi5x68AC, vwGHIJKLMN, 0);
      vo2p1 = vfma_laneq_f16(vo2p1, vi7x68AC, vwGHIJKLMN, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x68AC, vwGHIJKLMN, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi6x68AC, vwGHIJKLMN, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi8x68AC, vwGHIJKLMN, 5);

      // Left by 1 column
      const float16x4_t vi0x79BD = vext_f16(vi0x1357, vi0x9BDF, 3);
      const float16x4_t vi1x79BD = vext_f16(vi1x1357, vi1x9BDF, 3);
      const float16x4_t vi2x79BD = vext_f16(vi2x1357, vi2x9BDF, 3);
      const float16x4_t vi3x79BD = vext_f16(vi3x1357, vi3x9BDF, 3);
      const float16x4_t vi4x79BD = vext_f16(vi4x1357, vi4x9BDF, 3);
      const float16x4_t vi5x79BD = vext_f16(vi5x1357, vi5x9BDF, 3);
      const float16x4_t vi6x79BD = vext_f16(vi6x1357, vi6x9BDF, 3);
      const float16x4_t vi7x79BD = vext_f16(vi7x1357, vi7x9BDF, 3);
      const float16x4_t vi8x79BD = vext_f16(vi8x1357, vi8x9BDF, 3);

      vo0p1 = vfma_laneq_f16(vo0p1, vi0x79BD, vw01234567, 2);
      vo1p1 = vfma_laneq_f16(vo1p1, vi2x79BD, vw01234567, 2);
      vo2p1 = vfma_laneq_f16(vo2p1, vi4x79BD, vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x79BD, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x79BD, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x79BD, vw01234567, 7);

      vo0p1 = vfma_laneq_f16(vo0p1, vi2x79BD, vw89ABCDEF, 4);
      vo1p1 = vfma_laneq_f16(vo1p1, vi4x79BD, vw89ABCDEF, 4);
      vo2p1 = vfma_laneq_f16(vo2p1, vi6x79BD, vw89ABCDEF, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x79BD, vwGHIJKLMN, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x79BD, vwGHIJKLMN, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi7x79BD, vwGHIJKLMN, 1);

      vo0p1 = vfma_laneq_f16(vo0p1, vi4x79BD, vwGHIJKLMN, 6);
      vo1p1 = vfma_laneq_f16(vo1p1, vi6x79BD, vwGHIJKLMN, 6);
      vo2p1 = vfma_laneq_f16(vo2p1, vi8x79BD, vwGHIJKLMN, 6);

      // Right by 2 column
      const float16x4_t vzero = vmov_n_f16(0.0f);
      const float16x4_t vi0xACEG = vext_f16(vi0x8ACE, vzero, 1);
      const float16x4_t vi1xACEG = vext_f16(vi1x8ACE, vzero, 1);
      const float16x4_t vi2xACEG = vext_f16(vi2x8ACE, vzero, 1);
      const float16x4_t vi3xACEG = vext_f16(vi3x8ACE, vzero, 1);
      const float16x4_t vi4xACEG = vext_f16(vi4x8ACE, vzero, 1);
      const float16x4_t vi5xACEG = vext_f16(vi5x8ACE, vzero, 1);
      const float16x4_t vi6xACEG = vext_f16(vi6x8ACE, vzero, 1);
      const float16x4_t vi7xACEG = vext_f16(vi7x8ACE, vzero, 1);
      const float16x4_t vi8xACEG = vext_f16(vi8x8ACE, vzero, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0xACEG, vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2xACEG, vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4xACEG, vw01234567, 5);

      vo0p1 = vfma_laneq_f16(vo0p1, vi1xACEG, vw89ABCDEF, 2);
      vo1p1 = vfma_laneq_f16(vo1p1, vi3xACEG, vw89ABCDEF, 2);
      vo2p1 = vfma_laneq_f16(vo2p1, vi5xACEG, vw89ABCDEF, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2xACEG, vw89ABCDEF, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4xACEG, vw89ABCDEF, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6xACEG, vw89ABCDEF, 7);

      vo0p1 = vfma_laneq_f16(vo0p1, vi3xACEG, vwGHIJKLMN, 4);
      vo1p1 = vfma_laneq_f16(vo1p1, vi5xACEG, vwGHIJKLMN, 4);
      vo2p1 = vfma_laneq_f16(vo2p1, vi7xACEG, vwGHIJKLMN, 4);

      vo0p0 = vfma_lane_f16(vo0p0, vi4xACEG, vwOP, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi6xACEG, vwOP, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi8xACEG, vwOP, 1);

      vo0p0 = vadd_f16(vo0p0, vo0p1);
      vo1p0 = vadd_f16(vo1p0, vo1p1);
      vo2p0 = vadd_f16(vo2p0, vo2p1);

      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);

      size_t w_tmp = (w + 1 * sizeof(__fp16)) / (2 * sizeof(__fp16));
      if XNN_LIKELY(w_tmp >= 4) {
        vst1_f16(o2, vo2); o2 += 4;
        vst1_f16(o1, vo1); o1 += 4;
        vst1_f16(o0, vo0); o0 += 4;
      } else {
        if (w_tmp & 2) {
          vst1_lane_u32((void*) o2, vreinterpret_u32_f16(vo2), 0); o2 += 2;
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0), 0); o0 += 2;

          vo0 = vext_f16(vo0, vo0, 2);
          vo1 = vext_f16(vo1, vo1, 2);
          vo2 = vext_f16(vo2, vo2, 2);
        }
        if (w_tmp & 1) {
          vst1_lane_f16(o2, vo2, 0); o2 += 1;
          vst1_lane_f16(o1, vo1, 0); o1 += 1;
          vst1_lane_f16(o0, vo0, 0); o0 += 1;
        }
      }
    }

    i0 = (const __fp16*) ((uintptr_t) i6 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i7 - input_decrement);
    i2 = (const __fp16*) ((uintptr_t) i8 - input_decrement);
    i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
    i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
    i5 = (const __fp16*) ((uintptr_t) i4 + input_width);
    i6 = (const __fp16*) ((uintptr_t) i5 + input_width);
    i7 = (const __fp16*) ((uintptr_t) i6 + input_width);
    i8 = (const __fp16*) ((uintptr_t) i7 + input_width);

    o0 = o2;
    o1 = (__fp16*) ((uintptr_t) o0 + output_width);
    o2 = (__fp16*) ((uintptr_t) o1 + output_width);

    output_height = doz(output_height, 3);
    padded_input_height = doz(padded_input_height, 6);
  } while (output_height != 0);
}
