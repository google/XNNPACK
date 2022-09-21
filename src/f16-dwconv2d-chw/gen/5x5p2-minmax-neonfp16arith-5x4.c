// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in
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


void xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_5x4(
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
  assert(padding_top == 2);

  const uint16x4_t vmask = vld1_u16(params->neonfp16arith.maskx4);
  const float16x4_t vmax = vld1_dup_f16(&params->neonfp16arith.max);
  const float16x4_t vmin = vld1_dup_f16(&params->neonfp16arith.min);

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x8_t vw89ABCDEF = vld1q_f16(w0 + 8);
  const float16x8_t vwGHIJKLMN = vld1q_f16(w0 + 16);
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 24)));

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(__fp16));

  const __fp16* i0 = zero;
  const __fp16* i1 = zero;
  const __fp16* i2 = input;
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_width);
  const __fp16* i6 = (const __fp16*) ((uintptr_t) i5 + input_width);
  const __fp16* i7 = (const __fp16*) ((uintptr_t) i6 + input_width);
  const __fp16* i8 = (const __fp16*) ((uintptr_t) i7 + input_width);

  __fp16* o0 = output;
  __fp16* o1 = (__fp16*) ((uintptr_t) o0 + input_width);
  __fp16* o2 = (__fp16*) ((uintptr_t) o1 + input_width);
  __fp16* o3 = (__fp16*) ((uintptr_t) o2 + input_width);
  __fp16* o4 = (__fp16*) ((uintptr_t) o3 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i6 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i7 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 7) {
      i8 = zero;
    }

    float16x4_t vi0x0123 = vmov_n_f16(0);
    float16x4_t vi1x0123 = vmov_n_f16(0);
    float16x4_t vi2x0123 = vmov_n_f16(0);
    float16x4_t vi3x0123 = vmov_n_f16(0);
    float16x4_t vi4x0123 = vmov_n_f16(0);
    float16x4_t vi5x0123 = vmov_n_f16(0);
    float16x4_t vi6x0123 = vmov_n_f16(0);
    float16x4_t vi7x0123 = vmov_n_f16(0);
    float16x4_t vi8x0123 = vmov_n_f16(0);

    float16x4_t vi0x4567 = vld1_f16(i0); i0 += 4;
    float16x4_t vi1x4567 = vld1_f16(i1); i1 += 4;
    float16x4_t vi2x4567 = vld1_f16(i2); i2 += 4;
    float16x4_t vi3x4567 = vld1_f16(i3); i3 += 4;
    float16x4_t vi4x4567 = vld1_f16(i4); i4 += 4;
    float16x4_t vi5x4567 = vld1_f16(i5); i5 += 4;
    float16x4_t vi6x4567 = vld1_f16(i6); i6 += 4;
    float16x4_t vi7x4567 = vld1_f16(i7); i7 += 4;
    float16x4_t vi8x4567 = vld1_f16(i8); i8 += 4;

    size_t w = input_width;
    for (; w > 8 * sizeof(__fp16); w -= 4 * sizeof(__fp16)) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo3p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo4p0 = vdup_laneq_f16(vw01234567, 0);

      const float16x4_t vi0x89AB = vld1_f16(i0); i0 += 4;
      const float16x4_t vi1x89AB = vld1_f16(i1); i1 += 4;
      const float16x4_t vi2x89AB = vld1_f16(i2); i2 += 4;
      const float16x4_t vi3x89AB = vld1_f16(i3); i3 += 4;
      const float16x4_t vi4x89AB = vld1_f16(i4); i4 += 4;
      const float16x4_t vi5x89AB = vld1_f16(i5); i5 += 4;
      const float16x4_t vi6x89AB = vld1_f16(i6); i6 += 4;
      const float16x4_t vi7x89AB = vld1_f16(i7); i7 += 4;
      const float16x4_t vi8x89AB = vld1_f16(i8); i8 += 4;

      // Center column
      vo0p0 = vfma_laneq_f16(vo0p0, vi0x4567, vw01234567, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x4567, vw01234567, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x4567, vw01234567, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x4567, vw01234567, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x4567, vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x4567, vw89ABCDEF, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x4567, vw89ABCDEF, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x4567, vw89ABCDEF, 0);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x4567, vw89ABCDEF, 0);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x4567, vw89ABCDEF, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x4567, vw89ABCDEF, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x4567, vw89ABCDEF, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x4567, vw89ABCDEF, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x4567, vw89ABCDEF, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x4567, vw89ABCDEF, 5);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x4567, vwGHIJKLMN, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x4567, vwGHIJKLMN, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x4567, vwGHIJKLMN, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x4567, vwGHIJKLMN, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x4567, vwGHIJKLMN, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x4567, vwGHIJKLMN, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x4567, vwGHIJKLMN, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x4567, vwGHIJKLMN, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x4567, vwGHIJKLMN, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x4567, vwGHIJKLMN, 7);

      // Left by 1 column
      const float16x4_t vi0x3456 = vext_f16(vi0x0123, vi0x4567, 3);
      const float16x4_t vi1x3456 = vext_f16(vi1x0123, vi1x4567, 3);
      const float16x4_t vi2x3456 = vext_f16(vi2x0123, vi2x4567, 3);
      const float16x4_t vi3x3456 = vext_f16(vi3x0123, vi3x4567, 3);
      const float16x4_t vi4x3456 = vext_f16(vi4x0123, vi4x4567, 3);
      const float16x4_t vi5x3456 = vext_f16(vi5x0123, vi5x4567, 3);
      const float16x4_t vi6x3456 = vext_f16(vi6x0123, vi6x4567, 3);
      const float16x4_t vi7x3456 = vext_f16(vi7x0123, vi7x4567, 3);
      const float16x4_t vi8x3456 = vext_f16(vi8x0123, vi8x4567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x3456, vw01234567, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x3456, vw01234567, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x3456, vw01234567, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x3456, vw01234567, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x3456, vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x3456, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x3456, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x3456, vw01234567, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x3456, vw01234567, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x3456, vw01234567, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x3456, vw89ABCDEF, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x3456, vw89ABCDEF, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x3456, vw89ABCDEF, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x3456, vw89ABCDEF, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x3456, vw89ABCDEF, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x3456, vwGHIJKLMN, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x3456, vwGHIJKLMN, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x3456, vwGHIJKLMN, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x3456, vwGHIJKLMN, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x3456, vwGHIJKLMN, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x3456, vwGHIJKLMN, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x3456, vwGHIJKLMN, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x3456, vwGHIJKLMN, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x3456, vwGHIJKLMN, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x3456, vwGHIJKLMN, 6);

      // Left by 2 column
      const float16x4_t vi0x2345 = vext_f16(vi0x0123, vi0x4567, 2);
      vi0x0123 = vi0x4567;
      const float16x4_t vi1x2345 = vext_f16(vi1x0123, vi1x4567, 2);
      vi1x0123 = vi1x4567;
      const float16x4_t vi2x2345 = vext_f16(vi2x0123, vi2x4567, 2);
      vi2x0123 = vi2x4567;
      const float16x4_t vi3x2345 = vext_f16(vi3x0123, vi3x4567, 2);
      vi3x0123 = vi3x4567;
      const float16x4_t vi4x2345 = vext_f16(vi4x0123, vi4x4567, 2);
      vi4x0123 = vi4x4567;
      const float16x4_t vi5x2345 = vext_f16(vi5x0123, vi5x4567, 2);
      vi5x0123 = vi5x4567;
      const float16x4_t vi6x2345 = vext_f16(vi6x0123, vi6x4567, 2);
      vi6x0123 = vi6x4567;
      const float16x4_t vi7x2345 = vext_f16(vi7x0123, vi7x4567, 2);
      vi7x0123 = vi7x4567;
      const float16x4_t vi8x2345 = vext_f16(vi8x0123, vi8x4567, 2);
      vi8x0123 = vi8x4567;

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x2345, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x2345, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x2345, vw01234567, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x2345, vw01234567, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x2345, vw01234567, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x2345, vw01234567, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x2345, vw01234567, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x2345, vw01234567, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x2345, vw01234567, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x2345, vw01234567, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x2345, vw89ABCDEF, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x2345, vw89ABCDEF, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x2345, vw89ABCDEF, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x2345, vw89ABCDEF, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x2345, vw89ABCDEF, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x2345, vwGHIJKLMN, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x2345, vwGHIJKLMN, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x2345, vwGHIJKLMN, 0);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x2345, vwGHIJKLMN, 0);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x2345, vwGHIJKLMN, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x2345, vwGHIJKLMN, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x2345, vwGHIJKLMN, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x2345, vwGHIJKLMN, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x2345, vwGHIJKLMN, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x2345, vwGHIJKLMN, 5);

      // Right by 1 column
      const float16x4_t vi0x5678 = vext_f16(vi0x4567, vi0x89AB, 1);
      const float16x4_t vi1x5678 = vext_f16(vi1x4567, vi1x89AB, 1);
      const float16x4_t vi2x5678 = vext_f16(vi2x4567, vi2x89AB, 1);
      const float16x4_t vi3x5678 = vext_f16(vi3x4567, vi3x89AB, 1);
      const float16x4_t vi4x5678 = vext_f16(vi4x4567, vi4x89AB, 1);
      const float16x4_t vi5x5678 = vext_f16(vi5x4567, vi5x89AB, 1);
      const float16x4_t vi6x5678 = vext_f16(vi6x4567, vi6x89AB, 1);
      const float16x4_t vi7x5678 = vext_f16(vi7x4567, vi7x89AB, 1);
      const float16x4_t vi8x5678 = vext_f16(vi8x4567, vi8x89AB, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x5678, vw01234567, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x5678, vw01234567, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x5678, vw01234567, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x5678, vw01234567, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x5678, vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x5678, vw89ABCDEF, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x5678, vw89ABCDEF, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x5678, vw89ABCDEF, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x5678, vw89ABCDEF, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x5678, vw89ABCDEF, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x5678, vw89ABCDEF, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x5678, vw89ABCDEF, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x5678, vw89ABCDEF, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x5678, vw89ABCDEF, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x5678, vw89ABCDEF, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x5678, vwGHIJKLMN, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x5678, vwGHIJKLMN, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x5678, vwGHIJKLMN, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x5678, vwGHIJKLMN, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x5678, vwGHIJKLMN, 3);

      vo0p0 = vfma_lane_f16(vo0p0, vi4x5678, vwOP, 0);
      vo1p0 = vfma_lane_f16(vo1p0, vi5x5678, vwOP, 0);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x5678, vwOP, 0);
      vo3p0 = vfma_lane_f16(vo3p0, vi7x5678, vwOP, 0);
      vo4p0 = vfma_lane_f16(vo4p0, vi8x5678, vwOP, 0);

      // Right by 2 column
      const float16x4_t vi0x6789 = vext_f16(vi0x4567, vi0x89AB, 2);
      vi0x4567 = vi0x89AB;
      const float16x4_t vi1x6789 = vext_f16(vi1x4567, vi1x89AB, 2);
      vi1x4567 = vi1x89AB;
      const float16x4_t vi2x6789 = vext_f16(vi2x4567, vi2x89AB, 2);
      vi2x4567 = vi2x89AB;
      const float16x4_t vi3x6789 = vext_f16(vi3x4567, vi3x89AB, 2);
      vi3x4567 = vi3x89AB;
      const float16x4_t vi4x6789 = vext_f16(vi4x4567, vi4x89AB, 2);
      vi4x4567 = vi4x89AB;
      const float16x4_t vi5x6789 = vext_f16(vi5x4567, vi5x89AB, 2);
      vi5x4567 = vi5x89AB;
      const float16x4_t vi6x6789 = vext_f16(vi6x4567, vi6x89AB, 2);
      vi6x4567 = vi6x89AB;
      const float16x4_t vi7x6789 = vext_f16(vi7x4567, vi7x89AB, 2);
      vi7x4567 = vi7x89AB;
      const float16x4_t vi8x6789 = vext_f16(vi8x4567, vi8x89AB, 2);
      vi8x4567 = vi8x89AB;

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x6789, vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x6789, vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x6789, vw01234567, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x6789, vw01234567, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x6789, vw01234567, 5);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x6789, vw89ABCDEF, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x6789, vw89ABCDEF, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x6789, vw89ABCDEF, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x6789, vw89ABCDEF, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x6789, vw89ABCDEF, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x6789, vw89ABCDEF, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x6789, vw89ABCDEF, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x6789, vw89ABCDEF, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x6789, vw89ABCDEF, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x6789, vw89ABCDEF, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x6789, vwGHIJKLMN, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x6789, vwGHIJKLMN, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x6789, vwGHIJKLMN, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x6789, vwGHIJKLMN, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x6789, vwGHIJKLMN, 4);

      vo0p0 = vfma_lane_f16(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x6789, vwOP, 1);
      vo3p0 = vfma_lane_f16(vo3p0, vi7x6789, vwOP, 1);
      vo4p0 = vfma_lane_f16(vo4p0, vi8x6789, vwOP, 1);


      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);
      float16x4_t vo3 = vmax_f16(vo3p0, vmin);
      float16x4_t vo4 = vmax_f16(vo4p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);
      vo3 = vmin_f16(vo3, vmax);
      vo4 = vmin_f16(vo4, vmax);

      vst1_f16(o4, vo4); o4 += 4;
      vst1_f16(o3, vo3); o3 += 4;
      vst1_f16(o2, vo2); o2 += 4;
      vst1_f16(o1, vo1); o1 += 4;
      vst1_f16(o0, vo0); o0 += 4;
    }

    // Always process the last block of 5..8 pixels.
    assert(w <= 8 * sizeof(__fp16));
    if XNN_LIKELY(w > 4 * sizeof(__fp16)) {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo3p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo4p0 = vdup_laneq_f16(vw01234567, 0);

      float16x4_t vi0x89AB = vld1_f16(i0); i0 += 4;
      float16x4_t vi1x89AB = vld1_f16(i1); i1 += 4;
      float16x4_t vi2x89AB = vld1_f16(i2); i2 += 4;
      float16x4_t vi3x89AB = vld1_f16(i3); i3 += 4;
      float16x4_t vi4x89AB = vld1_f16(i4); i4 += 4;
      float16x4_t vi5x89AB = vld1_f16(i5); i5 += 4;
      float16x4_t vi6x89AB = vld1_f16(i6); i6 += 4;
      float16x4_t vi7x89AB = vld1_f16(i7); i7 += 4;
      float16x4_t vi8x89AB = vld1_f16(i8); i8 += 4;

      vi0x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi0x89AB)));
      vi1x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi1x89AB)));
      vi2x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi2x89AB)));
      vi3x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi3x89AB)));
      vi4x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi4x89AB)));
      vi5x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi5x89AB)));
      vi6x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi6x89AB)));
      vi7x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi7x89AB)));
      vi8x89AB = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi8x89AB)));

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x4567, vw01234567, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x4567, vw01234567, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x4567, vw01234567, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x4567, vw01234567, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x4567, vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x4567, vw89ABCDEF, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x4567, vw89ABCDEF, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x4567, vw89ABCDEF, 0);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x4567, vw89ABCDEF, 0);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x4567, vw89ABCDEF, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x4567, vw89ABCDEF, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x4567, vw89ABCDEF, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x4567, vw89ABCDEF, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x4567, vw89ABCDEF, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x4567, vw89ABCDEF, 5);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x4567, vwGHIJKLMN, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x4567, vwGHIJKLMN, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x4567, vwGHIJKLMN, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x4567, vwGHIJKLMN, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x4567, vwGHIJKLMN, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x4567, vwGHIJKLMN, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x4567, vwGHIJKLMN, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x4567, vwGHIJKLMN, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x4567, vwGHIJKLMN, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x4567, vwGHIJKLMN, 7);

      const float16x4_t vi0x3456 = vext_f16(vi0x0123, vi0x4567, 3);
      const float16x4_t vi1x3456 = vext_f16(vi1x0123, vi1x4567, 3);
      const float16x4_t vi2x3456 = vext_f16(vi2x0123, vi2x4567, 3);
      const float16x4_t vi3x3456 = vext_f16(vi3x0123, vi3x4567, 3);
      const float16x4_t vi4x3456 = vext_f16(vi4x0123, vi4x4567, 3);
      const float16x4_t vi5x3456 = vext_f16(vi5x0123, vi5x4567, 3);
      const float16x4_t vi6x3456 = vext_f16(vi6x0123, vi6x4567, 3);
      const float16x4_t vi7x3456 = vext_f16(vi7x0123, vi7x4567, 3);
      const float16x4_t vi8x3456 = vext_f16(vi8x0123, vi8x4567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x3456, vw01234567, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x3456, vw01234567, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x3456, vw01234567, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x3456, vw01234567, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x3456, vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x3456, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x3456, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x3456, vw01234567, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x3456, vw01234567, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x3456, vw01234567, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x3456, vw89ABCDEF, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x3456, vw89ABCDEF, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x3456, vw89ABCDEF, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x3456, vw89ABCDEF, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x3456, vw89ABCDEF, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x3456, vwGHIJKLMN, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x3456, vwGHIJKLMN, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x3456, vwGHIJKLMN, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x3456, vwGHIJKLMN, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x3456, vwGHIJKLMN, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x3456, vwGHIJKLMN, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x3456, vwGHIJKLMN, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x3456, vwGHIJKLMN, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x3456, vwGHIJKLMN, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x3456, vwGHIJKLMN, 6);

      const float16x4_t vi0x2345 = vext_f16(vi0x0123, vi0x4567, 2);
      vi0x0123 = vi0x4567;
      const float16x4_t vi1x2345 = vext_f16(vi1x0123, vi1x4567, 2);
      vi1x0123 = vi1x4567;
      const float16x4_t vi2x2345 = vext_f16(vi2x0123, vi2x4567, 2);
      vi2x0123 = vi2x4567;
      const float16x4_t vi3x2345 = vext_f16(vi3x0123, vi3x4567, 2);
      vi3x0123 = vi3x4567;
      const float16x4_t vi4x2345 = vext_f16(vi4x0123, vi4x4567, 2);
      vi4x0123 = vi4x4567;
      const float16x4_t vi5x2345 = vext_f16(vi5x0123, vi5x4567, 2);
      vi5x0123 = vi5x4567;
      const float16x4_t vi6x2345 = vext_f16(vi6x0123, vi6x4567, 2);
      vi6x0123 = vi6x4567;
      const float16x4_t vi7x2345 = vext_f16(vi7x0123, vi7x4567, 2);
      vi7x0123 = vi7x4567;
      const float16x4_t vi8x2345 = vext_f16(vi8x0123, vi8x4567, 2);
      vi8x0123 = vi8x4567;

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x2345, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x2345, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x2345, vw01234567, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x2345, vw01234567, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x2345, vw01234567, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x2345, vw01234567, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x2345, vw01234567, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x2345, vw01234567, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x2345, vw01234567, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x2345, vw01234567, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x2345, vw89ABCDEF, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x2345, vw89ABCDEF, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x2345, vw89ABCDEF, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x2345, vw89ABCDEF, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x2345, vw89ABCDEF, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x2345, vwGHIJKLMN, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x2345, vwGHIJKLMN, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x2345, vwGHIJKLMN, 0);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x2345, vwGHIJKLMN, 0);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x2345, vwGHIJKLMN, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x2345, vwGHIJKLMN, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x2345, vwGHIJKLMN, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x2345, vwGHIJKLMN, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x2345, vwGHIJKLMN, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x2345, vwGHIJKLMN, 5);

      const float16x4_t vi0x5678 = vext_f16(vi0x4567, vi0x89AB, 1);
      const float16x4_t vi1x5678 = vext_f16(vi1x4567, vi1x89AB, 1);
      const float16x4_t vi2x5678 = vext_f16(vi2x4567, vi2x89AB, 1);
      const float16x4_t vi3x5678 = vext_f16(vi3x4567, vi3x89AB, 1);
      const float16x4_t vi4x5678 = vext_f16(vi4x4567, vi4x89AB, 1);
      const float16x4_t vi5x5678 = vext_f16(vi5x4567, vi5x89AB, 1);
      const float16x4_t vi6x5678 = vext_f16(vi6x4567, vi6x89AB, 1);
      const float16x4_t vi7x5678 = vext_f16(vi7x4567, vi7x89AB, 1);
      const float16x4_t vi8x5678 = vext_f16(vi8x4567, vi8x89AB, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x5678, vw01234567, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x5678, vw01234567, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x5678, vw01234567, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x5678, vw01234567, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x5678, vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x5678, vw89ABCDEF, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x5678, vw89ABCDEF, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x5678, vw89ABCDEF, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x5678, vw89ABCDEF, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x5678, vw89ABCDEF, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x5678, vw89ABCDEF, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x5678, vw89ABCDEF, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x5678, vw89ABCDEF, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x5678, vw89ABCDEF, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x5678, vw89ABCDEF, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x5678, vwGHIJKLMN, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x5678, vwGHIJKLMN, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x5678, vwGHIJKLMN, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x5678, vwGHIJKLMN, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x5678, vwGHIJKLMN, 3);

      vo0p0 = vfma_lane_f16(vo0p0, vi4x5678, vwOP, 0);
      vo1p0 = vfma_lane_f16(vo1p0, vi5x5678, vwOP, 0);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x5678, vwOP, 0);
      vo3p0 = vfma_lane_f16(vo3p0, vi7x5678, vwOP, 0);
      vo4p0 = vfma_lane_f16(vo4p0, vi8x5678, vwOP, 0);

      const float16x4_t vi0x6789 = vext_f16(vi0x4567, vi0x89AB, 2);
      vi0x4567 = vi0x89AB;
      const float16x4_t vi1x6789 = vext_f16(vi1x4567, vi1x89AB, 2);
      vi1x4567 = vi1x89AB;
      const float16x4_t vi2x6789 = vext_f16(vi2x4567, vi2x89AB, 2);
      vi2x4567 = vi2x89AB;
      const float16x4_t vi3x6789 = vext_f16(vi3x4567, vi3x89AB, 2);
      vi3x4567 = vi3x89AB;
      const float16x4_t vi4x6789 = vext_f16(vi4x4567, vi4x89AB, 2);
      vi4x4567 = vi4x89AB;
      const float16x4_t vi5x6789 = vext_f16(vi5x4567, vi5x89AB, 2);
      vi5x4567 = vi5x89AB;
      const float16x4_t vi6x6789 = vext_f16(vi6x4567, vi6x89AB, 2);
      vi6x4567 = vi6x89AB;
      const float16x4_t vi7x6789 = vext_f16(vi7x4567, vi7x89AB, 2);
      vi7x4567 = vi7x89AB;
      const float16x4_t vi8x6789 = vext_f16(vi8x4567, vi8x89AB, 2);
      vi8x4567 = vi8x89AB;

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x6789, vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x6789, vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x6789, vw01234567, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x6789, vw01234567, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x6789, vw01234567, 5);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x6789, vw89ABCDEF, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x6789, vw89ABCDEF, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x6789, vw89ABCDEF, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x6789, vw89ABCDEF, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x6789, vw89ABCDEF, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x6789, vw89ABCDEF, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x6789, vw89ABCDEF, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x6789, vw89ABCDEF, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x6789, vw89ABCDEF, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x6789, vw89ABCDEF, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x6789, vwGHIJKLMN, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x6789, vwGHIJKLMN, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x6789, vwGHIJKLMN, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x6789, vwGHIJKLMN, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x6789, vwGHIJKLMN, 4);

      vo0p0 = vfma_lane_f16(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x6789, vwOP, 1);
      vo3p0 = vfma_lane_f16(vo3p0, vi7x6789, vwOP, 1);
      vo4p0 = vfma_lane_f16(vo4p0, vi8x6789, vwOP, 1);


      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);
      float16x4_t vo3 = vmax_f16(vo3p0, vmin);
      float16x4_t vo4 = vmax_f16(vo4p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);
      vo3 = vmin_f16(vo3, vmax);
      vo4 = vmin_f16(vo4, vmax);

      vst1_f16(o4, vo4); o4 += 4;
      vst1_f16(o3, vo3); o3 += 4;
      vst1_f16(o2, vo2); o2 += 4;
      vst1_f16(o1, vo1); o1 += 4;
      vst1_f16(o0, vo0); o0 += 4;

      w -= 4 * sizeof(__fp16);
    }

    assert(w >= 1 * sizeof(__fp16));
    assert(w <= 4 * sizeof(__fp16));
    {
      float16x4_t vo0p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo1p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo2p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo3p0 = vdup_laneq_f16(vw01234567, 0);
      float16x4_t vo4p0 = vdup_laneq_f16(vw01234567, 0);

      vi0x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi0x4567)));
      vi1x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi1x4567)));
      vi2x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi2x4567)));
      vi3x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi3x4567)));
      vi4x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi4x4567)));
      vi5x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi5x4567)));
      vi6x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi6x4567)));
      vi7x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi7x4567)));
      vi8x4567 = vreinterpret_f16_u16(vand_u16(vmask, vreinterpret_u16_f16(vi8x4567)));

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x4567, vw01234567, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x4567, vw01234567, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x4567, vw01234567, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x4567, vw01234567, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x4567, vw01234567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x4567, vw89ABCDEF, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x4567, vw89ABCDEF, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x4567, vw89ABCDEF, 0);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x4567, vw89ABCDEF, 0);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x4567, vw89ABCDEF, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x4567, vw89ABCDEF, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x4567, vw89ABCDEF, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x4567, vw89ABCDEF, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x4567, vw89ABCDEF, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x4567, vw89ABCDEF, 5);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x4567, vwGHIJKLMN, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x4567, vwGHIJKLMN, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x4567, vwGHIJKLMN, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x4567, vwGHIJKLMN, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x4567, vwGHIJKLMN, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x4567, vwGHIJKLMN, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x4567, vwGHIJKLMN, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x4567, vwGHIJKLMN, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x4567, vwGHIJKLMN, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x4567, vwGHIJKLMN, 7);

      const float16x4_t vi0x3456 = vext_f16(vi0x0123, vi0x4567, 3);
      const float16x4_t vi1x3456 = vext_f16(vi1x0123, vi1x4567, 3);
      const float16x4_t vi2x3456 = vext_f16(vi2x0123, vi2x4567, 3);
      const float16x4_t vi3x3456 = vext_f16(vi3x0123, vi3x4567, 3);
      const float16x4_t vi4x3456 = vext_f16(vi4x0123, vi4x4567, 3);
      const float16x4_t vi5x3456 = vext_f16(vi5x0123, vi5x4567, 3);
      const float16x4_t vi6x3456 = vext_f16(vi6x0123, vi6x4567, 3);
      const float16x4_t vi7x3456 = vext_f16(vi7x0123, vi7x4567, 3);
      const float16x4_t vi8x3456 = vext_f16(vi8x0123, vi8x4567, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x3456, vw01234567, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x3456, vw01234567, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x3456, vw01234567, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x3456, vw01234567, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x3456, vw01234567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x3456, vw01234567, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x3456, vw01234567, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x3456, vw01234567, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x3456, vw01234567, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x3456, vw01234567, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x3456, vw89ABCDEF, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x3456, vw89ABCDEF, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x3456, vw89ABCDEF, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x3456, vw89ABCDEF, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x3456, vw89ABCDEF, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x3456, vwGHIJKLMN, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x3456, vwGHIJKLMN, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x3456, vwGHIJKLMN, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x3456, vwGHIJKLMN, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x3456, vwGHIJKLMN, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x3456, vwGHIJKLMN, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x3456, vwGHIJKLMN, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x3456, vwGHIJKLMN, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x3456, vwGHIJKLMN, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x3456, vwGHIJKLMN, 6);

      const float16x4_t vi0x2345 = vext_f16(vi0x0123, vi0x4567, 2);
      const float16x4_t vi1x2345 = vext_f16(vi1x0123, vi1x4567, 2);
      const float16x4_t vi2x2345 = vext_f16(vi2x0123, vi2x4567, 2);
      const float16x4_t vi3x2345 = vext_f16(vi3x0123, vi3x4567, 2);
      const float16x4_t vi4x2345 = vext_f16(vi4x0123, vi4x4567, 2);
      const float16x4_t vi5x2345 = vext_f16(vi5x0123, vi5x4567, 2);
      const float16x4_t vi6x2345 = vext_f16(vi6x0123, vi6x4567, 2);
      const float16x4_t vi7x2345 = vext_f16(vi7x0123, vi7x4567, 2);
      const float16x4_t vi8x2345 = vext_f16(vi8x0123, vi8x4567, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x2345, vw01234567, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x2345, vw01234567, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x2345, vw01234567, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x2345, vw01234567, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x2345, vw01234567, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x2345, vw01234567, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x2345, vw01234567, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x2345, vw01234567, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x2345, vw01234567, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x2345, vw01234567, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x2345, vw89ABCDEF, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x2345, vw89ABCDEF, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x2345, vw89ABCDEF, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x2345, vw89ABCDEF, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x2345, vw89ABCDEF, 3);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x2345, vwGHIJKLMN, 0);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x2345, vwGHIJKLMN, 0);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x2345, vwGHIJKLMN, 0);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x2345, vwGHIJKLMN, 0);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x2345, vwGHIJKLMN, 0);

      vo0p0 = vfma_laneq_f16(vo0p0, vi4x2345, vwGHIJKLMN, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi5x2345, vwGHIJKLMN, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi6x2345, vwGHIJKLMN, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi7x2345, vwGHIJKLMN, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi8x2345, vwGHIJKLMN, 5);

      const float16x4_t vzero = vmov_n_f16(0);
      const float16x4_t vi0x5678 = vext_f16(vi0x4567, vzero, 1);
      const float16x4_t vi1x5678 = vext_f16(vi1x4567, vzero, 1);
      const float16x4_t vi2x5678 = vext_f16(vi2x4567, vzero, 1);
      const float16x4_t vi3x5678 = vext_f16(vi3x4567, vzero, 1);
      const float16x4_t vi4x5678 = vext_f16(vi4x4567, vzero, 1);
      const float16x4_t vi5x5678 = vext_f16(vi5x4567, vzero, 1);
      const float16x4_t vi6x5678 = vext_f16(vi6x4567, vzero, 1);
      const float16x4_t vi7x5678 = vext_f16(vi7x4567, vzero, 1);
      const float16x4_t vi8x5678 = vext_f16(vi8x4567, vzero, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x5678, vw01234567, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x5678, vw01234567, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x5678, vw01234567, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x5678, vw01234567, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x5678, vw01234567, 4);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x5678, vw89ABCDEF, 1);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x5678, vw89ABCDEF, 1);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x5678, vw89ABCDEF, 1);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x5678, vw89ABCDEF, 1);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x5678, vw89ABCDEF, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x5678, vw89ABCDEF, 6);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x5678, vw89ABCDEF, 6);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x5678, vw89ABCDEF, 6);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x5678, vw89ABCDEF, 6);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x5678, vw89ABCDEF, 6);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x5678, vwGHIJKLMN, 3);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x5678, vwGHIJKLMN, 3);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x5678, vwGHIJKLMN, 3);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x5678, vwGHIJKLMN, 3);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x5678, vwGHIJKLMN, 3);

      vo0p0 = vfma_lane_f16(vo0p0, vi4x5678, vwOP, 0);
      vo1p0 = vfma_lane_f16(vo1p0, vi5x5678, vwOP, 0);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x5678, vwOP, 0);
      vo3p0 = vfma_lane_f16(vo3p0, vi7x5678, vwOP, 0);
      vo4p0 = vfma_lane_f16(vo4p0, vi8x5678, vwOP, 0);

      const float16x4_t vi0x6789 = vext_f16(vi0x5678, vzero, 1);
      const float16x4_t vi1x6789 = vext_f16(vi1x5678, vzero, 1);
      const float16x4_t vi2x6789 = vext_f16(vi2x5678, vzero, 1);
      const float16x4_t vi3x6789 = vext_f16(vi3x5678, vzero, 1);
      const float16x4_t vi4x6789 = vext_f16(vi4x5678, vzero, 1);
      const float16x4_t vi5x6789 = vext_f16(vi5x5678, vzero, 1);
      const float16x4_t vi6x6789 = vext_f16(vi6x5678, vzero, 1);
      const float16x4_t vi7x6789 = vext_f16(vi7x5678, vzero, 1);
      const float16x4_t vi8x6789 = vext_f16(vi8x5678, vzero, 1);

      vo0p0 = vfma_laneq_f16(vo0p0, vi0x6789, vw01234567, 5);
      vo1p0 = vfma_laneq_f16(vo1p0, vi1x6789, vw01234567, 5);
      vo2p0 = vfma_laneq_f16(vo2p0, vi2x6789, vw01234567, 5);
      vo3p0 = vfma_laneq_f16(vo3p0, vi3x6789, vw01234567, 5);
      vo4p0 = vfma_laneq_f16(vo4p0, vi4x6789, vw01234567, 5);

      vo0p0 = vfma_laneq_f16(vo0p0, vi1x6789, vw89ABCDEF, 2);
      vo1p0 = vfma_laneq_f16(vo1p0, vi2x6789, vw89ABCDEF, 2);
      vo2p0 = vfma_laneq_f16(vo2p0, vi3x6789, vw89ABCDEF, 2);
      vo3p0 = vfma_laneq_f16(vo3p0, vi4x6789, vw89ABCDEF, 2);
      vo4p0 = vfma_laneq_f16(vo4p0, vi5x6789, vw89ABCDEF, 2);

      vo0p0 = vfma_laneq_f16(vo0p0, vi2x6789, vw89ABCDEF, 7);
      vo1p0 = vfma_laneq_f16(vo1p0, vi3x6789, vw89ABCDEF, 7);
      vo2p0 = vfma_laneq_f16(vo2p0, vi4x6789, vw89ABCDEF, 7);
      vo3p0 = vfma_laneq_f16(vo3p0, vi5x6789, vw89ABCDEF, 7);
      vo4p0 = vfma_laneq_f16(vo4p0, vi6x6789, vw89ABCDEF, 7);

      vo0p0 = vfma_laneq_f16(vo0p0, vi3x6789, vwGHIJKLMN, 4);
      vo1p0 = vfma_laneq_f16(vo1p0, vi4x6789, vwGHIJKLMN, 4);
      vo2p0 = vfma_laneq_f16(vo2p0, vi5x6789, vwGHIJKLMN, 4);
      vo3p0 = vfma_laneq_f16(vo3p0, vi6x6789, vwGHIJKLMN, 4);
      vo4p0 = vfma_laneq_f16(vo4p0, vi7x6789, vwGHIJKLMN, 4);

      vo0p0 = vfma_lane_f16(vo0p0, vi4x6789, vwOP, 1);
      vo1p0 = vfma_lane_f16(vo1p0, vi5x6789, vwOP, 1);
      vo2p0 = vfma_lane_f16(vo2p0, vi6x6789, vwOP, 1);
      vo3p0 = vfma_lane_f16(vo3p0, vi7x6789, vwOP, 1);
      vo4p0 = vfma_lane_f16(vo4p0, vi8x6789, vwOP, 1);


      float16x4_t vo0 = vmax_f16(vo0p0, vmin);
      float16x4_t vo1 = vmax_f16(vo1p0, vmin);
      float16x4_t vo2 = vmax_f16(vo2p0, vmin);
      float16x4_t vo3 = vmax_f16(vo3p0, vmin);
      float16x4_t vo4 = vmax_f16(vo4p0, vmin);

      vo0 = vmin_f16(vo0, vmax);
      vo1 = vmin_f16(vo1, vmax);
      vo2 = vmin_f16(vo2, vmax);
      vo3 = vmin_f16(vo3, vmax);
      vo4 = vmin_f16(vo4, vmax);

      if XNN_LIKELY(w & (4 * sizeof(__fp16))) {
        vst1_f16(o4, vo4); o4 += 4;
        vst1_f16(o3, vo3); o3 += 4;
        vst1_f16(o2, vo2); o2 += 4;
        vst1_f16(o1, vo1); o1 += 4;
        vst1_f16(o0, vo0); o0 += 4;
      } else {
        if (w & (2 * sizeof(__fp16))) {
          vst1_lane_u32((void*) o4, vreinterpret_u32_f16(vo4), 0); o4 += 2;
          vst1_lane_u32((void*) o3, vreinterpret_u32_f16(vo3), 0); o3 += 2;
          vst1_lane_u32((void*) o2, vreinterpret_u32_f16(vo2), 0); o2 += 2;
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0), 0); o0 += 2;

          vo0 = vext_f16(vo0, vo0, 2);
          vo1 = vext_f16(vo1, vo1, 2);
          vo2 = vext_f16(vo2, vo2, 2);
          vo3 = vext_f16(vo3, vo3, 2);
          vo4 = vext_f16(vo4, vo4, 2);
        }
        if (w & (1 * sizeof(__fp16))) {
          vst1_lane_f16(o4, vo4, 0); o4 += 1;
          vst1_lane_f16(o3, vo3, 0); o3 += 1;
          vst1_lane_f16(o2, vo2, 0); o2 += 1;
          vst1_lane_f16(o1, vo1, 0); o1 += 1;
          vst1_lane_f16(o0, vo0, 0); o0 += 1;
        }
      }
    }

    i0 = (const __fp16*) ((uintptr_t) i5 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i6 - input_decrement);
    i2 = (const __fp16*) ((uintptr_t) i1 + input_width);
    i3 = (const __fp16*) ((uintptr_t) i2 + input_width);
    i4 = (const __fp16*) ((uintptr_t) i3 + input_width);
    i5 = (const __fp16*) ((uintptr_t) i4 + input_width);
    i6 = (const __fp16*) ((uintptr_t) i5 + input_width);
    i7 = (const __fp16*) ((uintptr_t) i6 + input_width);
    i8 = (const __fp16*) ((uintptr_t) i7 + input_width);

    o0 = o4;
    o1 = (__fp16*) ((uintptr_t) o0 + input_width);
    o2 = (__fp16*) ((uintptr_t) o1 + input_width);
    o3 = (__fp16*) ((uintptr_t) o2 + input_width);
    o4 = (__fp16*) ((uintptr_t) o3 + input_width);

    output_height = doz(output_height, 5);
  } while (output_height != 0);
}
