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

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_3x8(
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
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  #if XNN_ARCH_ARM64
    const uint16x8x2_t vminmax = vld2q_dup_u16(&params->neonfp16arith_stride2.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vminmax.val[0]);
    const float16x8_t vmax = vreinterpretq_f16_u16(vminmax.val[1]);
  #else
    // vld2_dup is to work around aarch32 clang bug with vld1q_dup
    const uint16x4x2_t vminmax = vld2_dup_u16(&params->neonfp16arith_stride2.min);
    const float16x8_t vmin = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[0], vminmax.val[0]));
    const float16x8_t vmax = vreinterpretq_f16_u16(vcombine_u16(vminmax.val[1], vminmax.val[1]));
  #endif
  const uint16x8_t vmask_even = vld1q_u16(params->neonfp16arith_stride2.mask_even);
  const uint16x8_t vmask_odd = vld1q_u16(params->neonfp16arith_stride2.mask_odd);

  const uint16_t* w = (const uint16_t*) weights;
  const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));
  const float16x8_t vw89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w + 8));
  const float16x8_t vwGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(w + 16));
  const float16x4_t vwOP = vreinterpret_f16_u32(vld1_dup_u32((const void*) (w + 24)));

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 16 * sizeof(uint16_t));

  const uint16_t* i0 = zero;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_width);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_width);
  const uint16_t* i7 = (const uint16_t*) ((uintptr_t) i6 + input_width);
  const uint16_t* i8 = (const uint16_t*) ((uintptr_t) i7 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(uint16_t)) / 2, sizeof(uint16_t));

  uint16_t* o0 = output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_width);
  uint16_t* o2 = (uint16_t*) ((uintptr_t) o1 + output_width);

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

    float16x8_t vi0x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi4x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi5x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi6x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi7x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi8x02468ACE = vreinterpretq_f16_u16(vmovq_n_u16(0));

    float16x8_t vi0x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi1x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi2x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi3x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi4x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi5x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi6x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi7x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));
    float16x8_t vi8x13579BDF = vreinterpretq_f16_u16(vmovq_n_u16(0));

    uint16x8x2_t vi0xGIKMOQSUHJLNPRTV = vld2q_u16(i0); i0 += 16;
    uint16x8x2_t vi1xGIKMOQSUHJLNPRTV = vld2q_u16(i1); i1 += 16;
    uint16x8x2_t vi2xGIKMOQSUHJLNPRTV = vld2q_u16(i2); i2 += 16;
    uint16x8x2_t vi3xGIKMOQSUHJLNPRTV = vld2q_u16(i3); i3 += 16;
    uint16x8x2_t vi4xGIKMOQSUHJLNPRTV = vld2q_u16(i4); i4 += 16;
    uint16x8x2_t vi5xGIKMOQSUHJLNPRTV = vld2q_u16(i5); i5 += 16;
    uint16x8x2_t vi6xGIKMOQSUHJLNPRTV = vld2q_u16(i6); i6 += 16;
    uint16x8x2_t vi7xGIKMOQSUHJLNPRTV = vld2q_u16(i7); i7 += 16;
    uint16x8x2_t vi8xGIKMOQSUHJLNPRTV = vld2q_u16(i8); i8 += 16;

    size_t w = input_width;
    for (; w > 16 * sizeof(uint16_t); w -= 16 * sizeof(uint16_t)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo2p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vw01234567, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vw01234567, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]), vw89ABCDEF, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 2);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[0]), vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[0]), vwGHIJKLMN, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[0]), vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Right by 2 column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vw01234567, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vw01234567, 4);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[1]), vw89ABCDEF, 6);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[1]), vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[1]), vwGHIJKLMN, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[1]), vwGHIJKLMN, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[1]), vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[1]), vwOP, 0);
      #endif
      // Left by 2 column
      const float16x8_t vi0xEGIKMOQS = vextq_f16(vi0x02468ACE, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi0x02468ACE = vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi1xEGIKMOQS = vextq_f16(vi1x02468ACE, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi1x02468ACE = vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi2xEGIKMOQS = vextq_f16(vi2x02468ACE, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi2x02468ACE = vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi3xEGIKMOQS = vextq_f16(vi3x02468ACE, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi3x02468ACE = vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi4xEGIKMOQS = vextq_f16(vi4x02468ACE, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi4x02468ACE = vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi5xEGIKMOQS = vextq_f16(vi5x02468ACE, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi5x02468ACE = vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi6xEGIKMOQS = vextq_f16(vi6x02468ACE, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi6x02468ACE = vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi7xEGIKMOQS = vextq_f16(vi7x02468ACE, vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi7x02468ACE = vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[0]);
      const float16x8_t vi8xEGIKMOQS = vextq_f16(vi8x02468ACE, vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[0]), 7);
      vi8x02468ACE = vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[0]);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xEGIKMOQS, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xEGIKMOQS, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xEGIKMOQS, vw01234567, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xEGIKMOQS, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xEGIKMOQS, vw01234567, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xEGIKMOQS, vw01234567, 6);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi6xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi8xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Left by 1 column, s1
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi0x13579BDF = vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi1x13579BDF = vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi2x13579BDF = vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi3xFHJLNPRT = vextq_f16(vi3x13579BDF, vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi3x13579BDF = vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi4xFHJLNPRT = vextq_f16(vi4x13579BDF, vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi4x13579BDF = vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi5xFHJLNPRT = vextq_f16(vi5x13579BDF, vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi5x13579BDF = vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi6xFHJLNPRT = vextq_f16(vi6x13579BDF, vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi6x13579BDF = vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi7xFHJLNPRT = vextq_f16(vi7x13579BDF, vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi7x13579BDF = vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[1]);
      const float16x8_t vi8xFHJLNPRT = vextq_f16(vi8x13579BDF, vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[1]), 7);
      vi8x13579BDF = vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[1]);

      const uint16x8x2_t vi0xWYacegikXZbdfhjl = vld2q_u16(i0); i0 += 16;
      const uint16x8x2_t vi1xWYacegikXZbdfhjl = vld2q_u16(i1); i1 += 16;
      const uint16x8x2_t vi2xWYacegikXZbdfhjl = vld2q_u16(i2); i2 += 16;
      const uint16x8x2_t vi3xWYacegikXZbdfhjl = vld2q_u16(i3); i3 += 16;
      const uint16x8x2_t vi4xWYacegikXZbdfhjl = vld2q_u16(i4); i4 += 16;
      const uint16x8x2_t vi5xWYacegikXZbdfhjl = vld2q_u16(i5); i5 += 16;
      const uint16x8x2_t vi6xWYacegikXZbdfhjl = vld2q_u16(i6); i6 += 16;
      const uint16x8x2_t vi7xWYacegikXZbdfhjl = vld2q_u16(i7); i7 += 16;
      const uint16x8x2_t vi8xWYacegikXZbdfhjl = vld2q_u16(i8); i8 += 16;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xFHJLNPRT, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xFHJLNPRT, vw01234567, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xFHJLNPRT, vw01234567, 2);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xFHJLNPRT, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xFHJLNPRT, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xFHJLNPRT, vw01234567, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi6xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi8xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Right by 1 column
      const float16x8_t vi0xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi0xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi0xWYacegikXZbdfhjl.val[0]), 1);
      vi0xGIKMOQSUHJLNPRTV = vi0xWYacegikXZbdfhjl;
      const float16x8_t vi1xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi1xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi1xWYacegikXZbdfhjl.val[0]), 1);
      vi1xGIKMOQSUHJLNPRTV = vi1xWYacegikXZbdfhjl;
      const float16x8_t vi2xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi2xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi2xWYacegikXZbdfhjl.val[0]), 1);
      vi2xGIKMOQSUHJLNPRTV = vi2xWYacegikXZbdfhjl;
      const float16x8_t vi3xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi3xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi3xWYacegikXZbdfhjl.val[0]), 1);
      vi3xGIKMOQSUHJLNPRTV = vi3xWYacegikXZbdfhjl;
      const float16x8_t vi4xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi4xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi4xWYacegikXZbdfhjl.val[0]), 1);
      vi4xGIKMOQSUHJLNPRTV = vi4xWYacegikXZbdfhjl;
      const float16x8_t vi5xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi5xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi5xWYacegikXZbdfhjl.val[0]), 1);
      vi5xGIKMOQSUHJLNPRTV = vi5xWYacegikXZbdfhjl;
      const float16x8_t vi6xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi6xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi6xWYacegikXZbdfhjl.val[0]), 1);
      vi6xGIKMOQSUHJLNPRTV = vi6xWYacegikXZbdfhjl;
      const float16x8_t vi7xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi7xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi7xWYacegikXZbdfhjl.val[0]), 1);
      vi7xGIKMOQSUHJLNPRTV = vi7xWYacegikXZbdfhjl;
      const float16x8_t vi8xIKMOQSUW = vextq_f16(vreinterpretq_f16_u16(vi8xGIKMOQSUHJLNPRTV.val[0]), vreinterpretq_f16_u16(vi8xWYacegikXZbdfhjl.val[0]), 1);
      vi8xGIKMOQSUHJLNPRTV = vi8xWYacegikXZbdfhjl;

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xIKMOQSUW, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xIKMOQSUW, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xIKMOQSUW, vw01234567, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi6xIKMOQSUW, vwOP, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xIKMOQSUW, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vi8xIKMOQSUW, vwOP, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xIKMOQSUW, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);
      float16x8_t vo2 = vmaxq_f16(vo2p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);
      vo2 = vminq_f16(vo2, vmax);

      vst1q_u16(o2, vreinterpretq_u16_f16(vo2)); o2 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
      vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
    }

    // Last block has 1-16 pixels to process.
    assert(w <= 16 * sizeof(uint16_t));
    assert(w >= 1 * sizeof(uint16_t));
    {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo1p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);
      float16x8_t vo2p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8_t vi0xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi0xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi1xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi1xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi2xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi2xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi3xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi3xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi4xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi4xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi5xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi5xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi6xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi6xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi7xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi7xGIKMOQSUHJLNPRTV.val[0]));
      const float16x8_t vi8xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vi8xGIKMOQSUHJLNPRTV.val[0]));

      const float16x8_t vi0xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi0xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi1xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi1xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi2xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi2xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi3xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi3xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi4xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi4xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi5xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi5xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi6xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi6xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi7xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi7xGIKMOQSUHJLNPRTV.val[1]));
      const float16x8_t vi8xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd, vi8xGIKMOQSUHJLNPRTV.val[1]));

      // Center column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xGIKMOQSU, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xGIKMOQSU, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xGIKMOQSU, vw01234567, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xGIKMOQSU, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xGIKMOQSU, vw01234567, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xGIKMOQSU, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xGIKMOQSU, vw89ABCDEF, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xGIKMOQSU, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xGIKMOQSU, vw89ABCDEF, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xGIKMOQSU, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xGIKMOQSU, vw89ABCDEF, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xGIKMOQSU, vget_low_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xGIKMOQSU, vw89ABCDEF, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xGIKMOQSU, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xGIKMOQSU, vw89ABCDEF, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xGIKMOQSU, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xGIKMOQSU, vw89ABCDEF, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xGIKMOQSU, vget_high_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xGIKMOQSU, vwGHIJKLMN, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xGIKMOQSU, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xGIKMOQSU, vwGHIJKLMN, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xGIKMOQSU, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xGIKMOQSU, vwGHIJKLMN, 2);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xGIKMOQSU, vget_low_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xGIKMOQSU, vwGHIJKLMN, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xGIKMOQSU, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi6xGIKMOQSU, vwGHIJKLMN, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xGIKMOQSU, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi8xGIKMOQSU, vwGHIJKLMN, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xGIKMOQSU, vget_high_f16(vwGHIJKLMN), 3);
      #endif
      // Right by 1 column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xHJLNPRTV, vw01234567, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xHJLNPRTV, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xHJLNPRTV, vw01234567, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xHJLNPRTV, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xHJLNPRTV, vw01234567, 4);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xHJLNPRTV, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xHJLNPRTV, vw89ABCDEF, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xHJLNPRTV, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xHJLNPRTV, vw89ABCDEF, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xHJLNPRTV, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xHJLNPRTV, vw89ABCDEF, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xHJLNPRTV, vget_low_f16(vw89ABCDEF), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xHJLNPRTV, vw89ABCDEF, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xHJLNPRTV, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xHJLNPRTV, vw89ABCDEF, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xHJLNPRTV, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xHJLNPRTV, vw89ABCDEF, 6);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xHJLNPRTV, vget_high_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xHJLNPRTV, vwGHIJKLMN, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xHJLNPRTV, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xHJLNPRTV, vwGHIJKLMN, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xHJLNPRTV, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xHJLNPRTV, vwGHIJKLMN, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xHJLNPRTV, vget_low_f16(vwGHIJKLMN), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xHJLNPRTV, vwOP, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xHJLNPRTV, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi6xHJLNPRTV, vwOP, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xHJLNPRTV, vwOP, 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vi8xHJLNPRTV, vwOP, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xHJLNPRTV, vwOP, 0);
      #endif
      // Left by 2 columns
      const float16x8_t vi0xEGIKMOQS = vextq_f16(vi0x02468ACE, vi0xGIKMOQSU, 7);
      const float16x8_t vi1xEGIKMOQS = vextq_f16(vi1x02468ACE, vi1xGIKMOQSU, 7);
      const float16x8_t vi2xEGIKMOQS = vextq_f16(vi2x02468ACE, vi2xGIKMOQSU, 7);
      const float16x8_t vi3xEGIKMOQS = vextq_f16(vi3x02468ACE, vi3xGIKMOQSU, 7);
      const float16x8_t vi4xEGIKMOQS = vextq_f16(vi4x02468ACE, vi4xGIKMOQSU, 7);
      const float16x8_t vi5xEGIKMOQS = vextq_f16(vi5x02468ACE, vi5xGIKMOQSU, 7);
      const float16x8_t vi6xEGIKMOQS = vextq_f16(vi6x02468ACE, vi6xGIKMOQSU, 7);
      const float16x8_t vi7xEGIKMOQS = vextq_f16(vi7x02468ACE, vi7xGIKMOQSU, 7);
      const float16x8_t vi8xEGIKMOQS = vextq_f16(vi8x02468ACE, vi8xGIKMOQSU, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xEGIKMOQS, vw01234567, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xEGIKMOQS, vw01234567, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xEGIKMOQS, vw01234567, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xEGIKMOQS, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xEGIKMOQS, vw01234567, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xEGIKMOQS, vw01234567, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xEGIKMOQS, vw01234567, 6);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xEGIKMOQS, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xEGIKMOQS, vw89ABCDEF, 3);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xEGIKMOQS, vget_low_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xEGIKMOQS, vwGHIJKLMN, 0);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xEGIKMOQS, vget_low_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi6xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi8xEGIKMOQS, vwGHIJKLMN, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xEGIKMOQS, vget_high_f16(vwGHIJKLMN), 1);
      #endif
      // Left by 1 column
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vi0xHJLNPRTV, 7);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vi1xHJLNPRTV, 7);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vi2xHJLNPRTV, 7);
      const float16x8_t vi3xFHJLNPRT = vextq_f16(vi3x13579BDF, vi3xHJLNPRTV, 7);
      const float16x8_t vi4xFHJLNPRT = vextq_f16(vi4x13579BDF, vi4xHJLNPRTV, 7);
      const float16x8_t vi5xFHJLNPRT = vextq_f16(vi5x13579BDF, vi5xHJLNPRTV, 7);
      const float16x8_t vi6xFHJLNPRT = vextq_f16(vi6x13579BDF, vi6xHJLNPRTV, 7);
      const float16x8_t vi7xFHJLNPRT = vextq_f16(vi7x13579BDF, vi7xHJLNPRTV, 7);
      const float16x8_t vi8xFHJLNPRT = vextq_f16(vi8x13579BDF, vi8xHJLNPRTV, 7);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xFHJLNPRT, vw01234567, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xFHJLNPRT, vw01234567, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xFHJLNPRT, vw01234567, 2);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xFHJLNPRT, vget_low_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xFHJLNPRT, vw01234567, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xFHJLNPRT, vw01234567, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xFHJLNPRT, vw01234567, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xFHJLNPRT, vw89ABCDEF, 4);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xFHJLNPRT, vget_high_f16(vw89ABCDEF), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xFHJLNPRT, vwGHIJKLMN, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xFHJLNPRT, vget_low_f16(vwGHIJKLMN), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi4xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi6xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi8xFHJLNPRT, vwGHIJKLMN, 6);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xFHJLNPRT, vget_high_f16(vwGHIJKLMN), 2);
      #endif
      // Right by 2 columns
      const float16x8_t vzero = vreinterpretq_f16_u16(vmovq_n_u16(0));
      const float16x8_t vi0xIKMOQSUW = vextq_f16(vi0xGIKMOQSU, vzero, 1);
      const float16x8_t vi1xIKMOQSUW = vextq_f16(vi1xGIKMOQSU, vzero, 1);
      const float16x8_t vi2xIKMOQSUW = vextq_f16(vi2xGIKMOQSU, vzero, 1);
      const float16x8_t vi3xIKMOQSUW = vextq_f16(vi3xGIKMOQSU, vzero, 1);
      const float16x8_t vi4xIKMOQSUW = vextq_f16(vi4xGIKMOQSU, vzero, 1);
      const float16x8_t vi5xIKMOQSUW = vextq_f16(vi5xGIKMOQSU, vzero, 1);
      const float16x8_t vi6xIKMOQSUW = vextq_f16(vi6xGIKMOQSU, vzero, 1);
      const float16x8_t vi7xIKMOQSUW = vextq_f16(vi7xGIKMOQSU, vzero, 1);
      const float16x8_t vi8xIKMOQSUW = vextq_f16(vi8xGIKMOQSU, vzero, 1);

      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xIKMOQSUW, vw01234567, 5);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi2xIKMOQSUW, vw01234567, 5);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi2xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi4xIKMOQSUW, vw01234567, 5);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi4xIKMOQSUW, vget_high_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi1xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi1xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi3xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi3xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi5xIKMOQSUW, vw89ABCDEF, 2);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi5xIKMOQSUW, vget_low_f16(vw89ABCDEF), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi2xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi2xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi4xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi4xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi6xIKMOQSUW, vw89ABCDEF, 7);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi6xIKMOQSUW, vget_high_f16(vw89ABCDEF), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi3xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi3xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_laneq_f16(vo1p0, vi5xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi5xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_laneq_f16(vo2p0, vi7xIKMOQSUW, vwGHIJKLMN, 4);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi7xIKMOQSUW, vget_high_f16(vwGHIJKLMN), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi4xIKMOQSUW, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo1p0 = vfmaq_lane_f16(vo1p0, vi6xIKMOQSUW, vwOP, 1);
      #else
        vo1p0 = vmlaq_lane_f16(vo1p0, vi6xIKMOQSUW, vwOP, 1);
      #endif
      #if XNN_ARCH_ARM64
        vo2p0 = vfmaq_lane_f16(vo2p0, vi8xIKMOQSUW, vwOP, 1);
      #else
        vo2p0 = vmlaq_lane_f16(vo2p0, vi8xIKMOQSUW, vwOP, 1);
      #endif

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);
      float16x8_t vo1 = vmaxq_f16(vo1p0, vmin);
      float16x8_t vo2 = vmaxq_f16(vo2p0, vmin);

      vo0 = vminq_f16(vo0, vmax);
      vo1 = vminq_f16(vo1, vmax);
      vo2 = vminq_f16(vo2, vmax);

      const size_t w_tmp = (w + 1 * sizeof(uint16_t)) / (2 * sizeof(uint16_t));

      if XNN_LIKELY(w_tmp == 8) {
        vst1q_u16(o2, vreinterpretq_u16_f16(vo2)); o2 += 8;
        vst1q_u16(o1, vreinterpretq_u16_f16(vo1)); o1 += 8;
        vst1q_u16(o0, vreinterpretq_u16_f16(vo0)); o0 += 8;
      } else {
        float16x4_t vo2_lo = vget_low_f16(vo2);
        float16x4_t vo1_lo = vget_low_f16(vo1);
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w_tmp & 4) {
         vst1_u16(o2, vreinterpret_u16_f16(vo2_lo)); o2 += 4;
         vst1_u16(o1, vreinterpret_u16_f16(vo1_lo)); o1 += 4;
         vst1_u16(o0, vreinterpret_u16_f16(vo0_lo)); o0 += 4;

          vo2_lo = vget_high_f16(vo2);
          vo1_lo = vget_high_f16(vo1);
          vo0_lo = vget_high_f16(vo0);
        }
        if (w_tmp & 2) {
          vst1_lane_u32((void*) o2, vreinterpret_u32_f16(vo2_lo), 0); o2 += 2;
          vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vo1_lo), 0); o1 += 2;
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
          vo1_lo = vext_f16(vo1_lo, vo1_lo, 2);
          vo2_lo = vext_f16(vo2_lo, vo2_lo, 2);
        }
        if (w_tmp & 1) {
          vst1_lane_u16(o2, vreinterpret_u16_f16(vo2_lo), 0); o2 += 1;
          vst1_lane_u16(o1, vreinterpret_u16_f16(vo1_lo), 0); o1 += 1;
          vst1_lane_u16(o0, vreinterpret_u16_f16(vo0_lo), 0); o0 += 1;
        }
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i6 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i7 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i8 - input_decrement);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);
    i5 = (const uint16_t*) ((uintptr_t) i4 + input_width);
    i6 = (const uint16_t*) ((uintptr_t) i5 + input_width);
    i7 = (const uint16_t*) ((uintptr_t) i6 + input_width);
    i8 = (const uint16_t*) ((uintptr_t) i7 + input_width);

    o0 = o2;
    o1 = (uint16_t*) ((uintptr_t) o0 + output_width);
    o2 = (uint16_t*) ((uintptr_t) o1 + output_width);

    output_height = doz(output_height, 3);
    padded_input_height = doz(padded_input_height, 6);
  } while (output_height != 0);
}
