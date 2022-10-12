// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>


void xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x8_acc4(
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
  assert(padding_top <= 1);

  const uint16x8_t vmask_even = vld1q_u16(params->neonfp16arith.mask_even);
  const uint16x8_t vmask_odd  = vld1q_u16(params->neonfp16arith.mask_odd);
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith.max));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith.min));

  const __fp16* w0 = (const __fp16*)weights;
  const float16x8_t vw01234567 = vld1q_f16(w0);
  const float16x4_t vw89 = vreinterpret_f16_u32(vld1_dup_u32((const void*)(w0 + 8)));

  const size_t input_decrement = round_down_po2(input_width, 8 /* SIMD output width */ * 2 /* subsampling */ * sizeof(__fp16));

  const __fp16* i0 = (const __fp16*) ((uintptr_t) input - ((-padding_top) & input_width));
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_width);

  __fp16* o0 = output;

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }

    float16x8_t vi0x13579BDF = vmovq_n_f16(0);
    float16x8_t vi1x13579BDF = vmovq_n_f16(0);
    float16x8_t vi2x13579BDF = vmovq_n_f16(0);

    size_t w = input_width;
    for (; w >= 16 * sizeof(__fp16); w -= 16 * sizeof(__fp16)) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8x2_t vi0xGIKMOQSUHJLNPRTV = vld2q_f16(i0); i0 += 16;
      const float16x8x2_t vi1xGIKMOQSUHJLNPRTV = vld2q_f16(i1); i1 += 16;
      const float16x8x2_t vi2xGIKMOQSUHJLNPRTV = vld2q_f16(i2); i2 += 16;

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0xGIKMOQSUHJLNPRTV.val[0], vget_low_f16(vw01234567), 2);
      float16x8_t vo0p2 = vmulq_lane_f16(vi1xGIKMOQSUHJLNPRTV.val[0], vget_high_f16(vw01234567), 1);
      float16x8_t vo0p3 = vmulq_lane_f16(vi2xGIKMOQSUHJLNPRTV.val[0], vw89, 0);
      // Left column
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vi0xGIKMOQSUHJLNPRTV.val[1], 7);
      vi0x13579BDF = vi0xGIKMOQSUHJLNPRTV.val[1];
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vi1xGIKMOQSUHJLNPRTV.val[1], 7);
      vi1x13579BDF = vi1xGIKMOQSUHJLNPRTV.val[1];
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vi2xGIKMOQSUHJLNPRTV.val[1], 7);
      vi2x13579BDF = vi2xGIKMOQSUHJLNPRTV.val[1];

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0xFHJLNPRT, vw01234567, 1);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0xFHJLNPRT, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p2 = vfmaq_laneq_f16(vo0p2, vi1xFHJLNPRT, vw01234567, 4);
      #else
        vo0p2 = vmlaq_lane_f16(vo0p2, vi1xFHJLNPRT, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p3 = vfmaq_laneq_f16(vo0p3, vi2xFHJLNPRT, vw01234567, 7);
      #else
        vo0p3 = vmlaq_lane_f16(vo0p3, vi2xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      // Right column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xGIKMOQSUHJLNPRTV.val[1], vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xGIKMOQSUHJLNPRTV.val[1], vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1xGIKMOQSUHJLNPRTV.val[1], vw01234567, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1xGIKMOQSUHJLNPRTV.val[1], vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p2 = vfmaq_lane_f16(vo0p2, vi2xGIKMOQSUHJLNPRTV.val[1], vw89, 1);
      #else
        vo0p2 = vmlaq_lane_f16(vo0p2, vi2xGIKMOQSUHJLNPRTV.val[1], vw89, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo0p2 = vaddq_f16(vo0p2, vo0p3);
      vo0p0 = vaddq_f16(vo0p0, vo0p2);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      vst1q_f16(o0, vo0); o0 += 8;
    }

    // Last block has 0-15 pixels to process.
    assert(w < 16 * sizeof(__fp16));
    if XNN_LIKELY(w != 0) {
      float16x8_t vo0p0 = vdupq_lane_f16(vget_low_f16(vw01234567), 0);

      const float16x8x2_t vi0xGIKMOQSUHJLNPRTV = vld2q_f16(i0);
      const float16x8x2_t vi1xGIKMOQSUHJLNPRTV = vld2q_f16(i1);
      const float16x8x2_t vi2xGIKMOQSUHJLNPRTV = vld2q_f16(i2);

      const float16x8_t vi0xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi0xGIKMOQSUHJLNPRTV.val[0])));
      const float16x8_t vi0xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd,  vreinterpretq_u16_f16(vi0xGIKMOQSUHJLNPRTV.val[1])));
      const float16x8_t vi1xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi1xGIKMOQSUHJLNPRTV.val[0])));
      const float16x8_t vi1xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd,  vreinterpretq_u16_f16(vi1xGIKMOQSUHJLNPRTV.val[1])));
      const float16x8_t vi2xGIKMOQSU = vreinterpretq_f16_u16(vandq_u16(vmask_even, vreinterpretq_u16_f16(vi2xGIKMOQSUHJLNPRTV.val[0])));
      const float16x8_t vi2xHJLNPRTV = vreinterpretq_f16_u16(vandq_u16(vmask_odd,  vreinterpretq_u16_f16(vi2xGIKMOQSUHJLNPRTV.val[1])));

      // Center column
      float16x8_t vo0p1 = vmulq_lane_f16(vi0xGIKMOQSU, vget_low_f16(vw01234567), 2);
      float16x8_t vo0p2 = vmulq_lane_f16(vi1xGIKMOQSU, vget_high_f16(vw01234567), 1);
      float16x8_t vo0p3 = vmulq_lane_f16(vi2xGIKMOQSU, vw89, 0);
      // Left column
      const float16x8_t vi0xFHJLNPRT = vextq_f16(vi0x13579BDF, vi0xHJLNPRTV, 7);
      const float16x8_t vi1xFHJLNPRT = vextq_f16(vi1x13579BDF, vi1xHJLNPRTV, 7);
      const float16x8_t vi2xFHJLNPRT = vextq_f16(vi2x13579BDF, vi2xHJLNPRTV, 7);

      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi0xFHJLNPRT, vw01234567, 1);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi0xFHJLNPRT, vget_low_f16(vw01234567), 1);
      #endif
      #if XNN_ARCH_ARM64
        vo0p2 = vfmaq_laneq_f16(vo0p2, vi1xFHJLNPRT, vw01234567, 4);
      #else
        vo0p2 = vmlaq_lane_f16(vo0p2, vi1xFHJLNPRT, vget_high_f16(vw01234567), 0);
      #endif
      #if XNN_ARCH_ARM64
        vo0p3 = vfmaq_laneq_f16(vo0p3, vi2xFHJLNPRT, vw01234567, 7);
      #else
        vo0p3 = vmlaq_lane_f16(vo0p3, vi2xFHJLNPRT, vget_high_f16(vw01234567), 3);
      #endif
      // Right column
      #if XNN_ARCH_ARM64
        vo0p0 = vfmaq_laneq_f16(vo0p0, vi0xHJLNPRTV, vw01234567, 3);
      #else
        vo0p0 = vmlaq_lane_f16(vo0p0, vi0xHJLNPRTV, vget_low_f16(vw01234567), 3);
      #endif
      #if XNN_ARCH_ARM64
        vo0p1 = vfmaq_laneq_f16(vo0p1, vi1xHJLNPRTV, vw01234567, 6);
      #else
        vo0p1 = vmlaq_lane_f16(vo0p1, vi1xHJLNPRTV, vget_high_f16(vw01234567), 2);
      #endif
      #if XNN_ARCH_ARM64
        vo0p2 = vfmaq_lane_f16(vo0p2, vi2xHJLNPRTV, vw89, 1);
      #else
        vo0p2 = vmlaq_lane_f16(vo0p2, vi2xHJLNPRTV, vw89, 1);
      #endif
      vo0p0 = vaddq_f16(vo0p0, vo0p1);
      vo0p2 = vaddq_f16(vo0p2, vo0p3);
      vo0p0 = vaddq_f16(vo0p0, vo0p2);

      float16x8_t vo0 = vmaxq_f16(vo0p0, vmin);

      vo0 = vminq_f16(vo0, vmax);

      w += 1 * sizeof(__fp16);

      if XNN_LIKELY(w == 16 * sizeof(__fp16)) {
        vst1q_f16(o0, vo0); o0 += 8;
      } else {
        float16x4_t vo0_lo = vget_low_f16(vo0);

        if (w & (8 * sizeof(__fp16))) {
         vst1_f16(o0, vo0_lo); o0 += 4;

          vo0_lo = vget_high_f16(vo0);
        }
        if (w & (4 * sizeof(__fp16))) {
          vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vo0_lo), 0); o0 += 2;

          vo0_lo = vext_f16(vo0_lo, vo0_lo, 2);
        }
        if (w & (2 * sizeof(__fp16))) {
          vst1_lane_f16(o0, vo0_lo, 0); o0 += 1;
        }
      }
    }

    i0 = (const __fp16*) ((uintptr_t) i2 - input_decrement);
    i1 = (const __fp16*) ((uintptr_t) i0 + input_width);
    i2 = (const __fp16*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
