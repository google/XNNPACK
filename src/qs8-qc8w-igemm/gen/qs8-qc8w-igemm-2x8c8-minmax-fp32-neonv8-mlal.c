// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c8-neon-mull.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  do {
    int32x4_t vacc0x0 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x1 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x2 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x3 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x4 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x5 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x6 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc0x7 = vld1q_lane_s32(w, vmovq_n_s32(0), 0); w = (const int32_t*) w + 1;
    int32x4_t vacc1x0 = vacc0x0;
    int32x4_t vacc1x1 = vacc0x1;
    int32x4_t vacc1x2 = vacc0x2;
    int32x4_t vacc1x3 = vacc0x3;
    int32x4_t vacc1x4 = vacc0x4;
    int32x4_t vacc1x5 = vacc0x5;
    int32x4_t vacc1x6 = vacc0x6;
    int32x4_t vacc1x7 = vacc0x7;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      // 2x partial unrolled loop to load 16 bytes at a time using MLA.
      while (k >= 16 * sizeof(int8_t)) {
        const int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        const int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
        const int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb5x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb6x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb7x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        const int8x8_t vb0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x0 = vmull_s8(vb0x0, va0x0);
        int16x8_t vprod1x0 = vmull_s8(vb0x0, va1x0);
        vprod0x0 = vmlal_s8(vprod0x0, vb0x1, va0x1);
        vprod1x0 = vmlal_s8(vprod1x0, vb0x1, va1x1);
        vacc0x0 = vpadalq_s16(vacc0x0, vprod0x0);
        vacc1x0 = vpadalq_s16(vacc1x0, vprod1x0);
        const int8x8_t vb1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x1 = vmull_s8(vb1x0, va0x0);
        int16x8_t vprod1x1 = vmull_s8(vb1x0, va1x0);
        vprod0x1 = vmlal_s8(vprod0x1, vb1x1, va0x1);
        vprod1x1 = vmlal_s8(vprod1x1, vb1x1, va1x1);
        vacc0x1 = vpadalq_s16(vacc0x1, vprod0x1);
        vacc1x1 = vpadalq_s16(vacc1x1, vprod1x1);
        const int8x8_t vb2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x2 = vmull_s8(vb2x0, va0x0);
        int16x8_t vprod1x2 = vmull_s8(vb2x0, va1x0);
        vprod0x2 = vmlal_s8(vprod0x2, vb2x1, va0x1);
        vprod1x2 = vmlal_s8(vprod1x2, vb2x1, va1x1);
        vacc0x2 = vpadalq_s16(vacc0x2, vprod0x2);
        vacc1x2 = vpadalq_s16(vacc1x2, vprod1x2);
        const int8x8_t vb3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x3 = vmull_s8(vb3x0, va0x0);
        int16x8_t vprod1x3 = vmull_s8(vb3x0, va1x0);
        vprod0x3 = vmlal_s8(vprod0x3, vb3x1, va0x1);
        vprod1x3 = vmlal_s8(vprod1x3, vb3x1, va1x1);
        vacc0x3 = vpadalq_s16(vacc0x3, vprod0x3);
        vacc1x3 = vpadalq_s16(vacc1x3, vprod1x3);
        const int8x8_t vb4x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x4 = vmull_s8(vb4x0, va0x0);
        int16x8_t vprod1x4 = vmull_s8(vb4x0, va1x0);
        vprod0x4 = vmlal_s8(vprod0x4, vb4x1, va0x1);
        vprod1x4 = vmlal_s8(vprod1x4, vb4x1, va1x1);
        vacc0x4 = vpadalq_s16(vacc0x4, vprod0x4);
        vacc1x4 = vpadalq_s16(vacc1x4, vprod1x4);
        const int8x8_t vb5x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x5 = vmull_s8(vb5x0, va0x0);
        int16x8_t vprod1x5 = vmull_s8(vb5x0, va1x0);
        vprod0x5 = vmlal_s8(vprod0x5, vb5x1, va0x1);
        vprod1x5 = vmlal_s8(vprod1x5, vb5x1, va1x1);
        vacc0x5 = vpadalq_s16(vacc0x5, vprod0x5);
        vacc1x5 = vpadalq_s16(vacc1x5, vprod1x5);
        const int8x8_t vb6x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x6 = vmull_s8(vb6x0, va0x0);
        int16x8_t vprod1x6 = vmull_s8(vb6x0, va1x0);
        vprod0x6 = vmlal_s8(vprod0x6, vb6x1, va0x1);
        vprod1x6 = vmlal_s8(vprod1x6, vb6x1, va1x1);
        vacc0x6 = vpadalq_s16(vacc0x6, vprod0x6);
        vacc1x6 = vpadalq_s16(vacc1x6, vprod1x6);
        const int8x8_t vb7x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        int16x8_t vprod0x7 = vmull_s8(vb7x0, va0x0);
        int16x8_t vprod1x7 = vmull_s8(vb7x0, va1x0);
        vprod0x7 = vmlal_s8(vprod0x7, vb7x1, va0x1);
        vprod1x7 = vmlal_s8(vprod1x7, vb7x1, va1x1);
        vacc0x7 = vpadalq_s16(vacc0x7, vprod0x7);
        vacc1x7 = vpadalq_s16(vacc1x7, vprod1x7);

        k -= 16 * sizeof(int8_t);
      }

      // Handle 8 bytes at a time using MUL.
      if (k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;
        const int8x8_t va1 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x0 = vmull_s8(vb0, va0);
        const int16x8_t vprod1x0 = vmull_s8(vb0, va1);
        vacc0x0 = vpadalq_s16(vacc0x0, vprod0x0);
        vacc1x0 = vpadalq_s16(vacc1x0, vprod1x0);
        const int8x8_t vb1 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x1 = vmull_s8(vb1, va0);
        const int16x8_t vprod1x1 = vmull_s8(vb1, va1);
        vacc0x1 = vpadalq_s16(vacc0x1, vprod0x1);
        vacc1x1 = vpadalq_s16(vacc1x1, vprod1x1);
        const int8x8_t vb2 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x2 = vmull_s8(vb2, va0);
        const int16x8_t vprod1x2 = vmull_s8(vb2, va1);
        vacc0x2 = vpadalq_s16(vacc0x2, vprod0x2);
        vacc1x2 = vpadalq_s16(vacc1x2, vprod1x2);
        const int8x8_t vb3 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x3 = vmull_s8(vb3, va0);
        const int16x8_t vprod1x3 = vmull_s8(vb3, va1);
        vacc0x3 = vpadalq_s16(vacc0x3, vprod0x3);
        vacc1x3 = vpadalq_s16(vacc1x3, vprod1x3);
        const int8x8_t vb4 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x4 = vmull_s8(vb4, va0);
        const int16x8_t vprod1x4 = vmull_s8(vb4, va1);
        vacc0x4 = vpadalq_s16(vacc0x4, vprod0x4);
        vacc1x4 = vpadalq_s16(vacc1x4, vprod1x4);
        const int8x8_t vb5 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x5 = vmull_s8(vb5, va0);
        const int16x8_t vprod1x5 = vmull_s8(vb5, va1);
        vacc0x5 = vpadalq_s16(vacc0x5, vprod0x5);
        vacc1x5 = vpadalq_s16(vacc1x5, vprod1x5);
        const int8x8_t vb6 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x6 = vmull_s8(vb6, va0);
        const int16x8_t vprod1x6 = vmull_s8(vb6, va1);
        vacc0x6 = vpadalq_s16(vacc0x6, vprod0x6);
        vacc1x6 = vpadalq_s16(vacc1x6, vprod1x6);
        const int8x8_t vb7 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int16x8_t vprod0x7 = vmull_s8(vb7, va0);
        const int16x8_t vprod1x7 = vmull_s8(vb7, va1);
        vacc0x7 = vpadalq_s16(vacc0x7, vprod0x7);
        vacc1x7 = vpadalq_s16(vacc1x7, vprod1x7);

        k -= 8 * sizeof(int8_t);
      }

      p -= 2 * sizeof(void*);
    } while (p != 0);

#if XNN_ARCH_ARM64
    const int32x4_t vsum0x01 = vpaddq_s32(vacc0x0, vacc0x1);
    const int32x4_t vsum0x23 = vpaddq_s32(vacc0x2, vacc0x3);
    const int32x4_t vsum0x45 = vpaddq_s32(vacc0x4, vacc0x5);
    const int32x4_t vsum0x67 = vpaddq_s32(vacc0x6, vacc0x7);
    const int32x4_t vsum1x01 = vpaddq_s32(vacc1x0, vacc1x1);
    const int32x4_t vsum1x23 = vpaddq_s32(vacc1x2, vacc1x3);
    const int32x4_t vsum1x45 = vpaddq_s32(vacc1x4, vacc1x5);
    const int32x4_t vsum1x67 = vpaddq_s32(vacc1x6, vacc1x7);

    int32x4_t vacc0x0123 = vpaddq_s32(vsum0x01, vsum0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vsum0x45, vsum0x67);
    int32x4_t vacc1x0123 = vpaddq_s32(vsum1x01, vsum1x23);
    int32x4_t vacc1x4567 = vpaddq_s32(vsum1x45, vsum1x67);
#else
    const int32x2_t vpsum0x0 = vadd_s32(vget_low_s32(vacc0x0), vget_high_s32(vacc0x0));
    const int32x2_t vpsum0x1 = vadd_s32(vget_low_s32(vacc0x1), vget_high_s32(vacc0x1));
    const int32x2_t vpsum0x2 = vadd_s32(vget_low_s32(vacc0x2), vget_high_s32(vacc0x2));
    const int32x2_t vpsum0x3 = vadd_s32(vget_low_s32(vacc0x3), vget_high_s32(vacc0x3));
    const int32x2_t vsum0x01 = vpadd_s32(vpsum0x0, vpsum0x1);
    const int32x2_t vsum0x23 = vpadd_s32(vpsum0x2, vpsum0x3);
    int32x4_t vacc0x0123 = vcombine_s32(vsum0x01, vsum0x23 );
    const int32x2_t vpsum0x4 = vadd_s32(vget_low_s32(vacc0x4), vget_high_s32(vacc0x4));
    const int32x2_t vpsum0x5 = vadd_s32(vget_low_s32(vacc0x5), vget_high_s32(vacc0x5));
    const int32x2_t vpsum0x6 = vadd_s32(vget_low_s32(vacc0x6), vget_high_s32(vacc0x6));
    const int32x2_t vpsum0x7 = vadd_s32(vget_low_s32(vacc0x7), vget_high_s32(vacc0x7));
    const int32x2_t vsum0x45 = vpadd_s32(vpsum0x4, vpsum0x5);
    const int32x2_t vsum0x67 = vpadd_s32(vpsum0x6, vpsum0x7);
    int32x4_t vacc0x4567 = vcombine_s32(vsum0x45, vsum0x67 );
    const int32x2_t vpsum1x0 = vadd_s32(vget_low_s32(vacc1x0), vget_high_s32(vacc1x0));
    const int32x2_t vpsum1x1 = vadd_s32(vget_low_s32(vacc1x1), vget_high_s32(vacc1x1));
    const int32x2_t vpsum1x2 = vadd_s32(vget_low_s32(vacc1x2), vget_high_s32(vacc1x2));
    const int32x2_t vpsum1x3 = vadd_s32(vget_low_s32(vacc1x3), vget_high_s32(vacc1x3));
    const int32x2_t vsum1x01 = vpadd_s32(vpsum1x0, vpsum1x1);
    const int32x2_t vsum1x23 = vpadd_s32(vpsum1x2, vpsum1x3);
    int32x4_t vacc1x0123 = vcombine_s32(vsum1x01, vsum1x23 );
    const int32x2_t vpsum1x4 = vadd_s32(vget_low_s32(vacc1x4), vget_high_s32(vacc1x4));
    const int32x2_t vpsum1x5 = vadd_s32(vget_low_s32(vacc1x5), vget_high_s32(vacc1x5));
    const int32x2_t vpsum1x6 = vadd_s32(vget_low_s32(vacc1x6), vget_high_s32(vacc1x6));
    const int32x2_t vpsum1x7 = vadd_s32(vget_low_s32(vacc1x7), vget_high_s32(vacc1x7));
    const int32x2_t vsum1x45 = vpadd_s32(vpsum1x4, vpsum1x5);
    const int32x2_t vsum1x67 = vpadd_s32(vpsum1x6, vpsum1x7);
    int32x4_t vacc1x4567 = vcombine_s32(vsum1x45, vsum1x67 );
#endif

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc1x0123 = vcvtq_f32_s32(vacc1x0123);
    float32x4_t vfpacc1x4567 = vcvtq_f32_s32(vacc1x4567);

    const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    vfpacc1x0123 = vmulq_f32(vfpacc1x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);
    vfpacc1x4567 = vmulq_f32(vfpacc1x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc1x0123 = vcvtnq_s32_f32(vfpacc1x0123);
    vacc1x4567 = vcvtnq_s32_f32(vfpacc1x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
#if XNN_ARCH_ARM64
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
    int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
    int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
    vacc1x01234567 = vqaddq_s16(vacc1x01234567, voutput_zero_point);

    int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
#endif

    const int8x16_t voutput_min = vld1q_dup_s8(&params->fp32_neonv8.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    const int8x16_t voutput_max = vld1q_dup_s8(&params->fp32_neonv8.output_max);
    vout0x01234567_1x01234567 = vminq_s8(vout0x01234567_1x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c1 + 0, vget_high_s8(vout0x01234567_1x01234567));
      vst1_s8(c0 + 0, vget_low_s8(vout0x01234567_1x01234567));

      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1q_lane_u32((void*) c1, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 2); c1 += 4;
        vst1q_lane_u32((void*) c0, vreinterpretq_u32_s8(vout0x01234567_1x01234567), 0); c0 += 4;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 4);
      }
      if (nc & 2) {
        vst1q_lane_u16((void*) c1, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 4); c1 += 2;
        vst1q_lane_u16((void*) c0, vreinterpretq_u16_s8(vout0x01234567_1x01234567), 0); c0 += 2;
        vout0x01234567_1x01234567 = vextq_s8(vout0x01234567_1x01234567, vout0x01234567_1x01234567, 2);
      }
      if (nc & 1) {
        vst1q_lane_s8(c1, vout0x01234567_1x01234567, 8);
        vst1q_lane_s8(c0, vout0x01234567_1x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
