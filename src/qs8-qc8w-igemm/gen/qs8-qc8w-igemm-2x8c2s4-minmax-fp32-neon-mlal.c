// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c2-neon-mull-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal(
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
    const union xnn_qs8_qc8w_conv_minmax_params* restrict params) XNN_OOB_READS
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

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc1x0123 = vacc0x0123;
    int32x4_t vacc1x4567 = vacc0x4567;

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
      while (k >= 16 * sizeof(int8_t)) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va0x1 = vld1_s8(a0); a0 += 8;
        int8x8_t va1x0 = vld1_s8(a1); a1 += 8;
        int8x8_t va1x1 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
        const int8x8_t vb0123c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c0 = vmlal_s8(vprod0x0123c0, vb0123c0x1, va0x1);
        vprod1x0123c0 = vmlal_s8(vprod1x0123c0, vb0123c0x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
        const int8x8_t vb4567c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c0 = vmlal_s8(vprod0x4567c0, vb4567c0x1, va0x1);
        vprod1x4567c0 = vmlal_s8(vprod1x4567c0, vb4567c0x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
        const int8x8_t vb0123c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c1 = vmlal_s8(vprod0x0123c1, vb0123c1x1, va0x1);
        vprod1x0123c1 = vmlal_s8(vprod1x0123c1, vb0123c1x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
        const int8x8_t vb4567c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c1 = vmlal_s8(vprod0x4567c1, vb4567c1x1, va0x1);
        vprod1x4567c1 = vmlal_s8(vprod1x4567c1, vb4567c1x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
        const int8x8_t vb0123c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c2 = vmlal_s8(vprod0x0123c2, vb0123c2x1, va0x1);
        vprod1x0123c2 = vmlal_s8(vprod1x0123c2, vb0123c2x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
        const int8x8_t vb4567c2x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c2 = vmlal_s8(vprod0x4567c2, vb4567c2x1, va0x1);
        vprod1x4567c2 = vmlal_s8(vprod1x4567c2, vb4567c2x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va0x1 = vext_s8(va0x1, va0x1, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        va1x1 = vext_s8(va1x1, va1x1, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
        const int8x8_t vb0123c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x0123c3 = vmlal_s8(vprod0x0123c3, vb0123c3x1, va0x1);
        vprod1x0123c3 = vmlal_s8(vprod1x0123c3, vb0123c3x1, va1x1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
        const int8x8_t vb4567c3x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x4567c3 = vmlal_s8(vprod0x4567c3, vb4567c3x1, va0x1);
        vprod1x4567c3 = vmlal_s8(vprod1x4567c3, vb4567c3x1, va1x1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

        k -= 16 * sizeof(int8_t);
      }
      if (k != 0) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va1x0 = vld1_s8(a1); a1 += 8;

        const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
        int16x8_t vprod1x0123c0 = vmull_s8(vb0123c0x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c0);
        int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
        int16x8_t vprod1x4567c0 = vmull_s8(vb4567c0x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c0);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
        int16x8_t vprod1x0123c1 = vmull_s8(vb0123c1x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c1);
        int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
        int16x8_t vprod1x4567c1 = vmull_s8(vb4567c1x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c1);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
        int16x8_t vprod1x0123c2 = vmull_s8(vb0123c2x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c2);
        int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
        int16x8_t vprod1x4567c2 = vmull_s8(vb4567c2x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c2);
        va0x0 = vext_s8(va0x0, va0x0, 2);
        va1x0 = vext_s8(va1x0, va1x0, 2);
        int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
        int16x8_t vprod1x0123c3 = vmull_s8(vb0123c3x0, va1x0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        vacc1x0123 = vpadalq_s16(vacc1x0123, vprod1x0123c3);
        int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
        int16x8_t vprod1x4567c3 = vmull_s8(vb4567c3x0, va1x0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        vacc1x4567 = vpadalq_s16(vacc1x4567, vprod1x4567c3);

      }

      p -= 2 * sizeof(void*);
    } while (p != 0);

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

    const float32x4_t vmagic_bias = vdupq_n_f32(params->fp32_neon.magic_bias);
    vacc0x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x0123, vmagic_bias));
    vacc0x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0x4567, vmagic_bias));
    vacc1x0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc1x0123, vmagic_bias));
    vacc1x4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc1x4567, vmagic_bias));

    const int32x4_t vmagic_bias_less_output_zero_point = vdupq_n_s32(params->fp32_neon.magic_bias_less_output_zero_point);
    vacc0x0123 = vqsubq_s32(vacc0x0123, vmagic_bias_less_output_zero_point);
    vacc0x4567 = vqsubq_s32(vacc0x4567, vmagic_bias_less_output_zero_point);
    vacc1x0123 = vqsubq_s32(vacc1x0123, vmagic_bias_less_output_zero_point);
    vacc1x4567 = vqsubq_s32(vacc1x4567, vmagic_bias_less_output_zero_point);

    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc1x01234567 = vqmovn_high_s32(vqmovn_s32(vacc1x0123), vacc1x4567);


      int8x16_t vout0x01234567_1x01234567 = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc1x01234567);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc1x01234567 = vcombine_s16(vqmovn_s32(vacc1x0123), vqmovn_s32(vacc1x4567));


      int8x16_t vout0x01234567_1x01234567 = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc1x01234567));
    #endif

    const int8x16_t voutput_min = vdupq_n_s8(params->fp32_neon.output_min);
    vout0x01234567_1x01234567 = vmaxq_s8(vout0x01234567_1x01234567, voutput_min);

    const int8x16_t voutput_max = vdupq_n_s8(params->fp32_neon.output_max);
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
