// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c2-neon-mull-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qs8_gemm_minmax_rndnu_ukernel_1x8c2s4__neon_mull(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const int8_t* a0 = a;
  int8_t* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

      const int8x8_t vb0123c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c2x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb0123c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb4567c3x0 = vld1_s8(w); w = (const int8_t*) w + 8;

      int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
      int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
      int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
      int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
      va0x0 = vext_s8(va0x0, va0x0, 2);
      int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3x0, va0x0);
      vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
      int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3x0, va0x0);
      vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);

      k -= 8 * sizeof(int8_t);
    } while (k != 0);

    const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
    const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);

    vacc0x0123 = vqshlq_s32(vacc0x0123, vright_pre_shift);
    vacc0x4567 = vqshlq_s32(vacc0x4567, vright_pre_shift);

    vacc0x0123 = vqdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqdmulhq_s32(vacc0x4567, vmultiplier);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_post_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_post_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
#if XNN_ARCH_ARM64
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

    int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

    int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
#endif

    const int8x8_t voutput_min = vld1_dup_s8(&params->rndnu_neon.output_min);
    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    const int8x8_t voutput_max = vld1_dup_s8(&params->rndnu_neon.output_max);
    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vout0x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      // Final case where not all of the 8 columns fit in the destination.
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
