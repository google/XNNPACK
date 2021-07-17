// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c2-neon-mull-padal-dup.c.in
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


void xnn_qs8_igemm_minmax_gemmlowp_ukernel_1x16c2__neon_mull_padal_dup(
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
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 2 * sizeof(int8_t));
  int8_t* c0 = c;

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;


      while (k >= 8 * sizeof(int8_t)) {
        const int8x8_t va0 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb0123c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
        const int16x8_t vprod0x0123c3 = vmull_s8(vb0123c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c3);
        const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
        const int16x8_t vprod0x4567c3 = vmull_s8(vb4567c3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c3);
        const int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        const int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        const int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
        const int16x8_t vprod0x89ABc3 = vmull_s8(vb89ABc3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc3);
        const int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        const int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
        const int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
        const int16x8_t vprod0xCDEFc3 = vmull_s8(vbCDEFc3, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 3)));
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc3);

        k -= 8 * sizeof(int8_t);
      }

      if XNN_UNLIKELY(k != 0) {
        const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);

        const int8x8_t vb0123c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb4567c0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vb89ABc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        const int8x8_t vbCDEFc0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

        const int16x8_t vprod0x0123c0 = vmull_s8(vb0123c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c0);
        const int16x8_t vprod0x4567c0 = vmull_s8(vb4567c0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c0);
        const int16x8_t vprod0x89ABc0 = vmull_s8(vb89ABc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc0);
        const int16x8_t vprod0xCDEFc0 = vmull_s8(vbCDEFc0, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 0)));
        vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc0);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb0123c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb4567c1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vb89ABc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
          const int8x8_t vbCDEFc1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

          const int16x8_t vprod0x0123c1 = vmull_s8(vb0123c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
          vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c1);
          const int16x8_t vprod0x4567c1 = vmull_s8(vb4567c1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
          vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c1);
          const int16x8_t vprod0x89ABc1 = vmull_s8(vb89ABc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
          vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc1);
          const int16x8_t vprod0xCDEFc1 = vmull_s8(vbCDEFc1, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 1)));
          vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc1);

          if (k > 4 * sizeof(int8_t)) {
            const int8x8_t vb0123c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vb4567c2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vb89ABc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
            const int8x8_t vbCDEFc2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

            const int16x8_t vprod0x0123c2 = vmull_s8(vb0123c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
            vacc0x0123 = vpadalq_s16(vacc0x0123, vprod0x0123c2);
            const int16x8_t vprod0x4567c2 = vmull_s8(vb4567c2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
            vacc0x4567 = vpadalq_s16(vacc0x4567, vprod0x4567c2);
            const int16x8_t vprod0x89ABc2 = vmull_s8(vb89ABc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
            vacc0x89AB = vpadalq_s16(vacc0x89AB, vprod0x89ABc2);
            const int16x8_t vprod0xCDEFc2 = vmull_s8(vbCDEFc2, vreinterpret_s8_s16(vdup_lane_s16(vreinterpret_s16_s8(va0), 2)));
            vacc0xCDEF = vpadalq_s16(vacc0xCDEF, vprod0xCDEFc2);
          }
        }
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    const int32x4_t vmultiplier = vld1q_dup_s32(&params->gemmlowp_neon.multiplier);
    const int32x4_t vright_shift = vld1q_dup_s32(&params->gemmlowp_neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));

    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqrdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqrdmulhq_s32(vacc0xCDEF, vmultiplier);

    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc0x89AB = vsraq_n_s32(vacc0x89AB, vbicq_s32(vacc0x89AB, vzero_shift_mask), 31);
    vacc0xCDEF = vsraq_n_s32(vacc0xCDEF, vbicq_s32(vacc0xCDEF, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->gemmlowp_neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->gemmlowp_neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->gemmlowp_neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      int8x8_t vout0x01234567 = vget_low_s8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_s8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_s8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32(__builtin_assume_aligned(c0, 1), vreinterpret_u32_s8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16(__builtin_assume_aligned(c0, 1), vreinterpret_u16_s8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_s8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_s8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
