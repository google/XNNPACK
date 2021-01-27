// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c8-neon-mull-addlv.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gemm.h>


static const int32x4_t combine4(const int32x4_t a, const int32x4_t b, const int32x4_t c, const int32x4_t d) {
#if XNN_ARCH_ARM64
  const int32x4_t ab = vpaddq_s32(a, b);
  const int32x4_t cd = vpaddq_s32(c, d);
  const int32x4_t abcd = vpaddq_s32(ab, cd);
#else
  const int32x2_t ab_low  = vpadd_s32(vget_low_s32(a),  vget_low_s32(b));
  const int32x2_t ab_high = vpadd_s32(vget_high_s32(a), vget_high_s32(b));
  const int32x2_t cd_low  = vpadd_s32(vget_low_s32(c), vget_low_s32(d));
  const int32x2_t cd_high = vpadd_s32(vget_high_s32(c), vget_high_s32(d));
  const int32x2_t ab = vadd_s32(ab_low, ab_high);
  const int32x2_t cd = vadd_s32(cd_low, cd_high);
  const int32x4_t abcd = vcombine_s32(ab, cd);
#endif
  return abcd;
}

void xnn_qs8_gemm_minmax_ukernel_1x16c8__neon_mull_addlv(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
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

  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t));

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;

      const int8x8_t vbx0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0 = vmull_s8(vbx0, va0);
      const int32x4_t vacc0x0 = vpaddlq_s16(vprod0x0);
      const int8x8_t vbx1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x1 = vmull_s8(vbx1, va0);
      const int32x4_t vacc0x1 = vpaddlq_s16(vprod0x1);
      const int8x8_t vbx2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x2 = vmull_s8(vbx2, va0);
      const int32x4_t vacc0x2 = vpaddlq_s16(vprod0x2);
      const int8x8_t vbx3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x3 = vmull_s8(vbx3, va0);
      const int32x4_t vacc0x3 = vpaddlq_s16(vprod0x3);
      const int8x8_t vbx4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x4 = vmull_s8(vbx4, va0);
      const int32x4_t vacc0x4 = vpaddlq_s16(vprod0x4);
      const int8x8_t vbx5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x5 = vmull_s8(vbx5, va0);
      const int32x4_t vacc0x5 = vpaddlq_s16(vprod0x5);
      const int8x8_t vbx6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x6 = vmull_s8(vbx6, va0);
      const int32x4_t vacc0x6 = vpaddlq_s16(vprod0x6);
      const int8x8_t vbx7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x7 = vmull_s8(vbx7, va0);
      const int32x4_t vacc0x7 = vpaddlq_s16(vprod0x7);
      const int8x8_t vbx8 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x8 = vmull_s8(vbx8, va0);
      const int32x4_t vacc0x8 = vpaddlq_s16(vprod0x8);
      const int8x8_t vbx9 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x9 = vmull_s8(vbx9, va0);
      const int32x4_t vacc0x9 = vpaddlq_s16(vprod0x9);
      const int8x8_t vbx10 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x10 = vmull_s8(vbx10, va0);
      const int32x4_t vacc0x10 = vpaddlq_s16(vprod0x10);
      const int8x8_t vbx11 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x11 = vmull_s8(vbx11, va0);
      const int32x4_t vacc0x11 = vpaddlq_s16(vprod0x11);
      const int8x8_t vbx12 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x12 = vmull_s8(vbx12, va0);
      const int32x4_t vacc0x12 = vpaddlq_s16(vprod0x12);
      const int8x8_t vbx13 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x13 = vmull_s8(vbx13, va0);
      const int32x4_t vacc0x13 = vpaddlq_s16(vprod0x13);
      const int8x8_t vbx14 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x14 = vmull_s8(vbx14, va0);
      const int32x4_t vacc0x14 = vpaddlq_s16(vprod0x14);
      const int8x8_t vbx15 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x15 = vmull_s8(vbx15, va0);
      const int32x4_t vacc0x15 = vpaddlq_s16(vprod0x15);

      const int32x4_t vprod0x0123 = combine4(vacc0x0, vacc0x1, vacc0x2, vacc0x3);
      vacc0x0123 = vaddq_s32(vacc0x0123, vprod0x0123);
      const int32x4_t vprod0x4567 = combine4(vacc0x4, vacc0x5, vacc0x6, vacc0x7);
      vacc0x4567 = vaddq_s32(vacc0x4567, vprod0x4567);
      const int32x4_t vprod0x89AB = combine4(vacc0x8, vacc0x9, vacc0x10, vacc0x11);
      vacc0x89AB = vaddq_s32(vacc0x89AB, vprod0x89AB);
      const int32x4_t vprod0xCDEF = combine4(vacc0x12, vacc0x13, vacc0x14, vacc0x15);
      vacc0xCDEF = vaddq_s32(vacc0xCDEF, vprod0xCDEF);

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);

      const int8x8_t vbx0 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x0 = vmull_s8(vbx0, va0);
      const int32x4_t vacc0x0 = vpaddlq_s16(vprod0x0);
      const int8x8_t vbx1 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x1 = vmull_s8(vbx1, va0);
      const int32x4_t vacc0x1 = vpaddlq_s16(vprod0x1);
      const int8x8_t vbx2 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x2 = vmull_s8(vbx2, va0);
      const int32x4_t vacc0x2 = vpaddlq_s16(vprod0x2);
      const int8x8_t vbx3 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x3 = vmull_s8(vbx3, va0);
      const int32x4_t vacc0x3 = vpaddlq_s16(vprod0x3);
      const int8x8_t vbx4 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x4 = vmull_s8(vbx4, va0);
      const int32x4_t vacc0x4 = vpaddlq_s16(vprod0x4);
      const int8x8_t vbx5 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x5 = vmull_s8(vbx5, va0);
      const int32x4_t vacc0x5 = vpaddlq_s16(vprod0x5);
      const int8x8_t vbx6 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x6 = vmull_s8(vbx6, va0);
      const int32x4_t vacc0x6 = vpaddlq_s16(vprod0x6);
      const int8x8_t vbx7 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x7 = vmull_s8(vbx7, va0);
      const int32x4_t vacc0x7 = vpaddlq_s16(vprod0x7);
      const int8x8_t vbx8 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x8 = vmull_s8(vbx8, va0);
      const int32x4_t vacc0x8 = vpaddlq_s16(vprod0x8);
      const int8x8_t vbx9 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x9 = vmull_s8(vbx9, va0);
      const int32x4_t vacc0x9 = vpaddlq_s16(vprod0x9);
      const int8x8_t vbx10 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x10 = vmull_s8(vbx10, va0);
      const int32x4_t vacc0x10 = vpaddlq_s16(vprod0x10);
      const int8x8_t vbx11 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x11 = vmull_s8(vbx11, va0);
      const int32x4_t vacc0x11 = vpaddlq_s16(vprod0x11);
      const int8x8_t vbx12 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x12 = vmull_s8(vbx12, va0);
      const int32x4_t vacc0x12 = vpaddlq_s16(vprod0x12);
      const int8x8_t vbx13 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x13 = vmull_s8(vbx13, va0);
      const int32x4_t vacc0x13 = vpaddlq_s16(vprod0x13);
      const int8x8_t vbx14 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x14 = vmull_s8(vbx14, va0);
      const int32x4_t vacc0x14 = vpaddlq_s16(vprod0x14);
      const int8x8_t vbx15 = vld1_s8(w); w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));

      const int16x8_t vprod0x15 = vmull_s8(vbx15, va0);
      const int32x4_t vacc0x15 = vpaddlq_s16(vprod0x15);

      const int32x4_t vprod0x0123 = combine4(vacc0x0, vacc0x1, vacc0x2, vacc0x3);
      vacc0x0123 = vaddq_s32(vacc0x0123, vprod0x0123);
      const int32x4_t vprod0x4567 = combine4(vacc0x4, vacc0x5, vacc0x6, vacc0x7);
      vacc0x4567 = vaddq_s32(vacc0x4567, vprod0x4567);
      const int32x4_t vprod0x89AB = combine4(vacc0x8, vacc0x9, vacc0x10, vacc0x11);
      vacc0x89AB = vaddq_s32(vacc0x89AB, vprod0x89AB);
      const int32x4_t vprod0xCDEF = combine4(vacc0x12, vacc0x13, vacc0x14, vacc0x15);
      vacc0xCDEF = vaddq_s32(vacc0xCDEF, vprod0xCDEF);

    }
    const int32x4_t vmultiplier = vld1q_dup_s32(&params->neon.multiplier);
    vacc0x0123 = vqrdmulhq_s32(vacc0x0123, vmultiplier);
    vacc0x4567 = vqrdmulhq_s32(vacc0x4567, vmultiplier);
    vacc0x89AB = vqrdmulhq_s32(vacc0x89AB, vmultiplier);
    vacc0xCDEF = vqrdmulhq_s32(vacc0xCDEF, vmultiplier);

    const int32x4_t vright_shift = vld1q_dup_s32(&params->neon.right_shift);
    const int32x4_t vzero_shift_mask = vreinterpretq_s32_u32(vceqq_s32(vright_shift, vmovq_n_s32(0)));
    vacc0x0123 = vsraq_n_s32(vacc0x0123, vbicq_s32(vacc0x0123, vzero_shift_mask), 31);
    vacc0x4567 = vsraq_n_s32(vacc0x4567, vbicq_s32(vacc0x4567, vzero_shift_mask), 31);
    vacc0x89AB = vsraq_n_s32(vacc0x89AB, vbicq_s32(vacc0x89AB, vzero_shift_mask), 31);
    vacc0xCDEF = vsraq_n_s32(vacc0xCDEF, vbicq_s32(vacc0xCDEF, vzero_shift_mask), 31);

    vacc0x0123 = vrshlq_s32(vacc0x0123, vright_shift);
    vacc0x4567 = vrshlq_s32(vacc0x4567, vright_shift);
    vacc0x89AB = vrshlq_s32(vacc0x89AB, vright_shift);
    vacc0xCDEF = vrshlq_s32(vacc0xCDEF, vright_shift);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
#if XNN_ARCH_ARM64
    const int16x8_t vacc0x01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF), voutput_zero_point);
    int8x16_t vout0x0123456789ABCDEF = vqmovn_high_s16(vqmovn_s16(vacc0x01234567), vacc0x89ABCDEF);
#else
    const int16x8_t vacc0x01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567)), voutput_zero_point);
    const int16x8_t vacc0x89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF)), voutput_zero_point);

    int8x16_t vout0x0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc0x01234567), vqmovn_s16(vacc0x89ABCDEF));
#endif
    const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.output_min);
    const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.output_max);

    vout0x0123456789ABCDEF = vmaxq_s8(vout0x0123456789ABCDEF, voutput_min);

    vout0x0123456789ABCDEF = vminq_s8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_s8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

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
