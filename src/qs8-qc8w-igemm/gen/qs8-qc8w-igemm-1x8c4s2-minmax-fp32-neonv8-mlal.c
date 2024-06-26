// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/c4-neon-mull-shuffle.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;

  kc = round_up_po2(kc, 8 * sizeof(int8_t));
  do {
    int32x4_t vacc0x01 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc0x23 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc0x45 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;
    int32x4_t vacc0x67 = vreinterpretq_s32_u64(vmovl_u32(vld1_u32(w))); w = (const int32_t*) w + 2;

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 16 * sizeof(int8_t)) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;
        int8x8_t va0x1 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb01c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb23c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb45c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb67c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb01c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb23c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb45c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb67c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x01c0 = vmull_s8(vb01c0x0, va0x0);
        const int8x8_t vb01c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x01c0 = vmlal_s8(vprod0x01c0, vb01c0x1, va0x1);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
        int16x8_t vprod0x23c0 = vmull_s8(vb23c0x0, va0x0);
        const int8x8_t vb23c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x23c0 = vmlal_s8(vprod0x23c0, vb23c0x1, va0x1);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
        int16x8_t vprod0x45c0 = vmull_s8(vb45c0x0, va0x0);
        const int8x8_t vb45c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x45c0 = vmlal_s8(vprod0x45c0, vb45c0x1, va0x1);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
        int16x8_t vprod0x67c0 = vmull_s8(vb67c0x0, va0x0);
        const int8x8_t vb67c0x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x67c0 = vmlal_s8(vprod0x67c0, vb67c0x1, va0x1);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
        va0x0 = vext_s8(va0x0, va0x0, 4);
        va0x1 = vext_s8(va0x1, va0x1, 4);
        int16x8_t vprod0x01c1 = vmull_s8(vb01c1x0, va0x0);
        const int8x8_t vb01c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x01c1 = vmlal_s8(vprod0x01c1, vb01c1x1, va0x1);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c1);
        int16x8_t vprod0x23c1 = vmull_s8(vb23c1x0, va0x0);
        const int8x8_t vb23c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x23c1 = vmlal_s8(vprod0x23c1, vb23c1x1, va0x1);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c1);
        int16x8_t vprod0x45c1 = vmull_s8(vb45c1x0, va0x0);
        const int8x8_t vb45c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x45c1 = vmlal_s8(vprod0x45c1, vb45c1x1, va0x1);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c1);
        int16x8_t vprod0x67c1 = vmull_s8(vb67c1x0, va0x0);
        const int8x8_t vb67c1x1 = vld1_s8(w); w = (const int8_t*) w + 8;
        vprod0x67c1 = vmlal_s8(vprod0x67c1, vb67c1x1, va0x1);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c1);

        k -= 16 * sizeof(int8_t);
      }
      if (k != 0) {
        int8x8_t va0x0 = vld1_s8(a0); a0 += 8;

        const int8x8_t vb01c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb23c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb45c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb67c0x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb01c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb23c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb45c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;
        const int8x8_t vb67c1x0 = vld1_s8(w); w = (const int8_t*) w + 8;

        int16x8_t vprod0x01c0 = vmull_s8(vb01c0x0, va0x0);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c0);
        int16x8_t vprod0x23c0 = vmull_s8(vb23c0x0, va0x0);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c0);
        int16x8_t vprod0x45c0 = vmull_s8(vb45c0x0, va0x0);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c0);
        int16x8_t vprod0x67c0 = vmull_s8(vb67c0x0, va0x0);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c0);
        va0x0 = vext_s8(va0x0, va0x0, 4);
        int16x8_t vprod0x01c1 = vmull_s8(vb01c1x0, va0x0);
        vacc0x01 = vpadalq_s16(vacc0x01, vprod0x01c1);
        int16x8_t vprod0x23c1 = vmull_s8(vb23c1x0, va0x0);
        vacc0x23 = vpadalq_s16(vacc0x23, vprod0x23c1);
        int16x8_t vprod0x45c1 = vmull_s8(vb45c1x0, va0x0);
        vacc0x45 = vpadalq_s16(vacc0x45, vprod0x45c1);
        int16x8_t vprod0x67c1 = vmull_s8(vb67c1x0, va0x0);
        vacc0x67 = vpadalq_s16(vacc0x67, vprod0x67c1);

      }

      p -= 1 * sizeof(void*);
    } while (p != 0);

#if XNN_ARCH_ARM64
    int32x4_t vacc0x0123 = vpaddq_s32(vacc0x01, vacc0x23);
    int32x4_t vacc0x4567 = vpaddq_s32(vacc0x45, vacc0x67);
#else
    const int32x2_t vsum0x01 = vpadd_s32(vget_low_s32(vacc0x01), vget_high_s32(vacc0x01));
    const int32x2_t vsum0x23 = vpadd_s32(vget_low_s32(vacc0x23), vget_high_s32(vacc0x23));
    int32x4_t vacc0x0123 = vcombine_s32(vsum0x01, vsum0x23);
    const int32x2_t vsum0x45 = vpadd_s32(vget_low_s32(vacc0x45), vget_high_s32(vacc0x45));
    const int32x2_t vsum0x67 = vpadd_s32(vget_low_s32(vacc0x67), vget_high_s32(vacc0x67));
    int32x4_t vacc0x4567 = vcombine_s32(vsum0x45, vsum0x67);
#endif

    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);

    const float32x4_t vscale0123 = vld1q_f32(w); w = (const float*) w + 4;
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale0123);
    const float32x4_t vscale4567 = vld1q_f32(w); w = (const float*) w + 4;
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale4567);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
#if XNN_ARCH_ARM64
    int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

    int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
#else
    int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));

    vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);

    int8x8_t vout0x01234567 = vqmovn_s16(vacc0x01234567);
#endif

    const int8x8_t voutput_min = vld1_dup_s8(&params->fp32_neonv8.output_min);
    vout0x01234567 = vmax_s8(vout0x01234567, voutput_min);

    const int8x8_t voutput_max = vld1_dup_s8(&params->fp32_neonv8.output_max);
    vout0x01234567 = vmin_s8(vout0x01234567, voutput_max);

    if (nc >= 8) {
      vst1_s8(c0 + 0, vout0x01234567);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);

      nc -= 8;
    } else {
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
