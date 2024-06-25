// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/neon-mlal-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_qu8_igemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(uint8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;

  const uint8x8_t vb_zero_point = vld1_dup_u8(&params->fp32_neonv8.kernel_zero_point[0]);
  do {
    int32x4_t vacc0x0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0x89AB = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
    int32x4_t vacc0xCDEF = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 8 * sizeof(uint8_t)) {
        const uint8x8_t va0 = vld1_u8(a0); a0 += 8;
        const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));

        const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        const uint8x8_t vb89ABCDEFc0 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc0 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc0, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        const uint8x8_t vb89ABCDEFc1 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc1 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc1, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
        const uint8x8_t vb89ABCDEFc2 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc2 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc2, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
        const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
        const uint8x8_t vb89ABCDEFc3 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc3 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc3, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);


        const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
        const uint8x8_t vb89ABCDEFc4 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc4 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc4, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
        const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
        const uint8x8_t vb89ABCDEFc5 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc5 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc5, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
        const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
        const uint8x8_t vb89ABCDEFc6 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc6 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc6, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
        const uint8x8_t vb01234567c7 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c7 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c7, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
        const uint8x8_t vb89ABCDEFc7 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc7 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc7, vb_zero_point));

        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);

        k -= 8 * sizeof(uint8_t);
      }
      if XNN_UNLIKELY(k != 0) {
        const uint8x8_t va0 = vld1_u8(a0); a0 = (const uint8_t*) ((uintptr_t) a0 + k);
        const int16x8_t vxa0 = vreinterpretq_s16_u16(vmovl_u8(va0));

        const uint8x8_t vb01234567c0 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb01234567c0 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c0, vb_zero_point));
        const uint8x8_t vb89ABCDEFc0 = vld1_u8(w); w = (const uint8_t*) w + 8;
        const int16x8_t vxb89ABCDEFc0 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc0, vb_zero_point));

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);

        if (k >= 2 * sizeof(uint8_t)) {
          const uint8x8_t vb01234567c1 = vld1_u8(w); w = (const uint8_t*) w + 8;
          const int16x8_t vxb01234567c1 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c1, vb_zero_point));
          const uint8x8_t vb89ABCDEFc1 = vld1_u8(w); w = (const uint8_t*) w + 8;
          const int16x8_t vxb89ABCDEFc1 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc1, vb_zero_point));

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);

          if (k > 2 * sizeof(uint8_t)) {
            const uint8x8_t vb01234567c2 = vld1_u8(w); w = (const uint8_t*) w + 8;
            const int16x8_t vxb01234567c2 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c2, vb_zero_point));
            const uint8x8_t vb89ABCDEFc2 = vld1_u8(w); w = (const uint8_t*) w + 8;
            const int16x8_t vxb89ABCDEFc2 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc2, vb_zero_point));

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);

            if (k >= 4 * sizeof(uint8_t)) {
              const uint8x8_t vb01234567c3 = vld1_u8(w); w = (const uint8_t*) w + 8;
              const int16x8_t vxb01234567c3 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c3, vb_zero_point));
              const uint8x8_t vb89ABCDEFc3 = vld1_u8(w); w = (const uint8_t*) w + 8;
              const int16x8_t vxb89ABCDEFc3 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc3, vb_zero_point));

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);

              if (k > 4 * sizeof(uint8_t)) {
                const uint8x8_t vb01234567c4 = vld1_u8(w); w = (const uint8_t*) w + 8;
                const int16x8_t vxb01234567c4 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c4, vb_zero_point));
                const uint8x8_t vb89ABCDEFc4 = vld1_u8(w); w = (const uint8_t*) w + 8;
                const int16x8_t vxb89ABCDEFc4 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc4, vb_zero_point));

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);

                if (k >= 6 * sizeof(uint8_t)) {
                  const uint8x8_t vb01234567c5 = vld1_u8(w); w = (const uint8_t*) w + 8;
                  const int16x8_t vxb01234567c5 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c5, vb_zero_point));
                  const uint8x8_t vb89ABCDEFc5 = vld1_u8(w); w = (const uint8_t*) w + 8;
                  const int16x8_t vxb89ABCDEFc5 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc5, vb_zero_point));

                  vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                  vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                  vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);

                  if (k > 6 * sizeof(uint8_t)) {
                    const uint8x8_t vb01234567c6 = vld1_u8(w); w = (const uint8_t*) w + 8;
                    const int16x8_t vxb01234567c6 = vreinterpretq_s16_u16(vsubl_u8(vb01234567c6, vb_zero_point));
                    const uint8x8_t vb89ABCDEFc6 = vld1_u8(w); w = (const uint8_t*) w + 8;
                    const int16x8_t vxb89ABCDEFc6 = vreinterpretq_s16_u16(vsubl_u8(vb89ABCDEFc6, vb_zero_point));

                    vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
                    vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                    vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
                  }
                }
              }
            }
          }
        }
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    // Post-accumulation work
    float32x4_t vfpacc0x0123 = vcvtq_f32_s32(vacc0x0123);
    float32x4_t vfpacc0x4567 = vcvtq_f32_s32(vacc0x4567);
    float32x4_t vfpacc0x89AB = vcvtq_f32_s32(vacc0x89AB);
    float32x4_t vfpacc0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neonv8.scale);
    vfpacc0x0123 = vmulq_f32(vfpacc0x0123, vscale);
    vfpacc0x4567 = vmulq_f32(vfpacc0x4567, vscale);
    vfpacc0x89AB = vmulq_f32(vfpacc0x89AB, vscale);
    vfpacc0xCDEF = vmulq_f32(vfpacc0xCDEF, vscale);

    vacc0x0123 = vcvtnq_s32_f32(vfpacc0x0123);
    vacc0x4567 = vcvtnq_s32_f32(vfpacc0x4567);
    vacc0x89AB = vcvtnq_s32_f32(vfpacc0x89AB);
    vacc0xCDEF = vcvtnq_s32_f32(vfpacc0xCDEF);

    const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
    #if XNN_ARCH_ARM64
      int16x8_t vacc0x01234567 = vqmovn_high_s32(vqmovn_s32(vacc0x0123), vacc0x4567);
      int16x8_t vacc0x89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc0x89AB), vacc0xCDEF);

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

      uint8x16_t vout0x0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc0x01234567), vacc0x89ABCDEF);
    #else
      int16x8_t vacc0x01234567 = vcombine_s16(vqmovn_s32(vacc0x0123), vqmovn_s32(vacc0x4567));
      int16x8_t vacc0x89ABCDEF = vcombine_s16(vqmovn_s32(vacc0x89AB), vqmovn_s32(vacc0xCDEF));

      vacc0x01234567 = vqaddq_s16(vacc0x01234567, voutput_zero_point);
      vacc0x89ABCDEF = vqaddq_s16(vacc0x89ABCDEF, voutput_zero_point);

      uint8x16_t vout0x0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc0x01234567), vqmovun_s16(vacc0x89ABCDEF));
    #endif

    const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neonv8.output_min);
    vout0x0123456789ABCDEF = vmaxq_u8(vout0x0123456789ABCDEF, voutput_min);

    const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neonv8.output_max);
    vout0x0123456789ABCDEF = vminq_u8(vout0x0123456789ABCDEF, voutput_max);

    if (nc >= 16) {
      vst1q_u8(c0 + 0, vout0x0123456789ABCDEF);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);

      nc -= 16;
    } else {
      uint8x8_t vout0x01234567 = vget_low_u8(vout0x0123456789ABCDEF);
      if (nc & 8) {
        vst1_u8(c0, vout0x01234567); c0 += 8;
        vout0x01234567 = vget_high_u8(vout0x0123456789ABCDEF);
      }
      if (nc & 4) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_u8(vout0x01234567), 0); c0 += 4;
        vout0x01234567 = vext_u8(vout0x01234567, vout0x01234567, 4);
      }
      if (nc & 2) {
        vst1_lane_u16((void*) c0, vreinterpret_u16_u8(vout0x01234567), 0); c0 += 2;
        vout0x01234567 = vext_u8(vout0x01234567, vout0x01234567, 2);
      }
      if (nc & 1) {
        vst1_lane_u8(c0, vout0x01234567, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
