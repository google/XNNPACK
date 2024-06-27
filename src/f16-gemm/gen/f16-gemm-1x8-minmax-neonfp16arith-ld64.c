// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/neonfp16arith-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;

      const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
      #endif
      const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
      #endif
      const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
      #endif
      const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;

        const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16(w)); w = (const float16x8_t*) w + 1;

        vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 8;
    } else {
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;

        vacc0 = vget_high_f16(vacc0x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
