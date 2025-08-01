// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/neonfp16arith-ld64.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_float16* restrict a,
    size_t a_stride,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint16_t* a6 = (const uint16_t*) ((uintptr_t) a5 + a_stride);
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const uint16_t* a7 = (const uint16_t*) ((uintptr_t) a6 + a_stride);
  uint16_t* c7 = (uint16_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  do {
    float16x8_t vacc0x0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;
    float16x8_t vacc1x0 = vacc0x0;
    float16x8_t vacc2x0 = vacc0x0;
    float16x8_t vacc3x0 = vacc0x0;
    float16x8_t vacc4x0 = vacc0x0;
    float16x8_t vacc5x0 = vacc0x0;
    float16x8_t vacc6x0 = vacc0x0;
    float16x8_t vacc7x0 = vacc0x0;

    size_t k = kc;
    while (k >= 4 * sizeof(uint16_t)) {
      const float16x4_t va0 = vreinterpret_f16_u16(vld1_u16(a0)); a0 += 4;
      const float16x4_t va1 = vreinterpret_f16_u16(vld1_u16(a1)); a1 += 4;
      const float16x4_t va2 = vreinterpret_f16_u16(vld1_u16(a2)); a2 += 4;
      const float16x4_t va3 = vreinterpret_f16_u16(vld1_u16(a3)); a3 += 4;
      const float16x4_t va4 = vreinterpret_f16_u16(vld1_u16(a4)); a4 += 4;
      const float16x4_t va5 = vreinterpret_f16_u16(vld1_u16(a5)); a5 += 4;
      const float16x4_t va6 = vreinterpret_f16_u16(vld1_u16(a6)); a6 += 4;
      const float16x4_t va7 = vreinterpret_f16_u16(vld1_u16(a7)); a7 += 4;

      const float16x8_t vb0c0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c0, va1, 0);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c0, va2, 0);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c0, va3, 0);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c0, va4, 0);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c0, va5, 0);
        vacc6x0 = vfmaq_lane_f16(vacc6x0, vb0c0, va6, 0);
        vacc7x0 = vfmaq_lane_f16(vacc7x0, vb0c0, va7, 0);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c0, va0, 0);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c0, va1, 0);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c0, va2, 0);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c0, va3, 0);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c0, va4, 0);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c0, va5, 0);
        vacc6x0 = vmlaq_lane_f16(vacc6x0, vb0c0, va6, 0);
        vacc7x0 = vmlaq_lane_f16(vacc7x0, vb0c0, va7, 0);
      #endif
      const float16x8_t vb0c1 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c1, va1, 1);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c1, va2, 1);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c1, va3, 1);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c1, va4, 1);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c1, va5, 1);
        vacc6x0 = vfmaq_lane_f16(vacc6x0, vb0c1, va6, 1);
        vacc7x0 = vfmaq_lane_f16(vacc7x0, vb0c1, va7, 1);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c1, va0, 1);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c1, va1, 1);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c1, va2, 1);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c1, va3, 1);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c1, va4, 1);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c1, va5, 1);
        vacc6x0 = vmlaq_lane_f16(vacc6x0, vb0c1, va6, 1);
        vacc7x0 = vmlaq_lane_f16(vacc7x0, vb0c1, va7, 1);
      #endif
      const float16x8_t vb0c2 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c2, va1, 2);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c2, va2, 2);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c2, va3, 2);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c2, va4, 2);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c2, va5, 2);
        vacc6x0 = vfmaq_lane_f16(vacc6x0, vb0c2, va6, 2);
        vacc7x0 = vfmaq_lane_f16(vacc7x0, vb0c2, va7, 2);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c2, va0, 2);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c2, va1, 2);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c2, va2, 2);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c2, va3, 2);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c2, va4, 2);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c2, va5, 2);
        vacc6x0 = vmlaq_lane_f16(vacc6x0, vb0c2, va6, 2);
        vacc7x0 = vmlaq_lane_f16(vacc7x0, vb0c2, va7, 2);
      #endif
      const float16x8_t vb0c3 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

      #if XNN_ARCH_ARM64
        vacc0x0 = vfmaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc1x0 = vfmaq_lane_f16(vacc1x0, vb0c3, va1, 3);
        vacc2x0 = vfmaq_lane_f16(vacc2x0, vb0c3, va2, 3);
        vacc3x0 = vfmaq_lane_f16(vacc3x0, vb0c3, va3, 3);
        vacc4x0 = vfmaq_lane_f16(vacc4x0, vb0c3, va4, 3);
        vacc5x0 = vfmaq_lane_f16(vacc5x0, vb0c3, va5, 3);
        vacc6x0 = vfmaq_lane_f16(vacc6x0, vb0c3, va6, 3);
        vacc7x0 = vfmaq_lane_f16(vacc7x0, vb0c3, va7, 3);
      #else
        vacc0x0 = vmlaq_lane_f16(vacc0x0, vb0c3, va0, 3);
        vacc1x0 = vmlaq_lane_f16(vacc1x0, vb0c3, va1, 3);
        vacc2x0 = vmlaq_lane_f16(vacc2x0, vb0c3, va2, 3);
        vacc3x0 = vmlaq_lane_f16(vacc3x0, vb0c3, va3, 3);
        vacc4x0 = vmlaq_lane_f16(vacc4x0, vb0c3, va4, 3);
        vacc5x0 = vmlaq_lane_f16(vacc5x0, vb0c3, va5, 3);
        vacc6x0 = vmlaq_lane_f16(vacc6x0, vb0c3, va6, 3);
        vacc7x0 = vmlaq_lane_f16(vacc7x0, vb0c3, va7, 3);
      #endif

      k -= 4 * sizeof(uint16_t);
    }
    if XNN_UNLIKELY(k != 0) {
      do {
        const float16x8_t va0 = vreinterpretq_f16_u16(vld1q_dup_u16(a0)); a0 += 1;
        const float16x8_t va1 = vreinterpretq_f16_u16(vld1q_dup_u16(a1)); a1 += 1;
        const float16x8_t va2 = vreinterpretq_f16_u16(vld1q_dup_u16(a2)); a2 += 1;
        const float16x8_t va3 = vreinterpretq_f16_u16(vld1q_dup_u16(a3)); a3 += 1;
        const float16x8_t va4 = vreinterpretq_f16_u16(vld1q_dup_u16(a4)); a4 += 1;
        const float16x8_t va5 = vreinterpretq_f16_u16(vld1q_dup_u16(a5)); a5 += 1;
        const float16x8_t va6 = vreinterpretq_f16_u16(vld1q_dup_u16(a6)); a6 += 1;
        const float16x8_t va7 = vreinterpretq_f16_u16(vld1q_dup_u16(a7)); a7 += 1;

        const float16x8_t vb0 = vreinterpretq_f16_u16(vld1q_u16((const uint16_t*) w)); w = (const xnn_float16*) w + 8;

        vacc0x0 = vfmaq_f16(vacc0x0, va0, vb0);
        vacc1x0 = vfmaq_f16(vacc1x0, va1, vb0);
        vacc2x0 = vfmaq_f16(vacc2x0, va2, vb0);
        vacc3x0 = vfmaq_f16(vacc3x0, va3, vb0);
        vacc4x0 = vfmaq_f16(vacc4x0, va4, vb0);
        vacc5x0 = vfmaq_f16(vacc5x0, va5, vb0);
        vacc6x0 = vfmaq_f16(vacc6x0, va6, vb0);
        vacc7x0 = vfmaq_f16(vacc7x0, va7, vb0);

        k -= sizeof(uint16_t);
      } while (k != 0);
    }

    const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.min));
    vacc0x0 = vmaxq_f16(vacc0x0, vmin);
    vacc1x0 = vmaxq_f16(vacc1x0, vmin);
    vacc2x0 = vmaxq_f16(vacc2x0, vmin);
    vacc3x0 = vmaxq_f16(vacc3x0, vmin);
    vacc4x0 = vmaxq_f16(vacc4x0, vmin);
    vacc5x0 = vmaxq_f16(vacc5x0, vmin);
    vacc6x0 = vmaxq_f16(vacc6x0, vmin);
    vacc7x0 = vmaxq_f16(vacc7x0, vmin);

    const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16((const uint16_t*) &params->scalar.max));
    vacc0x0 = vminq_f16(vacc0x0, vmax);
    vacc1x0 = vminq_f16(vacc1x0, vmax);
    vacc2x0 = vminq_f16(vacc2x0, vmax);
    vacc3x0 = vminq_f16(vacc3x0, vmax);
    vacc4x0 = vminq_f16(vacc4x0, vmax);
    vacc5x0 = vminq_f16(vacc5x0, vmax);
    vacc6x0 = vminq_f16(vacc6x0, vmax);
    vacc7x0 = vminq_f16(vacc7x0, vmax);

    if XNN_LIKELY(nc >= 8) {
      vst1q_u16(c0, vreinterpretq_u16_f16(vacc0x0));
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      vst1q_u16(c1, vreinterpretq_u16_f16(vacc1x0));
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1q_u16(c2, vreinterpretq_u16_f16(vacc2x0));
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1q_u16(c3, vreinterpretq_u16_f16(vacc3x0));
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      vst1q_u16(c4, vreinterpretq_u16_f16(vacc4x0));
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      vst1q_u16(c5, vreinterpretq_u16_f16(vacc5x0));
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      vst1q_u16(c6, vreinterpretq_u16_f16(vacc6x0));
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      vst1q_u16(c7, vreinterpretq_u16_f16(vacc7x0));
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);
      a6 = (const uint16_t*) ((uintptr_t) a6 - kc);
      a7 = (const uint16_t*) ((uintptr_t) a7 - kc);

      nc -= 8;
    } else {
      float16x4_t vacc0 = vget_low_f16(vacc0x0);
      float16x4_t vacc1 = vget_low_f16(vacc1x0);
      float16x4_t vacc2 = vget_low_f16(vacc2x0);
      float16x4_t vacc3 = vget_low_f16(vacc3x0);
      float16x4_t vacc4 = vget_low_f16(vacc4x0);
      float16x4_t vacc5 = vget_low_f16(vacc5x0);
      float16x4_t vacc6 = vget_low_f16(vacc6x0);
      float16x4_t vacc7 = vget_low_f16(vacc7x0);
      if (nc & 4) {
        vst1_u16(c0, vreinterpret_u16_f16(vacc0)); c0 += 4;
        vst1_u16(c1, vreinterpret_u16_f16(vacc1)); c1 += 4;
        vst1_u16(c2, vreinterpret_u16_f16(vacc2)); c2 += 4;
        vst1_u16(c3, vreinterpret_u16_f16(vacc3)); c3 += 4;
        vst1_u16(c4, vreinterpret_u16_f16(vacc4)); c4 += 4;
        vst1_u16(c5, vreinterpret_u16_f16(vacc5)); c5 += 4;
        vst1_u16(c6, vreinterpret_u16_f16(vacc6)); c6 += 4;
        vst1_u16(c7, vreinterpret_u16_f16(vacc7)); c7 += 4;

        vacc0 = vget_high_f16(vacc0x0);
        vacc1 = vget_high_f16(vacc1x0);
        vacc2 = vget_high_f16(vacc2x0);
        vacc3 = vget_high_f16(vacc3x0);
        vacc4 = vget_high_f16(vacc4x0);
        vacc5 = vget_high_f16(vacc5x0);
        vacc6 = vget_high_f16(vacc6x0);
        vacc7 = vget_high_f16(vacc7x0);
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_f16(vacc0), 0); c0 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_f16(vacc1), 0); c1 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_f16(vacc2), 0); c2 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_f16(vacc3), 0); c3 += 2;
        vst1_lane_u32((void*) c4, vreinterpret_u32_f16(vacc4), 0); c4 += 2;
        vst1_lane_u32((void*) c5, vreinterpret_u32_f16(vacc5), 0); c5 += 2;
        vst1_lane_u32((void*) c6, vreinterpret_u32_f16(vacc6), 0); c6 += 2;
        vst1_lane_u32((void*) c7, vreinterpret_u32_f16(vacc7), 0); c7 += 2;

        vacc0 = vext_f16(vacc0, vacc0, 2);
        vacc1 = vext_f16(vacc1, vacc1, 2);
        vacc2 = vext_f16(vacc2, vacc2, 2);
        vacc3 = vext_f16(vacc3, vacc3, 2);
        vacc4 = vext_f16(vacc4, vacc4, 2);
        vacc5 = vext_f16(vacc5, vacc5, 2);
        vacc6 = vext_f16(vacc6, vacc6, 2);
        vacc7 = vext_f16(vacc7, vacc7, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vreinterpret_u16_f16(vacc0), 0);
        vst1_lane_u16(c1, vreinterpret_u16_f16(vacc1), 0);
        vst1_lane_u16(c2, vreinterpret_u16_f16(vacc2), 0);
        vst1_lane_u16(c3, vreinterpret_u16_f16(vacc3), 0);
        vst1_lane_u16(c4, vreinterpret_u16_f16(vacc4), 0);
        vst1_lane_u16(c5, vreinterpret_u16_f16(vacc5), 0);
        vst1_lane_u16(c6, vreinterpret_u16_f16(vacc6), 0);
        vst1_lane_u16(c7, vreinterpret_u16_f16(vacc7), 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
