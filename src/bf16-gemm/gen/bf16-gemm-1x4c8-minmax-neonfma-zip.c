// Auto-generated file. Do not edit!
//   Template: src/bf16-gemm/c8-neon-zip.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"


void xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w_ptr,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_bf16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
  assert(c != NULL);

  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  const uint16_t* w = (const uint16_t*) w_ptr;
  const uint16x8_t vzero = vmovq_n_u16(0);
  do {
    float32x4_t vacc0x0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;

    size_t k = kc;
    for (; k >= 8 * sizeof(uint16_t); k -= 8 * sizeof(uint16_t)) {
      const uint16x8_t va0h = vld1q_u16(a0); a0 += 8;

      const uint16x8_t vb0h = vld1q_u16(w); w += 8;
      const uint16x8_t vb1h = vld1q_u16(w); w += 8;
      const uint16x8_t vb2h = vld1q_u16(w); w += 8;
      const uint16x8_t vb3h = vld1q_u16(w); w += 8;

      const uint16x8x2_t va0f = vzipq_u16(vzero, va0h);

      const uint16x8x2_t vb0f = vzipq_u16(vzero, vb0h);
      const uint16x8x2_t vb1f = vzipq_u16(vzero, vb1h);
      const uint16x8x2_t vb2f = vzipq_u16(vzero, vb2h);
      const uint16x8x2_t vb3f = vzipq_u16(vzero, vb3h);

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
    }
    if XNN_UNLIKELY(k != 0) {
      const uint16x8_t va0h = vld1q_u16(a0); a0 = (const uint16_t*) ((uintptr_t) a0 + k);

      const uint16x8_t vb0h = vld1q_u16(w); w += 8;
      const uint16x8_t vb1h = vld1q_u16(w); w += 8;
      const uint16x8_t vb2h = vld1q_u16(w); w += 8;
      const uint16x8_t vb3h = vld1q_u16(w); w += 8;

      const uint16x8_t vm0h = vceqq_u16(vb0h, vmovq_n_u16(0));
      const uint16x8_t vm1h = vceqq_u16(vb1h, vmovq_n_u16(0));
      const uint16x8_t vm2h = vceqq_u16(vb2h, vmovq_n_u16(0));
      const uint16x8_t vm3h = vceqq_u16(vb3h, vmovq_n_u16(0));

      const uint16x8x2_t vb0f = vzipq_u16(vzero, vb0h);
      const uint16x8x2_t vb1f = vzipq_u16(vzero, vb1h);
      const uint16x8x2_t vb2f = vzipq_u16(vzero, vb2h);
      const uint16x8x2_t vb3f = vzipq_u16(vzero, vb3h);

      const uint16x8_t va0x0h = vbicq_u16(va0h, vm0h);
      const uint16x8_t va0x1h = vbicq_u16(va0h, vm1h);
      const uint16x8_t va0x2h = vbicq_u16(va0h, vm2h);
      const uint16x8_t va0x3h = vbicq_u16(va0h, vm3h);

      const uint16x8x2_t va0x0f = vzipq_u16(vzero, va0x0h);
      const uint16x8x2_t va0x1f = vzipq_u16(vzero, va0x1h);
      const uint16x8x2_t va0x2f = vzipq_u16(vzero, va0x2h);
      const uint16x8x2_t va0x3f = vzipq_u16(vzero, va0x3h);

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0x0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0x1f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0x2f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0x3f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0x0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0x1f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0x2f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0x3f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
    }

#if XNN_ARCH_ARM64
    const float32x4_t vacc0x01 = vpaddq_f32(vacc0x0, vacc0x1);
    const float32x4_t vacc0x23 = vpaddq_f32(vacc0x2, vacc0x3);

    float32x4_t vacc0x0123 = vpaddq_f32(vacc0x01, vacc0x23);
#else
    const float32x2_t vsum0x0 = vadd_f32(vget_low_f32(vacc0x0), vget_high_f32(vacc0x0));
    const float32x2_t vsum0x1 = vadd_f32(vget_low_f32(vacc0x1), vget_high_f32(vacc0x1));
    const float32x2_t vsum0x2 = vadd_f32(vget_low_f32(vacc0x2), vget_high_f32(vacc0x2));
    const float32x2_t vsum0x3 = vadd_f32(vget_low_f32(vacc0x3), vget_high_f32(vacc0x3));

    float32x4_t vacc0x0123 = vcombine_f32(vpadd_f32(vsum0x0, vsum0x1), vpadd_f32(vsum0x2, vsum0x3));
#endif

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);

    uint16x4_t vout0x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc0x0123), 16);

    if XNN_LIKELY(nc >= 4) {
      vst1_u16(c0, vout0x0123);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_u16(vout0x0123), 0); c0 += 2;

        vout0x0123 = vext_u16(vout0x0123, vout0x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vout0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
