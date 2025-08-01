// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-gemm/c8-neon-shland.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
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


void xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland(
    size_t mr,
    size_t nc,
    size_t kc,
    const xnn_bfloat16* restrict a,
    size_t a_stride,
    const xnn_bfloat16* restrict w_ptr,
    xnn_bfloat16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_bf16_minmax_params* restrict params)
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
  const uint16x8_t vmask = vreinterpretq_u16_u32(vmovq_n_u32(UINT32_C(0xFFFF0000)));
  do {
    float32x4_t vacc0x0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;

    size_t k = kc;
    for (; k >= 8 * sizeof(uint16_t); k -= 8 * sizeof(uint16_t)) {
      const uint16x8_t va0 = vld1q_u16(a0); a0 += 8;

      const uint16x8_t vb0 = vld1q_u16(w); w += 8;
      const uint16x8_t vb1 = vld1q_u16(w); w += 8;
      const uint16x8_t vb2 = vld1q_u16(w); w += 8;
      const uint16x8_t vb3 = vld1q_u16(w); w += 8;

      const float32x4_t va0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0), 16));

      const float32x4_t vb0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb0), 16));
      const float32x4_t vb1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb1), 16));
      const float32x4_t vb2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb2), 16));
      const float32x4_t vb3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb3), 16));

      vacc0x0 = vfmaq_f32(vacc0x0, va0e, vb0e);
      vacc0x1 = vfmaq_f32(vacc0x1, va0e, vb1e);
      vacc0x2 = vfmaq_f32(vacc0x2, va0e, vb2e);
      vacc0x3 = vfmaq_f32(vacc0x3, va0e, vb3e);

      const float32x4_t va0o = vreinterpretq_f32_u16(vandq_u16(va0, vmask));

      const float32x4_t vb0o = vreinterpretq_f32_u16(vandq_u16(vb0, vmask));
      const float32x4_t vb1o = vreinterpretq_f32_u16(vandq_u16(vb1, vmask));
      const float32x4_t vb2o = vreinterpretq_f32_u16(vandq_u16(vb2, vmask));
      const float32x4_t vb3o = vreinterpretq_f32_u16(vandq_u16(vb3, vmask));

      vacc0x0 = vfmaq_f32(vacc0x0, va0o, vb0o);
      vacc0x1 = vfmaq_f32(vacc0x1, va0o, vb1o);
      vacc0x2 = vfmaq_f32(vacc0x2, va0o, vb2o);
      vacc0x3 = vfmaq_f32(vacc0x3, va0o, vb3o);
    }
    if XNN_UNLIKELY(k != 0) {
      const uint16x8_t va0 = vld1q_u16(a0); a0 = (const uint16_t*) ((uintptr_t) a0 + k);

      const uint16x8_t vb0 = vld1q_u16(w); w += 8;
      const uint16x8_t vb1 = vld1q_u16(w); w += 8;
      const uint16x8_t vb2 = vld1q_u16(w); w += 8;
      const uint16x8_t vb3 = vld1q_u16(w); w += 8;

      const uint16x8_t vm0 = vceqq_u16(vb0, vmovq_n_u16(0));
      const uint16x8_t vm1 = vceqq_u16(vb1, vmovq_n_u16(0));
      const uint16x8_t vm2 = vceqq_u16(vb2, vmovq_n_u16(0));
      const uint16x8_t vm3 = vceqq_u16(vb3, vmovq_n_u16(0));

      const float32x4_t vb0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb0), 16));
      const float32x4_t vb1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb1), 16));
      const float32x4_t vb2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb2), 16));
      const float32x4_t vb3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(vb3), 16));

      const uint16x8_t va0x0 = vbicq_u16(va0, vm0);
      const uint16x8_t va0x1 = vbicq_u16(va0, vm1);
      const uint16x8_t va0x2 = vbicq_u16(va0, vm2);
      const uint16x8_t va0x3 = vbicq_u16(va0, vm3);

      const float32x4_t va0x0e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x0), 16));
      const float32x4_t va0x1e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x1), 16));
      const float32x4_t va0x2e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x2), 16));
      const float32x4_t va0x3e = vreinterpretq_f32_u32(vshlq_n_u32(vreinterpretq_u32_u16(va0x3), 16));

      vacc0x0 = vfmaq_f32(vacc0x0, va0x0e, vb0e);
      vacc0x1 = vfmaq_f32(vacc0x1, va0x1e, vb1e);
      vacc0x2 = vfmaq_f32(vacc0x2, va0x2e, vb2e);
      vacc0x3 = vfmaq_f32(vacc0x3, va0x3e, vb3e);

      const float32x4_t vb0o = vreinterpretq_f32_u16(vandq_u16(vb0, vmask));
      const float32x4_t vb1o = vreinterpretq_f32_u16(vandq_u16(vb1, vmask));
      const float32x4_t vb2o = vreinterpretq_f32_u16(vandq_u16(vb2, vmask));
      const float32x4_t vb3o = vreinterpretq_f32_u16(vandq_u16(vb3, vmask));

      const float32x4_t va0x0o = vreinterpretq_f32_u16(vandq_u16(va0x0, vmask));
      const float32x4_t va0x1o = vreinterpretq_f32_u16(vandq_u16(va0x1, vmask));
      const float32x4_t va0x2o = vreinterpretq_f32_u16(vandq_u16(va0x2, vmask));
      const float32x4_t va0x3o = vreinterpretq_f32_u16(vandq_u16(va0x3, vmask));

      vacc0x0 = vfmaq_f32(vacc0x0, va0x0o, vb0o);
      vacc0x1 = vfmaq_f32(vacc0x1, va0x1o, vb1o);
      vacc0x2 = vfmaq_f32(vacc0x2, va0x2o, vb2o);
      vacc0x3 = vfmaq_f32(vacc0x3, va0x3o, vb3o);
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

    const float32x4_t vmax = vdupq_n_f32(params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);

    const float32x4_t vmin = vdupq_n_f32(params->scalar.min);
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
