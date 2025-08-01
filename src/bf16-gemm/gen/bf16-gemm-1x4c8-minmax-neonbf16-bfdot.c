// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-gemm/c8-neonbf16.c.in
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


void xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot(
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
  assert(kc % sizeof(bfloat16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
  assert(c != NULL);

  const bfloat16_t* a0 = (const bfloat16_t*) a;
  bfloat16_t* c0 = (bfloat16_t*) c;

  const bfloat16_t* w = (const bfloat16_t*) w_ptr;
  do {
    float32x4_t vacc0x0 = vcvt_f32_bf16(vld1_lane_bf16(w, vreinterpret_bf16_u16(vdup_n_u16(0)), 0)); w += 1;
    float32x4_t vacc0x1 = vcvt_f32_bf16(vld1_lane_bf16(w, vreinterpret_bf16_u16(vdup_n_u16(0)), 0)); w += 1;
    float32x4_t vacc0x2 = vcvt_f32_bf16(vld1_lane_bf16(w, vreinterpret_bf16_u16(vdup_n_u16(0)), 0)); w += 1;
    float32x4_t vacc0x3 = vcvt_f32_bf16(vld1_lane_bf16(w, vreinterpret_bf16_u16(vdup_n_u16(0)), 0)); w += 1;

    size_t k = kc;
    for (; k >= 8 * sizeof(bfloat16_t); k -= 8 * sizeof(bfloat16_t)) {
      const bfloat16x8_t va0 = vld1q_bf16(a0); a0 += 8;

      const bfloat16x8_t vb0 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb1 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb2 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb3 = vld1q_bf16(w); w += 8;

      vacc0x0 = vbfdotq_f32(vacc0x0, va0, vb0);
      vacc0x1 = vbfdotq_f32(vacc0x1, va0, vb1);
      vacc0x2 = vbfdotq_f32(vacc0x2, va0, vb2);
      vacc0x3 = vbfdotq_f32(vacc0x3, va0, vb3);
    }
    if XNN_UNLIKELY(k != 0) {
      const bfloat16x8_t va0 = vld1q_bf16(a0); a0 = (const bfloat16_t*) ((uintptr_t) a0 + k);

      const bfloat16x8_t vb0 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb1 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb2 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb3 = vld1q_bf16(w); w += 8;

      const uint16x8_t vm0 = vceqq_u16(vreinterpretq_u16_bf16(vb0), vmovq_n_u16(0));
      const uint16x8_t vm1 = vceqq_u16(vreinterpretq_u16_bf16(vb1), vmovq_n_u16(0));
      const uint16x8_t vm2 = vceqq_u16(vreinterpretq_u16_bf16(vb2), vmovq_n_u16(0));
      const uint16x8_t vm3 = vceqq_u16(vreinterpretq_u16_bf16(vb3), vmovq_n_u16(0));

      const bfloat16x8_t va0x0 = vreinterpretq_bf16_u16(vbicq_u16(vreinterpretq_u16_bf16(va0), vm0));
      vacc0x0 = vbfdotq_f32(vacc0x0, va0x0, vb0);
      const bfloat16x8_t va0x1 = vreinterpretq_bf16_u16(vbicq_u16(vreinterpretq_u16_bf16(va0), vm1));
      vacc0x1 = vbfdotq_f32(vacc0x1, va0x1, vb1);
      const bfloat16x8_t va0x2 = vreinterpretq_bf16_u16(vbicq_u16(vreinterpretq_u16_bf16(va0), vm2));
      vacc0x2 = vbfdotq_f32(vacc0x2, va0x2, vb2);
      const bfloat16x8_t va0x3 = vreinterpretq_bf16_u16(vbicq_u16(vreinterpretq_u16_bf16(va0), vm3));
      vacc0x3 = vbfdotq_f32(vacc0x3, va0x3, vb3);
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

    bfloat16x4_t vout0x0123 = vcvt_bf16_f32(vacc0x0123);

    if XNN_LIKELY(nc >= 4) {
      vst1_bf16(c0, vout0x0123);
      c0 = (bfloat16_t*) ((uintptr_t) c0 + cn_stride);

      a0 = (const bfloat16_t*) ((uintptr_t) a0 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_bf16(vout0x0123), 0); c0 += 2;

        vout0x0123 = vreinterpret_bf16_u16(vext_u16(vreinterpret_u16_bf16(vout0x0123), vreinterpret_u16_bf16(vout0x0123), 2));
      }
      if (nc & 1) {
        vst1_lane_bf16(c0, vout0x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
