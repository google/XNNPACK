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


void xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
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
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const uint16_t* w = (const uint16_t*) w_ptr;
  const uint16x8_t vzero = vmovq_n_u16(0);
  do {
    float32x4_t vacc0x0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc0x3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_lane_u16(w, vdup_n_u16(0), 0), 16)); w += 1;
    float32x4_t vacc1x0 = vacc0x0;
    float32x4_t vacc1x1 = vacc0x1;
    float32x4_t vacc1x2 = vacc0x2;
    float32x4_t vacc1x3 = vacc0x3;
    float32x4_t vacc2x0 = vacc0x0;
    float32x4_t vacc2x1 = vacc0x1;
    float32x4_t vacc2x2 = vacc0x2;
    float32x4_t vacc2x3 = vacc0x3;
    float32x4_t vacc3x0 = vacc0x0;
    float32x4_t vacc3x1 = vacc0x1;
    float32x4_t vacc3x2 = vacc0x2;
    float32x4_t vacc3x3 = vacc0x3;

    size_t k = kc;
    for (; k >= 8 * sizeof(uint16_t); k -= 8 * sizeof(uint16_t)) {
      const uint16x8_t va0h = vld1q_u16(a0); a0 += 8;
      const uint16x8_t va1h = vld1q_u16(a1); a1 += 8;
      const uint16x8_t va2h = vld1q_u16(a2); a2 += 8;
      const uint16x8_t va3h = vld1q_u16(a3); a3 += 8;

      const uint16x8_t vb0h = vld1q_u16(w); w += 8;
      const uint16x8_t vb1h = vld1q_u16(w); w += 8;
      const uint16x8_t vb2h = vld1q_u16(w); w += 8;
      const uint16x8_t vb3h = vld1q_u16(w); w += 8;

      const uint16x8x2_t va0f = vzipq_u16(vzero, va0h);
      const uint16x8x2_t va1f = vzipq_u16(vzero, va1h);
      const uint16x8x2_t va2f = vzipq_u16(vzero, va2h);
      const uint16x8x2_t va3f = vzipq_u16(vzero, va3h);

      const uint16x8x2_t vb0f = vzipq_u16(vzero, vb0h);
      const uint16x8x2_t vb1f = vzipq_u16(vzero, vb1h);
      const uint16x8x2_t vb2f = vzipq_u16(vzero, vb2h);
      const uint16x8x2_t vb3f = vzipq_u16(vzero, vb3h);

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc1x0 = vfmaq_f32(vacc1x0, vreinterpretq_f32_u16(va1f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc2x0 = vfmaq_f32(vacc2x0, vreinterpretq_f32_u16(va2f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc3x0 = vfmaq_f32(vacc3x0, vreinterpretq_f32_u16(va3f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc1x1 = vfmaq_f32(vacc1x1, vreinterpretq_f32_u16(va1f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc2x1 = vfmaq_f32(vacc2x1, vreinterpretq_f32_u16(va2f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc3x1 = vfmaq_f32(vacc3x1, vreinterpretq_f32_u16(va3f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc1x2 = vfmaq_f32(vacc1x2, vreinterpretq_f32_u16(va1f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc2x2 = vfmaq_f32(vacc2x2, vreinterpretq_f32_u16(va2f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc3x2 = vfmaq_f32(vacc3x2, vreinterpretq_f32_u16(va3f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));
      vacc1x3 = vfmaq_f32(vacc1x3, vreinterpretq_f32_u16(va1f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));
      vacc2x3 = vfmaq_f32(vacc2x3, vreinterpretq_f32_u16(va2f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));
      vacc3x3 = vfmaq_f32(vacc3x3, vreinterpretq_f32_u16(va3f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc1x0 = vfmaq_f32(vacc1x0, vreinterpretq_f32_u16(va1f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc2x0 = vfmaq_f32(vacc2x0, vreinterpretq_f32_u16(va2f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc3x0 = vfmaq_f32(vacc3x0, vreinterpretq_f32_u16(va3f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc1x1 = vfmaq_f32(vacc1x1, vreinterpretq_f32_u16(va1f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc2x1 = vfmaq_f32(vacc2x1, vreinterpretq_f32_u16(va2f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc3x1 = vfmaq_f32(vacc3x1, vreinterpretq_f32_u16(va3f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc1x2 = vfmaq_f32(vacc1x2, vreinterpretq_f32_u16(va1f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc2x2 = vfmaq_f32(vacc2x2, vreinterpretq_f32_u16(va2f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc3x2 = vfmaq_f32(vacc3x2, vreinterpretq_f32_u16(va3f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
      vacc1x3 = vfmaq_f32(vacc1x3, vreinterpretq_f32_u16(va1f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
      vacc2x3 = vfmaq_f32(vacc2x3, vreinterpretq_f32_u16(va2f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
      vacc3x3 = vfmaq_f32(vacc3x3, vreinterpretq_f32_u16(va3f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
    }
    if XNN_UNLIKELY(k != 0) {
      const uint16x8_t va0h = vld1q_u16(a0); a0 = (const uint16_t*) ((uintptr_t) a0 + k);
      const uint16x8_t va1h = vld1q_u16(a1); a1 = (const uint16_t*) ((uintptr_t) a1 + k);
      const uint16x8_t va2h = vld1q_u16(a2); a2 = (const uint16_t*) ((uintptr_t) a2 + k);
      const uint16x8_t va3h = vld1q_u16(a3); a3 = (const uint16_t*) ((uintptr_t) a3 + k);

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
      const uint16x8_t va1x0h = vbicq_u16(va1h, vm0h);
      const uint16x8_t va2x0h = vbicq_u16(va2h, vm0h);
      const uint16x8_t va3x0h = vbicq_u16(va3h, vm0h);
      const uint16x8_t va0x1h = vbicq_u16(va0h, vm1h);
      const uint16x8_t va1x1h = vbicq_u16(va1h, vm1h);
      const uint16x8_t va2x1h = vbicq_u16(va2h, vm1h);
      const uint16x8_t va3x1h = vbicq_u16(va3h, vm1h);
      const uint16x8_t va0x2h = vbicq_u16(va0h, vm2h);
      const uint16x8_t va1x2h = vbicq_u16(va1h, vm2h);
      const uint16x8_t va2x2h = vbicq_u16(va2h, vm2h);
      const uint16x8_t va3x2h = vbicq_u16(va3h, vm2h);
      const uint16x8_t va0x3h = vbicq_u16(va0h, vm3h);
      const uint16x8_t va1x3h = vbicq_u16(va1h, vm3h);
      const uint16x8_t va2x3h = vbicq_u16(va2h, vm3h);
      const uint16x8_t va3x3h = vbicq_u16(va3h, vm3h);

      const uint16x8x2_t va0x0f = vzipq_u16(vzero, va0x0h);
      const uint16x8x2_t va1x0f = vzipq_u16(vzero, va1x0h);
      const uint16x8x2_t va2x0f = vzipq_u16(vzero, va2x0h);
      const uint16x8x2_t va3x0f = vzipq_u16(vzero, va3x0h);
      const uint16x8x2_t va0x1f = vzipq_u16(vzero, va0x1h);
      const uint16x8x2_t va1x1f = vzipq_u16(vzero, va1x1h);
      const uint16x8x2_t va2x1f = vzipq_u16(vzero, va2x1h);
      const uint16x8x2_t va3x1f = vzipq_u16(vzero, va3x1h);
      const uint16x8x2_t va0x2f = vzipq_u16(vzero, va0x2h);
      const uint16x8x2_t va1x2f = vzipq_u16(vzero, va1x2h);
      const uint16x8x2_t va2x2f = vzipq_u16(vzero, va2x2h);
      const uint16x8x2_t va3x2f = vzipq_u16(vzero, va3x2h);
      const uint16x8x2_t va0x3f = vzipq_u16(vzero, va0x3h);
      const uint16x8x2_t va1x3f = vzipq_u16(vzero, va1x3h);
      const uint16x8x2_t va2x3f = vzipq_u16(vzero, va2x3h);
      const uint16x8x2_t va3x3f = vzipq_u16(vzero, va3x3h);

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0x0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc1x0 = vfmaq_f32(vacc1x0, vreinterpretq_f32_u16(va1x0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc2x0 = vfmaq_f32(vacc2x0, vreinterpretq_f32_u16(va2x0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc3x0 = vfmaq_f32(vacc3x0, vreinterpretq_f32_u16(va3x0f.val[0]), vreinterpretq_f32_u16(vb0f.val[0]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0x1f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc1x1 = vfmaq_f32(vacc1x1, vreinterpretq_f32_u16(va1x1f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc2x1 = vfmaq_f32(vacc2x1, vreinterpretq_f32_u16(va2x1f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc3x1 = vfmaq_f32(vacc3x1, vreinterpretq_f32_u16(va3x1f.val[0]), vreinterpretq_f32_u16(vb1f.val[0]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0x2f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc1x2 = vfmaq_f32(vacc1x2, vreinterpretq_f32_u16(va1x2f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc2x2 = vfmaq_f32(vacc2x2, vreinterpretq_f32_u16(va2x2f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc3x2 = vfmaq_f32(vacc3x2, vreinterpretq_f32_u16(va3x2f.val[0]), vreinterpretq_f32_u16(vb2f.val[0]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0x3f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));
      vacc1x3 = vfmaq_f32(vacc1x3, vreinterpretq_f32_u16(va1x3f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));
      vacc2x3 = vfmaq_f32(vacc2x3, vreinterpretq_f32_u16(va2x3f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));
      vacc3x3 = vfmaq_f32(vacc3x3, vreinterpretq_f32_u16(va3x3f.val[0]), vreinterpretq_f32_u16(vb3f.val[0]));

      vacc0x0 = vfmaq_f32(vacc0x0, vreinterpretq_f32_u16(va0x0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc1x0 = vfmaq_f32(vacc1x0, vreinterpretq_f32_u16(va1x0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc2x0 = vfmaq_f32(vacc2x0, vreinterpretq_f32_u16(va2x0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc3x0 = vfmaq_f32(vacc3x0, vreinterpretq_f32_u16(va3x0f.val[1]), vreinterpretq_f32_u16(vb0f.val[1]));
      vacc0x1 = vfmaq_f32(vacc0x1, vreinterpretq_f32_u16(va0x1f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc1x1 = vfmaq_f32(vacc1x1, vreinterpretq_f32_u16(va1x1f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc2x1 = vfmaq_f32(vacc2x1, vreinterpretq_f32_u16(va2x1f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc3x1 = vfmaq_f32(vacc3x1, vreinterpretq_f32_u16(va3x1f.val[1]), vreinterpretq_f32_u16(vb1f.val[1]));
      vacc0x2 = vfmaq_f32(vacc0x2, vreinterpretq_f32_u16(va0x2f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc1x2 = vfmaq_f32(vacc1x2, vreinterpretq_f32_u16(va1x2f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc2x2 = vfmaq_f32(vacc2x2, vreinterpretq_f32_u16(va2x2f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc3x2 = vfmaq_f32(vacc3x2, vreinterpretq_f32_u16(va3x2f.val[1]), vreinterpretq_f32_u16(vb2f.val[1]));
      vacc0x3 = vfmaq_f32(vacc0x3, vreinterpretq_f32_u16(va0x3f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
      vacc1x3 = vfmaq_f32(vacc1x3, vreinterpretq_f32_u16(va1x3f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
      vacc2x3 = vfmaq_f32(vacc2x3, vreinterpretq_f32_u16(va2x3f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
      vacc3x3 = vfmaq_f32(vacc3x3, vreinterpretq_f32_u16(va3x3f.val[1]), vreinterpretq_f32_u16(vb3f.val[1]));
    }

#if XNN_ARCH_ARM64
    const float32x4_t vacc0x01 = vpaddq_f32(vacc0x0, vacc0x1);
    const float32x4_t vacc1x01 = vpaddq_f32(vacc1x0, vacc1x1);
    const float32x4_t vacc2x01 = vpaddq_f32(vacc2x0, vacc2x1);
    const float32x4_t vacc3x01 = vpaddq_f32(vacc3x0, vacc3x1);
    const float32x4_t vacc0x23 = vpaddq_f32(vacc0x2, vacc0x3);
    const float32x4_t vacc1x23 = vpaddq_f32(vacc1x2, vacc1x3);
    const float32x4_t vacc2x23 = vpaddq_f32(vacc2x2, vacc2x3);
    const float32x4_t vacc3x23 = vpaddq_f32(vacc3x2, vacc3x3);

    float32x4_t vacc0x0123 = vpaddq_f32(vacc0x01, vacc0x23);
    float32x4_t vacc1x0123 = vpaddq_f32(vacc1x01, vacc1x23);
    float32x4_t vacc2x0123 = vpaddq_f32(vacc2x01, vacc2x23);
    float32x4_t vacc3x0123 = vpaddq_f32(vacc3x01, vacc3x23);
#else
    const float32x2_t vsum0x0 = vadd_f32(vget_low_f32(vacc0x0), vget_high_f32(vacc0x0));
    const float32x2_t vsum1x0 = vadd_f32(vget_low_f32(vacc1x0), vget_high_f32(vacc1x0));
    const float32x2_t vsum2x0 = vadd_f32(vget_low_f32(vacc2x0), vget_high_f32(vacc2x0));
    const float32x2_t vsum3x0 = vadd_f32(vget_low_f32(vacc3x0), vget_high_f32(vacc3x0));
    const float32x2_t vsum0x1 = vadd_f32(vget_low_f32(vacc0x1), vget_high_f32(vacc0x1));
    const float32x2_t vsum1x1 = vadd_f32(vget_low_f32(vacc1x1), vget_high_f32(vacc1x1));
    const float32x2_t vsum2x1 = vadd_f32(vget_low_f32(vacc2x1), vget_high_f32(vacc2x1));
    const float32x2_t vsum3x1 = vadd_f32(vget_low_f32(vacc3x1), vget_high_f32(vacc3x1));
    const float32x2_t vsum0x2 = vadd_f32(vget_low_f32(vacc0x2), vget_high_f32(vacc0x2));
    const float32x2_t vsum1x2 = vadd_f32(vget_low_f32(vacc1x2), vget_high_f32(vacc1x2));
    const float32x2_t vsum2x2 = vadd_f32(vget_low_f32(vacc2x2), vget_high_f32(vacc2x2));
    const float32x2_t vsum3x2 = vadd_f32(vget_low_f32(vacc3x2), vget_high_f32(vacc3x2));
    const float32x2_t vsum0x3 = vadd_f32(vget_low_f32(vacc0x3), vget_high_f32(vacc0x3));
    const float32x2_t vsum1x3 = vadd_f32(vget_low_f32(vacc1x3), vget_high_f32(vacc1x3));
    const float32x2_t vsum2x3 = vadd_f32(vget_low_f32(vacc2x3), vget_high_f32(vacc2x3));
    const float32x2_t vsum3x3 = vadd_f32(vget_low_f32(vacc3x3), vget_high_f32(vacc3x3));

    float32x4_t vacc0x0123 = vcombine_f32(vpadd_f32(vsum0x0, vsum0x1), vpadd_f32(vsum0x2, vsum0x3));
    float32x4_t vacc1x0123 = vcombine_f32(vpadd_f32(vsum1x0, vsum1x1), vpadd_f32(vsum1x2, vsum1x3));
    float32x4_t vacc2x0123 = vcombine_f32(vpadd_f32(vsum2x0, vsum2x1), vpadd_f32(vsum2x2, vsum2x3));
    float32x4_t vacc3x0123 = vcombine_f32(vpadd_f32(vsum3x0, vsum3x1), vpadd_f32(vsum3x2, vsum3x3));
#endif

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);

    uint16x4_t vout0x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc0x0123), 16);
    uint16x4_t vout1x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc1x0123), 16);
    uint16x4_t vout2x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc2x0123), 16);
    uint16x4_t vout3x0123 = vshrn_n_u32(vreinterpretq_u32_f32(vacc3x0123), 16);

    if XNN_LIKELY(nc >= 4) {
      vst1_u16(c0, vout0x0123);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      vst1_u16(c1, vout1x0123);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      vst1_u16(c2, vout2x0123);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      vst1_u16(c3, vout3x0123);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_u16(vout0x0123), 0); c0 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_u16(vout1x0123), 0); c1 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_u16(vout2x0123), 0); c2 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_u16(vout3x0123), 0); c3 += 2;

        vout0x0123 = vext_u16(vout0x0123, vout0x0123, 2);
        vout1x0123 = vext_u16(vout1x0123, vout1x0123, 2);
        vout2x0123 = vext_u16(vout2x0123, vout2x0123, 2);
        vout3x0123 = vext_u16(vout3x0123, vout3x0123, 2);
      }
      if (nc & 1) {
        vst1_lane_u16(c0, vout0x0123, 0);
        vst1_lane_u16(c1, vout1x0123, 0);
        vst1_lane_u16(c2, vout2x0123, 0);
        vst1_lane_u16(c3, vout3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
