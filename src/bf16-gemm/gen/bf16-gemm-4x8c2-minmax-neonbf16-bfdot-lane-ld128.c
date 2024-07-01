// Auto-generated file. Do not edit!
//   Template: src/bf16-gemm/c2-neonbf16-bfdot-lane-ld128.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gemm.h"


void xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128(
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
  assert(kc % sizeof(bfloat16_t) == 0);
  assert(a != NULL);
  assert(w_ptr != NULL);
  assert(c != NULL);

  const bfloat16_t* a0 = (const bfloat16_t*) a;
  bfloat16_t* c0 = (bfloat16_t*) c;
  const bfloat16_t* a1 = (const bfloat16_t*) ((uintptr_t) a0 + a_stride);
  bfloat16_t* c1 = (bfloat16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const bfloat16_t* a2 = (const bfloat16_t*) ((uintptr_t) a1 + a_stride);
  bfloat16_t* c2 = (bfloat16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const bfloat16_t* a3 = (const bfloat16_t*) ((uintptr_t) a2 + a_stride);
  bfloat16_t* c3 = (bfloat16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const bfloat16_t* w = (const bfloat16_t*) w_ptr;
  do {
    float32x4_t vacc0x0123 = vcvt_f32_bf16(vld1_bf16(w)); w += 4;
    float32x4_t vacc0x4567 = vcvt_f32_bf16(vld1_bf16(w)); w += 4;
    float32x4_t vacc1x0123 = vacc0x0123;
    float32x4_t vacc1x4567 = vacc0x4567;
    float32x4_t vacc2x0123 = vacc0x0123;
    float32x4_t vacc2x4567 = vacc0x4567;
    float32x4_t vacc3x0123 = vacc0x0123;
    float32x4_t vacc3x4567 = vacc0x4567;

    size_t k = kc;
    for (; k >= 8 * sizeof(bfloat16_t); k -= 8 * sizeof(bfloat16_t)) {
      const bfloat16x8_t va0 = vld1q_bf16(a0); a0 += 8;
      const bfloat16x8_t va1 = vld1q_bf16(a1); a1 += 8;
      const bfloat16x8_t va2 = vld1q_bf16(a2); a2 += 8;
      const bfloat16x8_t va3 = vld1q_bf16(a3); a3 += 8;

      const bfloat16x8_t vb0123c01 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb4567c01 = vld1q_bf16(w); w += 8;

      vacc0x0123 = vbfdotq_laneq_f32(vacc0x0123, vb0123c01, va0, 0);
      vacc1x0123 = vbfdotq_laneq_f32(vacc1x0123, vb0123c01, va1, 0);
      vacc2x0123 = vbfdotq_laneq_f32(vacc2x0123, vb0123c01, va2, 0);
      vacc3x0123 = vbfdotq_laneq_f32(vacc3x0123, vb0123c01, va3, 0);
      vacc0x4567 = vbfdotq_laneq_f32(vacc0x4567, vb4567c01, va0, 0);
      vacc1x4567 = vbfdotq_laneq_f32(vacc1x4567, vb4567c01, va1, 0);
      vacc2x4567 = vbfdotq_laneq_f32(vacc2x4567, vb4567c01, va2, 0);
      vacc3x4567 = vbfdotq_laneq_f32(vacc3x4567, vb4567c01, va3, 0);
      const bfloat16x8_t vb0123c23 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb4567c23 = vld1q_bf16(w); w += 8;

      vacc0x0123 = vbfdotq_laneq_f32(vacc0x0123, vb0123c23, va0, 1);
      vacc1x0123 = vbfdotq_laneq_f32(vacc1x0123, vb0123c23, va1, 1);
      vacc2x0123 = vbfdotq_laneq_f32(vacc2x0123, vb0123c23, va2, 1);
      vacc3x0123 = vbfdotq_laneq_f32(vacc3x0123, vb0123c23, va3, 1);
      vacc0x4567 = vbfdotq_laneq_f32(vacc0x4567, vb4567c23, va0, 1);
      vacc1x4567 = vbfdotq_laneq_f32(vacc1x4567, vb4567c23, va1, 1);
      vacc2x4567 = vbfdotq_laneq_f32(vacc2x4567, vb4567c23, va2, 1);
      vacc3x4567 = vbfdotq_laneq_f32(vacc3x4567, vb4567c23, va3, 1);
      const bfloat16x8_t vb0123c45 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb4567c45 = vld1q_bf16(w); w += 8;

      vacc0x0123 = vbfdotq_laneq_f32(vacc0x0123, vb0123c45, va0, 2);
      vacc1x0123 = vbfdotq_laneq_f32(vacc1x0123, vb0123c45, va1, 2);
      vacc2x0123 = vbfdotq_laneq_f32(vacc2x0123, vb0123c45, va2, 2);
      vacc3x0123 = vbfdotq_laneq_f32(vacc3x0123, vb0123c45, va3, 2);
      vacc0x4567 = vbfdotq_laneq_f32(vacc0x4567, vb4567c45, va0, 2);
      vacc1x4567 = vbfdotq_laneq_f32(vacc1x4567, vb4567c45, va1, 2);
      vacc2x4567 = vbfdotq_laneq_f32(vacc2x4567, vb4567c45, va2, 2);
      vacc3x4567 = vbfdotq_laneq_f32(vacc3x4567, vb4567c45, va3, 2);
      const bfloat16x8_t vb0123c67 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb4567c67 = vld1q_bf16(w); w += 8;

      vacc0x0123 = vbfdotq_laneq_f32(vacc0x0123, vb0123c67, va0, 3);
      vacc1x0123 = vbfdotq_laneq_f32(vacc1x0123, vb0123c67, va1, 3);
      vacc2x0123 = vbfdotq_laneq_f32(vacc2x0123, vb0123c67, va2, 3);
      vacc3x0123 = vbfdotq_laneq_f32(vacc3x0123, vb0123c67, va3, 3);
      vacc0x4567 = vbfdotq_laneq_f32(vacc0x4567, vb4567c67, va0, 3);
      vacc1x4567 = vbfdotq_laneq_f32(vacc1x4567, vb4567c67, va1, 3);
      vacc2x4567 = vbfdotq_laneq_f32(vacc2x4567, vb4567c67, va2, 3);
      vacc3x4567 = vbfdotq_laneq_f32(vacc3x4567, vb4567c67, va3, 3);
    }
    if XNN_UNLIKELY(k != 0) {
      const bfloat16x8_t va0 = vld1q_bf16(a0); a0 = (const bfloat16_t*) ((uintptr_t) a0 + k);
      const bfloat16x8_t va1 = vld1q_bf16(a1); a1 = (const bfloat16_t*) ((uintptr_t) a1 + k);
      const bfloat16x8_t va2 = vld1q_bf16(a2); a2 = (const bfloat16_t*) ((uintptr_t) a2 + k);
      const bfloat16x8_t va3 = vld1q_bf16(a3); a3 = (const bfloat16_t*) ((uintptr_t) a3 + k);

      const bfloat16x8_t vb0123c01 = vld1q_bf16(w); w += 8;
      const bfloat16x8_t vb4567c01 = vld1q_bf16(w); w += 8;

      const uint32x4_t va0c01 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va0)), 0);
      const uint32x4_t va1c01 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va1)), 0);
      const uint32x4_t va2c01 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va2)), 0);
      const uint32x4_t va3c01 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va3)), 0);

      const uint32x4_t vm0123c01 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb0123c01), vmovq_n_u16(0)));
      const uint32x4_t vm4567c01 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb4567c01), vmovq_n_u16(0)));

      const uint32x4_t va0x0123c01 = vbicq_u32(va0c01, vm0123c01);
      vacc0x0123 = vbfdotq_f32(vacc0x0123, vb0123c01, vreinterpretq_bf16_u32(va0x0123c01));
      const uint32x4_t va1x0123c01 = vbicq_u32(va1c01, vm0123c01);
      vacc1x0123 = vbfdotq_f32(vacc1x0123, vb0123c01, vreinterpretq_bf16_u32(va1x0123c01));
      const uint32x4_t va2x0123c01 = vbicq_u32(va2c01, vm0123c01);
      vacc2x0123 = vbfdotq_f32(vacc2x0123, vb0123c01, vreinterpretq_bf16_u32(va2x0123c01));
      const uint32x4_t va3x0123c01 = vbicq_u32(va3c01, vm0123c01);
      vacc3x0123 = vbfdotq_f32(vacc3x0123, vb0123c01, vreinterpretq_bf16_u32(va3x0123c01));
      const uint32x4_t va0x4567c01 = vbicq_u32(va0c01, vm4567c01);
      vacc0x4567 = vbfdotq_f32(vacc0x4567, vb4567c01, vreinterpretq_bf16_u32(va0x4567c01));
      const uint32x4_t va1x4567c01 = vbicq_u32(va1c01, vm4567c01);
      vacc1x4567 = vbfdotq_f32(vacc1x4567, vb4567c01, vreinterpretq_bf16_u32(va1x4567c01));
      const uint32x4_t va2x4567c01 = vbicq_u32(va2c01, vm4567c01);
      vacc2x4567 = vbfdotq_f32(vacc2x4567, vb4567c01, vreinterpretq_bf16_u32(va2x4567c01));
      const uint32x4_t va3x4567c01 = vbicq_u32(va3c01, vm4567c01);
      vacc3x4567 = vbfdotq_f32(vacc3x4567, vb4567c01, vreinterpretq_bf16_u32(va3x4567c01));

      if (k > 2 * sizeof(bfloat16_t)) {
        const bfloat16x8_t vb0123c23 = vld1q_bf16(w); w += 8;
        const bfloat16x8_t vb4567c23 = vld1q_bf16(w); w += 8;

        const uint32x4_t va0c23 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va0)), 1);
        const uint32x4_t va1c23 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va1)), 1);
        const uint32x4_t va2c23 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va2)), 1);
        const uint32x4_t va3c23 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_low_bf16(va3)), 1);

        const uint32x4_t vm0123c23 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb0123c23), vmovq_n_u16(0)));
        const uint32x4_t vm4567c23 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb4567c23), vmovq_n_u16(0)));

        const uint32x4_t va0x0123c23 = vbicq_u32(va0c23, vm0123c23);
        vacc0x0123 = vbfdotq_f32(vacc0x0123, vb0123c23, vreinterpretq_bf16_u32(va0x0123c23));
        const uint32x4_t va1x0123c23 = vbicq_u32(va1c23, vm0123c23);
        vacc1x0123 = vbfdotq_f32(vacc1x0123, vb0123c23, vreinterpretq_bf16_u32(va1x0123c23));
        const uint32x4_t va2x0123c23 = vbicq_u32(va2c23, vm0123c23);
        vacc2x0123 = vbfdotq_f32(vacc2x0123, vb0123c23, vreinterpretq_bf16_u32(va2x0123c23));
        const uint32x4_t va3x0123c23 = vbicq_u32(va3c23, vm0123c23);
        vacc3x0123 = vbfdotq_f32(vacc3x0123, vb0123c23, vreinterpretq_bf16_u32(va3x0123c23));
        const uint32x4_t va0x4567c23 = vbicq_u32(va0c23, vm4567c23);
        vacc0x4567 = vbfdotq_f32(vacc0x4567, vb4567c23, vreinterpretq_bf16_u32(va0x4567c23));
        const uint32x4_t va1x4567c23 = vbicq_u32(va1c23, vm4567c23);
        vacc1x4567 = vbfdotq_f32(vacc1x4567, vb4567c23, vreinterpretq_bf16_u32(va1x4567c23));
        const uint32x4_t va2x4567c23 = vbicq_u32(va2c23, vm4567c23);
        vacc2x4567 = vbfdotq_f32(vacc2x4567, vb4567c23, vreinterpretq_bf16_u32(va2x4567c23));
        const uint32x4_t va3x4567c23 = vbicq_u32(va3c23, vm4567c23);
        vacc3x4567 = vbfdotq_f32(vacc3x4567, vb4567c23, vreinterpretq_bf16_u32(va3x4567c23));

        if (k > 4 * sizeof(bfloat16_t)) {
          const bfloat16x8_t vb0123c45 = vld1q_bf16(w); w += 8;
          const bfloat16x8_t vb4567c45 = vld1q_bf16(w); w += 8;

          const uint32x4_t va0c45 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va0)), 0);
          const uint32x4_t va1c45 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va1)), 0);
          const uint32x4_t va2c45 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va2)), 0);
          const uint32x4_t va3c45 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va3)), 0);

          const uint32x4_t vm0123c45 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb0123c45), vmovq_n_u16(0)));
          const uint32x4_t vm4567c45 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb4567c45), vmovq_n_u16(0)));

          const uint32x4_t va0x0123c45 = vbicq_u32(va0c45, vm0123c45);
          vacc0x0123 = vbfdotq_f32(vacc0x0123, vb0123c45, vreinterpretq_bf16_u32(va0x0123c45));
          const uint32x4_t va1x0123c45 = vbicq_u32(va1c45, vm0123c45);
          vacc1x0123 = vbfdotq_f32(vacc1x0123, vb0123c45, vreinterpretq_bf16_u32(va1x0123c45));
          const uint32x4_t va2x0123c45 = vbicq_u32(va2c45, vm0123c45);
          vacc2x0123 = vbfdotq_f32(vacc2x0123, vb0123c45, vreinterpretq_bf16_u32(va2x0123c45));
          const uint32x4_t va3x0123c45 = vbicq_u32(va3c45, vm0123c45);
          vacc3x0123 = vbfdotq_f32(vacc3x0123, vb0123c45, vreinterpretq_bf16_u32(va3x0123c45));
          const uint32x4_t va0x4567c45 = vbicq_u32(va0c45, vm4567c45);
          vacc0x4567 = vbfdotq_f32(vacc0x4567, vb4567c45, vreinterpretq_bf16_u32(va0x4567c45));
          const uint32x4_t va1x4567c45 = vbicq_u32(va1c45, vm4567c45);
          vacc1x4567 = vbfdotq_f32(vacc1x4567, vb4567c45, vreinterpretq_bf16_u32(va1x4567c45));
          const uint32x4_t va2x4567c45 = vbicq_u32(va2c45, vm4567c45);
          vacc2x4567 = vbfdotq_f32(vacc2x4567, vb4567c45, vreinterpretq_bf16_u32(va2x4567c45));
          const uint32x4_t va3x4567c45 = vbicq_u32(va3c45, vm4567c45);
          vacc3x4567 = vbfdotq_f32(vacc3x4567, vb4567c45, vreinterpretq_bf16_u32(va3x4567c45));

          if (k > 6 * sizeof(bfloat16_t)) {
            const bfloat16x8_t vb0123c67 = vld1q_bf16(w); w += 8;
            const bfloat16x8_t vb4567c67 = vld1q_bf16(w); w += 8;

            const uint32x4_t va0c67 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va0)), 1);
            const uint32x4_t va1c67 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va1)), 1);
            const uint32x4_t va2c67 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va2)), 1);
            const uint32x4_t va3c67 = vdupq_lane_u32(vreinterpret_u32_bf16(vget_high_bf16(va3)), 1);

            const uint32x4_t vm0123c67 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb0123c67), vmovq_n_u16(0)));
            const uint32x4_t vm4567c67 = vreinterpretq_u32_u16(vceqq_u16(vreinterpretq_u16_bf16(vb4567c67), vmovq_n_u16(0)));

            const uint32x4_t va0x0123c67 = vbicq_u32(va0c67, vm0123c67);
            vacc0x0123 = vbfdotq_f32(vacc0x0123, vb0123c67, vreinterpretq_bf16_u32(va0x0123c67));
            const uint32x4_t va1x0123c67 = vbicq_u32(va1c67, vm0123c67);
            vacc1x0123 = vbfdotq_f32(vacc1x0123, vb0123c67, vreinterpretq_bf16_u32(va1x0123c67));
            const uint32x4_t va2x0123c67 = vbicq_u32(va2c67, vm0123c67);
            vacc2x0123 = vbfdotq_f32(vacc2x0123, vb0123c67, vreinterpretq_bf16_u32(va2x0123c67));
            const uint32x4_t va3x0123c67 = vbicq_u32(va3c67, vm0123c67);
            vacc3x0123 = vbfdotq_f32(vacc3x0123, vb0123c67, vreinterpretq_bf16_u32(va3x0123c67));
            const uint32x4_t va0x4567c67 = vbicq_u32(va0c67, vm4567c67);
            vacc0x4567 = vbfdotq_f32(vacc0x4567, vb4567c67, vreinterpretq_bf16_u32(va0x4567c67));
            const uint32x4_t va1x4567c67 = vbicq_u32(va1c67, vm4567c67);
            vacc1x4567 = vbfdotq_f32(vacc1x4567, vb4567c67, vreinterpretq_bf16_u32(va1x4567c67));
            const uint32x4_t va2x4567c67 = vbicq_u32(va2c67, vm4567c67);
            vacc2x4567 = vbfdotq_f32(vacc2x4567, vb4567c67, vreinterpretq_bf16_u32(va2x4567c67));
            const uint32x4_t va3x4567c67 = vbicq_u32(va3c67, vm4567c67);
            vacc3x4567 = vbfdotq_f32(vacc3x4567, vb4567c67, vreinterpretq_bf16_u32(va3x4567c67));
          }
        }
      }
    }

    const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
    vacc0x0123 = vminq_f32(vacc0x0123, vmax);
    vacc1x0123 = vminq_f32(vacc1x0123, vmax);
    vacc2x0123 = vminq_f32(vacc2x0123, vmax);
    vacc3x0123 = vminq_f32(vacc3x0123, vmax);
    vacc0x4567 = vminq_f32(vacc0x4567, vmax);
    vacc1x4567 = vminq_f32(vacc1x4567, vmax);
    vacc2x4567 = vminq_f32(vacc2x4567, vmax);
    vacc3x4567 = vminq_f32(vacc3x4567, vmax);

    const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
    vacc0x0123 = vmaxq_f32(vacc0x0123, vmin);
    vacc1x0123 = vmaxq_f32(vacc1x0123, vmin);
    vacc2x0123 = vmaxq_f32(vacc2x0123, vmin);
    vacc3x0123 = vmaxq_f32(vacc3x0123, vmin);
    vacc0x4567 = vmaxq_f32(vacc0x4567, vmin);
    vacc1x4567 = vmaxq_f32(vacc1x4567, vmin);
    vacc2x4567 = vmaxq_f32(vacc2x4567, vmin);
    vacc3x4567 = vmaxq_f32(vacc3x4567, vmin);

    bfloat16x4_t vout0x0123 = vcvt_bf16_f32(vacc0x0123);
    bfloat16x4_t vout1x0123 = vcvt_bf16_f32(vacc1x0123);
    bfloat16x4_t vout2x0123 = vcvt_bf16_f32(vacc2x0123);
    bfloat16x4_t vout3x0123 = vcvt_bf16_f32(vacc3x0123);
    bfloat16x4_t vout0x4567 = vcvt_bf16_f32(vacc0x4567);
    bfloat16x4_t vout1x4567 = vcvt_bf16_f32(vacc1x4567);
    bfloat16x4_t vout2x4567 = vcvt_bf16_f32(vacc2x4567);
    bfloat16x4_t vout3x4567 = vcvt_bf16_f32(vacc3x4567);

    if XNN_LIKELY(nc >= 8) {
      vst1_bf16(c0, vout0x0123);
      vst1_bf16(c0 + 4, vout0x4567);
      c0 = (bfloat16_t*) ((uintptr_t) c0 + cn_stride);
      vst1_bf16(c1, vout1x0123);
      vst1_bf16(c1 + 4, vout1x4567);
      c1 = (bfloat16_t*) ((uintptr_t) c1 + cn_stride);
      vst1_bf16(c2, vout2x0123);
      vst1_bf16(c2 + 4, vout2x4567);
      c2 = (bfloat16_t*) ((uintptr_t) c2 + cn_stride);
      vst1_bf16(c3, vout3x0123);
      vst1_bf16(c3 + 4, vout3x4567);
      c3 = (bfloat16_t*) ((uintptr_t) c3 + cn_stride);

      a0 = (const bfloat16_t*) ((uintptr_t) a0 - kc);
      a1 = (const bfloat16_t*) ((uintptr_t) a1 - kc);
      a2 = (const bfloat16_t*) ((uintptr_t) a2 - kc);
      a3 = (const bfloat16_t*) ((uintptr_t) a3 - kc);

      nc -= 8;
    } else {
      if (nc & 4) {
        vst1_bf16(c0, vout0x0123); c0 += 4;
        vst1_bf16(c1, vout1x0123); c1 += 4;
        vst1_bf16(c2, vout2x0123); c2 += 4;
        vst1_bf16(c3, vout3x0123); c3 += 4;

        vout0x0123 = vout0x4567;
        vout1x0123 = vout1x4567;
        vout2x0123 = vout2x4567;
        vout3x0123 = vout3x4567;
      }
      if (nc & 2) {
        vst1_lane_u32((void*) c0, vreinterpret_u32_bf16(vout0x0123), 0); c0 += 2;
        vst1_lane_u32((void*) c1, vreinterpret_u32_bf16(vout1x0123), 0); c1 += 2;
        vst1_lane_u32((void*) c2, vreinterpret_u32_bf16(vout2x0123), 0); c2 += 2;
        vst1_lane_u32((void*) c3, vreinterpret_u32_bf16(vout3x0123), 0); c3 += 2;

        vout0x0123 = vreinterpret_bf16_u16(vext_u16(vreinterpret_u16_bf16(vout0x0123), vreinterpret_u16_bf16(vout0x0123), 2));
        vout1x0123 = vreinterpret_bf16_u16(vext_u16(vreinterpret_u16_bf16(vout1x0123), vreinterpret_u16_bf16(vout1x0123), 2));
        vout2x0123 = vreinterpret_bf16_u16(vext_u16(vreinterpret_u16_bf16(vout2x0123), vreinterpret_u16_bf16(vout2x0123), 2));
        vout3x0123 = vreinterpret_bf16_u16(vext_u16(vreinterpret_u16_bf16(vout3x0123), vreinterpret_u16_bf16(vout3x0123), 2));
      }
      if (nc & 1) {
        vst1_lane_bf16(c0, vout0x0123, 0);
        vst1_lane_bf16(c1, vout1x0123, 0);
        vst1_lane_bf16(c2, vout2x0123, 0);
        vst1_lane_bf16(c3, vout3x0123, 0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
