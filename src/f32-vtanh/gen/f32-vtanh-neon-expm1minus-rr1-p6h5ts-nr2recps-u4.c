// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/tanh-neon-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"
#include "xnnpack/microparams.h"



void xnn_f32_vtanh_ukernel__neon_expm1minus_rr1_p6h5ts_nr2recps_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const float32x4_t vsat_cutoff = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.sat_cutoff);
  const float32x4_t vminus_log2e = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.minus_log2e);

  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.magic_bias);

  const float32x4_t vln2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.ln2);

  const float32x4_t vc6 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c6);
  const float32x4_t vc5 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c5);
  const float32x4_t vc4 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c4);
  const float32x4_t vc3 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c3);
  const float32x4_t vc2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_p6h5.c2);

  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vtwo = vmovq_n_f32(2.0f);

  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));


  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vmlaq_f32(vz, vn, vln2);

    float32x4_t vp = vmlaq_f32(vc5, vc6, vt);
    vp = vmlaq_f32(vc4, vp, vt);
    vp = vmlaq_f32(vc3, vp, vt);
    vp = vmlaq_f32(vc2, vp, vt);
    vp = vmlsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vmlsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);
    verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);

    float32x4_t vy = vmulq_f32(vemo, vrepo);

    vy = vbslq_f32(vsign_mask, vx, vy);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vmlaq_f32(vz, vn, vln2);

    float32x4_t vp = vmlaq_f32(vc5, vc6, vt);
    vp = vmlaq_f32(vc4, vp, vt);
    vp = vmlaq_f32(vc3, vp, vt);
    vp = vmlaq_f32(vc2, vp, vt);
    vp = vmlsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vmlsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);
    verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);

    float32x4_t vy = vmulq_f32(vemo, vrepo);

    vy = vbslq_f32(vsign_mask, vx, vy);

    float32x2_t vy_low = vget_low_f32(vy);

    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_low); output += 2;
      vy_low = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_low, 0);
    }
  }
}
