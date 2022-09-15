// Auto-generated file. Do not edit!
//   Template: src/f32-velu/neon-p6.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>


void xnn_f32_velu_ukernel__neon_rr2_p6_x4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vprescale = vld1q_dup_f32(&params->neon_rr2_p6.prescale);
  const float32x4_t valpha = vld1q_dup_f32(&params->neon_rr2_p6.alpha);
  const float32x4_t vbeta = vld1q_dup_f32(&params->neon_rr2_p6.beta);
  const float32x4_t vsat_cutoff = vld1q_dup_f32(&params->neon_rr2_p6.sat_cutoff);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_rr2_p6.magic_bias);
  const float32x4_t vlog2e = vld1q_dup_f32(&params->neon_rr2_p6.log2e);
  const float32x4_t vminus_ln2_hi = vld1q_dup_f32(&params->neon_rr2_p6.minus_ln2_hi);
  const float32x4_t vminus_ln2_lo = vld1q_dup_f32(&params->neon_rr2_p6.minus_ln2_lo);
  const float32x4_t vc6 = vld1q_dup_f32(&params->neon_rr2_p6.c6);
  const float32x4_t vc5 = vld1q_dup_f32(&params->neon_rr2_p6.c5);
  const float32x4_t vc4 = vld1q_dup_f32(&params->neon_rr2_p6.c4);
  const float32x4_t vc3 = vld1q_dup_f32(&params->neon_rr2_p6.c3);
  const float32x4_t vc2 = vld1q_dup_f32(&params->neon_rr2_p6.c2);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vlog2e);
    float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vmlaq_f32(vz, vn, vminus_ln2_hi);
    vt = vmlaq_f32(vt, vn, vminus_ln2_lo);

    float32x4_t vp = vmlaq_f32(vc5, vc6, vt);
    vp = vmlaq_f32(vc4, vp, vt);
    vp = vmlaq_f32(vc3, vp, vt);
    vp = vmlaq_f32(vc2, vp, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vmlaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vlog2e);
    float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vmlaq_f32(vz, vn, vminus_ln2_hi);
    vt = vmlaq_f32(vt, vn, vminus_ln2_lo);

    float32x4_t vp = vmlaq_f32(vc5, vc6, vt);
    vp = vmlaq_f32(vc4, vp, vt);
    vp = vmlaq_f32(vc3, vp, vt);
    vp = vmlaq_f32(vc2, vp, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vmlaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
