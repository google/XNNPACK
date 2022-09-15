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


void xnn_f32_velu_ukernel__neon_rr2_p6_x8(
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

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vmaxq_f32(vmulq_f32(vx0123, vprescale), vsat_cutoff);
    const float32x4_t vz4567 = vmaxq_f32(vmulq_f32(vx4567, vprescale), vsat_cutoff);

    float32x4_t vn0123 = vmlaq_f32(vmagic_bias, vz0123, vlog2e);
    float32x4_t vn4567 = vmlaq_f32(vmagic_bias, vz4567, vlog2e);

    float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    float32x4_t vt0123 = vmlaq_f32(vz0123, vn0123, vminus_ln2_hi);
    float32x4_t vt4567 = vmlaq_f32(vz4567, vn4567, vminus_ln2_hi);

    vt0123 = vmlaq_f32(vt0123, vn0123, vminus_ln2_lo);
    vt4567 = vmlaq_f32(vt4567, vn4567, vminus_ln2_lo);

    float32x4_t vp0123 = vmlaq_f32(vc5, vc6, vt0123);
    float32x4_t vp4567 = vmlaq_f32(vc5, vc6, vt4567);

    vp0123 = vmlaq_f32(vc4, vp0123, vt0123);
    vp4567 = vmlaq_f32(vc4, vp4567, vt4567);

    vp0123 = vmlaq_f32(vc3, vp0123, vt0123);
    vp4567 = vmlaq_f32(vc3, vp4567, vt4567);

    vp0123 = vmlaq_f32(vc2, vp0123, vt0123);
    vp4567 = vmlaq_f32(vc2, vp4567, vt4567);

    vp0123 = vmulq_f32(vp0123, vt0123);
    vp4567 = vmulq_f32(vp4567, vt4567);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vs0123 = vsubq_f32(vs0123, vone);
    vt4567 = vmulq_f32(vt4567, vs4567);
    vs4567 = vsubq_f32(vs4567, vone);

    vp0123 = vmlaq_f32(vt0123, vp0123, vt0123);
    vp4567 = vmlaq_f32(vt4567, vp4567, vt4567);

    const float32x4_t ve0123 = vmulq_f32(vaddq_f32(vp0123, vs0123), valpha);
    const float32x4_t ve4567 = vmulq_f32(vaddq_f32(vp4567, vs4567), valpha);

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    vx0123 = vmulq_f32(vx0123, vbeta);
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));
    vx4567 = vmulq_f32(vx4567, vbeta);

    const float32x4_t vy0123 = vbslq_f32(vm0123, ve0123, vx0123);
    const float32x4_t vy4567 = vbslq_f32(vm4567, ve4567, vx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
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
