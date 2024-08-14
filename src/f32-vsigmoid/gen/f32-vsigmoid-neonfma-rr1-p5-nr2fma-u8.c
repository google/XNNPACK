// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/neon-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p0f);
  const float32x4_t vc5 = vmovq_n_f32(-0x1.0F9F9Cp-7f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc3 = vmovq_n_f32(-0x1.555A80p-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc1 = vmovq_n_f32(-0x1.FFFFF6p-1f);
  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float32x4_t vln2 = vmovq_n_f32(0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vln2);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vabsq_f32(vx0123);
    const float32x4_t vz4567 = vabsq_f32(vx4567);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2);

    float32x4_t vp0123 = vfmaq_f32(vc4, vc5, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc4, vc5, vt4567);

    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);

    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);

    vp0123 = vfmaq_f32(vc1, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc1, vp4567, vt4567);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vt4567 = vmulq_f32(vt4567, vs4567);

    const float32x4_t ve0123 = vfmaq_f32(vs0123, vp0123, vt0123);
    const float32x4_t ve4567 = vfmaq_f32(vs4567, vp4567, vt4567);

    const float32x4_t vd0123 = vaddq_f32(ve0123, vone);
    const float32x4_t vd4567 = vaddq_f32(ve4567, vone);

    float32x4_t vr0123 = vrecpeq_f32(vd0123);
    float32x4_t vr4567 = vrecpeq_f32(vd4567);

    vr0123 = vfmaq_f32(vr0123, vr0123, vfmsq_f32(vone, vr0123, vd0123));
    vr4567 = vfmaq_f32(vr4567, vr4567, vfmsq_f32(vone, vr4567, vd4567));

    vr0123 = vfmaq_f32(vr0123, vr0123, vfmsq_f32(vone, vr0123, vd0123));
    vr4567 = vfmaq_f32(vr4567, vr4567, vfmsq_f32(vone, vr4567, vd4567));

    float32x4_t vf0123 = vmulq_f32(ve0123, vr0123);
    float32x4_t vf4567 = vmulq_f32(ve4567, vr4567);

    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcagtq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcagtq_f32(vx4567, vdenorm_cutoff)));

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));

    vf0123 = vbslq_f32(vm0123, vf0123, vsubq_f32(vone, vf0123));
    vf4567 = vbslq_f32(vm4567, vf4567, vsubq_f32(vone, vf4567));

    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    vt = vmulq_f32(vt, vs);
    const float32x4_t ve = vfmaq_f32(vs, vp, vt);
    const float32x4_t vd = vaddq_f32(ve, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));

    float32x4_t vf = vmulq_f32(ve, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(output, vf); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    vt = vmulq_f32(vt, vs);
    const float32x4_t ve = vfmaq_f32(vs, vp, vt);
    const float32x4_t vd = vaddq_f32(ve, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));

    float32x4_t vf = vmulq_f32(ve, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vf_lo); output += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vf_lo, 0);
    }
  }
}
