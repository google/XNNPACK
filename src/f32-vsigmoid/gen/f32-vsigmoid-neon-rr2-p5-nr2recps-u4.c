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


void xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_u4(
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

  const float32x4_t vln2_hi = vmovq_n_f32(0x1.62E400p-1f);
  const float32x4_t vln2_lo = vmovq_n_f32(0x1.7F7D1Cp-20f);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vmlaq_f32(vz, vn, vln2_hi);
    vt = vmlaq_f32(vt, vn, vln2_lo);

    float32x4_t vp = vmlaq_f32(vc4, vc5, vt);
    vp = vmlaq_f32(vc3, vp, vt);
    vp = vmlaq_f32(vc2, vp, vt);
    vp = vmlaq_f32(vc1, vp, vt);

    vt = vmulq_f32(vt, vs);
    const float32x4_t ve = vmlaq_f32(vs, vp, vt);
    const float32x4_t vd = vaddq_f32(ve, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    float32x4_t vf = vmulq_f32(ve, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(output, vf); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);
    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vmlaq_f32(vz, vn, vln2_hi);
    vt = vmlaq_f32(vt, vn, vln2_lo);

    float32x4_t vp = vmlaq_f32(vc4, vc5, vt);
    vp = vmlaq_f32(vc3, vp, vt);
    vp = vmlaq_f32(vc2, vp, vt);
    vp = vmlaq_f32(vc1, vp, vt);

    vt = vmulq_f32(vt, vs);
    const float32x4_t ve = vmlaq_f32(vs, vp, vt);
    const float32x4_t vd = vaddq_f32(ve, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

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
