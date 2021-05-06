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

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4(
    size_t n,
    const float* x,
    float* y,
    const void* params) XNN_DISABLE_TSAN
{
  assert(n % sizeof(float) == 0);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p+0f);
  const float32x4_t vln2 = vmovq_n_f32(0x1.62E43p-1f);
  const float32x4_t vc5 = vmovq_n_f32(-0x1.0F9F9Cp-7f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc3 = vmovq_n_f32(-0x1.555A80p-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc1 = vmovq_n_f32(-0x1.FFFFF6p-1f);
  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep+6f);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(x); x += 4;

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
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));

    float32x4_t vf = vmulq_f32(ve, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(y, vf); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const float32x4_t vx = vld1q_f32(x);

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
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vfmaq_f32(vr, vr, vfmsq_f32(vone, vr, vd));

    float32x4_t vf = vmulq_f32(ve, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vf_lo); y += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vf_lo, 0);
    }
  }
}
