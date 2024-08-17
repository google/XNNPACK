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


void xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_u24(
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

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;
    const float32x4_t vxGHIJ = vld1q_f32(input); input += 4;
    const float32x4_t vxKLMN = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vabsq_f32(vx0123);
    const float32x4_t vz4567 = vabsq_f32(vx4567);
    const float32x4_t vz89AB = vabsq_f32(vx89AB);
    const float32x4_t vzCDEF = vabsq_f32(vxCDEF);
    const float32x4_t vzGHIJ = vabsq_f32(vxGHIJ);
    const float32x4_t vzKLMN = vabsq_f32(vxKLMN);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vz89AB, vminus_log2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vzCDEF, vminus_log2e);
    float32x4_t vnGHIJ = vfmaq_f32(vmagic_bias, vzGHIJ, vminus_log2e);
    float32x4_t vnKLMN = vfmaq_f32(vmagic_bias, vzKLMN, vminus_log2e);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 23));
    const float32x4_t vsCDEF = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 23));
    const float32x4_t vsGHIJ = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vnGHIJ), 23));
    const float32x4_t vsKLMN = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vnKLMN), 23));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);
    vnGHIJ = vsubq_f32(vnGHIJ, vmagic_bias);
    vnKLMN = vsubq_f32(vnKLMN, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2);
    float32x4_t vt89AB = vfmaq_f32(vz89AB, vn89AB, vln2);
    float32x4_t vtCDEF = vfmaq_f32(vzCDEF, vnCDEF, vln2);
    float32x4_t vtGHIJ = vfmaq_f32(vzGHIJ, vnGHIJ, vln2);
    float32x4_t vtKLMN = vfmaq_f32(vzKLMN, vnKLMN, vln2);

    float32x4_t vp0123 = vfmaq_f32(vc4, vc5, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc4, vc5, vt4567);
    float32x4_t vp89AB = vfmaq_f32(vc4, vc5, vt89AB);
    float32x4_t vpCDEF = vfmaq_f32(vc4, vc5, vtCDEF);
    float32x4_t vpGHIJ = vfmaq_f32(vc4, vc5, vtGHIJ);
    float32x4_t vpKLMN = vfmaq_f32(vc4, vc5, vtKLMN);

    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc3, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc3, vpCDEF, vtCDEF);
    vpGHIJ = vfmaq_f32(vc3, vpGHIJ, vtGHIJ);
    vpKLMN = vfmaq_f32(vc3, vpKLMN, vtKLMN);

    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc2, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc2, vpCDEF, vtCDEF);
    vpGHIJ = vfmaq_f32(vc2, vpGHIJ, vtGHIJ);
    vpKLMN = vfmaq_f32(vc2, vpKLMN, vtKLMN);

    vp0123 = vfmaq_f32(vc1, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc1, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc1, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc1, vpCDEF, vtCDEF);
    vpGHIJ = vfmaq_f32(vc1, vpGHIJ, vtGHIJ);
    vpKLMN = vfmaq_f32(vc1, vpKLMN, vtKLMN);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vt4567 = vmulq_f32(vt4567, vs4567);
    vt89AB = vmulq_f32(vt89AB, vs89AB);
    vtCDEF = vmulq_f32(vtCDEF, vsCDEF);
    vtGHIJ = vmulq_f32(vtGHIJ, vsGHIJ);
    vtKLMN = vmulq_f32(vtKLMN, vsKLMN);

    const float32x4_t ve0123 = vfmaq_f32(vs0123, vp0123, vt0123);
    const float32x4_t ve4567 = vfmaq_f32(vs4567, vp4567, vt4567);
    const float32x4_t ve89AB = vfmaq_f32(vs89AB, vp89AB, vt89AB);
    const float32x4_t veCDEF = vfmaq_f32(vsCDEF, vpCDEF, vtCDEF);
    const float32x4_t veGHIJ = vfmaq_f32(vsGHIJ, vpGHIJ, vtGHIJ);
    const float32x4_t veKLMN = vfmaq_f32(vsKLMN, vpKLMN, vtKLMN);

    const float32x4_t vd0123 = vaddq_f32(ve0123, vone);
    const float32x4_t vd4567 = vaddq_f32(ve4567, vone);
    const float32x4_t vd89AB = vaddq_f32(ve89AB, vone);
    const float32x4_t vdCDEF = vaddq_f32(veCDEF, vone);
    const float32x4_t vdGHIJ = vaddq_f32(veGHIJ, vone);
    const float32x4_t vdKLMN = vaddq_f32(veKLMN, vone);

    float32x4_t vr0123 = vrecpeq_f32(vd0123);
    float32x4_t vr4567 = vrecpeq_f32(vd4567);
    float32x4_t vr89AB = vrecpeq_f32(vd89AB);
    float32x4_t vrCDEF = vrecpeq_f32(vdCDEF);
    float32x4_t vrGHIJ = vrecpeq_f32(vdGHIJ);
    float32x4_t vrKLMN = vrecpeq_f32(vdKLMN);

    vr0123 = vmulq_f32(vr0123, vrecpsq_f32(vr0123, vd0123));
    vr4567 = vmulq_f32(vr4567, vrecpsq_f32(vr4567, vd4567));
    vr89AB = vmulq_f32(vr89AB, vrecpsq_f32(vr89AB, vd89AB));
    vrCDEF = vmulq_f32(vrCDEF, vrecpsq_f32(vrCDEF, vdCDEF));
    vrGHIJ = vmulq_f32(vrGHIJ, vrecpsq_f32(vrGHIJ, vdGHIJ));
    vrKLMN = vmulq_f32(vrKLMN, vrecpsq_f32(vrKLMN, vdKLMN));

    vr0123 = vmulq_f32(vr0123, vrecpsq_f32(vr0123, vd0123));
    vr4567 = vmulq_f32(vr4567, vrecpsq_f32(vr4567, vd4567));
    vr89AB = vmulq_f32(vr89AB, vrecpsq_f32(vr89AB, vd89AB));
    vrCDEF = vmulq_f32(vrCDEF, vrecpsq_f32(vrCDEF, vdCDEF));
    vrGHIJ = vmulq_f32(vrGHIJ, vrecpsq_f32(vrGHIJ, vdGHIJ));
    vrKLMN = vmulq_f32(vrKLMN, vrecpsq_f32(vrKLMN, vdKLMN));

    float32x4_t vf0123 = vmulq_f32(ve0123, vr0123);
    float32x4_t vf4567 = vmulq_f32(ve4567, vr4567);
    float32x4_t vf89AB = vmulq_f32(ve89AB, vr89AB);
    float32x4_t vfCDEF = vmulq_f32(veCDEF, vrCDEF);
    float32x4_t vfGHIJ = vmulq_f32(veGHIJ, vrGHIJ);
    float32x4_t vfKLMN = vmulq_f32(veKLMN, vrKLMN);

    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcagtq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcagtq_f32(vx4567, vdenorm_cutoff)));
    vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcagtq_f32(vx89AB, vdenorm_cutoff)));
    vfCDEF = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfCDEF), vcagtq_f32(vxCDEF, vdenorm_cutoff)));
    vfGHIJ = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfGHIJ), vcagtq_f32(vxGHIJ, vdenorm_cutoff)));
    vfKLMN = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfKLMN), vcagtq_f32(vxKLMN, vdenorm_cutoff)));

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));
    const uint32x4_t vm89AB = vcltq_f32(vx89AB, vmovq_n_f32(0.0f));
    const uint32x4_t vmCDEF = vcltq_f32(vxCDEF, vmovq_n_f32(0.0f));
    const uint32x4_t vmGHIJ = vcltq_f32(vxGHIJ, vmovq_n_f32(0.0f));
    const uint32x4_t vmKLMN = vcltq_f32(vxKLMN, vmovq_n_f32(0.0f));

    vf0123 = vbslq_f32(vm0123, vf0123, vsubq_f32(vone, vf0123));
    vf4567 = vbslq_f32(vm4567, vf4567, vsubq_f32(vone, vf4567));
    vf89AB = vbslq_f32(vm89AB, vf89AB, vsubq_f32(vone, vf89AB));
    vfCDEF = vbslq_f32(vmCDEF, vfCDEF, vsubq_f32(vone, vfCDEF));
    vfGHIJ = vbslq_f32(vmGHIJ, vfGHIJ, vsubq_f32(vone, vfGHIJ));
    vfKLMN = vbslq_f32(vmKLMN, vfKLMN, vsubq_f32(vone, vfKLMN));

    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;
    vst1q_f32(output, vf89AB); output += 4;
    vst1q_f32(output, vfCDEF); output += 4;
    vst1q_f32(output, vfGHIJ); output += 4;
    vst1q_f32(output, vfKLMN); output += 4;
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
