// Auto-generated file. Do not edit!
//   Template: src/f16-vsigmoid/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(__fp16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.magic_bias));
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.minus_log2e));
  const float16x8_t vln2_hi = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.ln2_hi));
  const float16x8_t vln2_lo = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.ln2_lo));
  const float16x8_t vc2 = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.c2));
  const float16x8_t vc1 = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.c1));
  const float16x8_t vone = vmovq_n_f16(1.0f);
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith_rr2_p2.denorm_cutoff));

  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;
  for (; batch >= 8 * sizeof(__fp16); batch -= 8 * sizeof(__fp16)) {
    const float16x8_t vx = vld1q_f16(i); i += 8;

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmaq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vfmsq_f16(vone, vr, vd);
    vr = vfmaq_f16(vr, vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vmovq_n_f16(0.0f));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_f16(o, vf); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vld1q_f16(i);

    const float16x8_t vz = vabsq_f16(vx);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vz, vn, vln2_hi);
    vt = vfmaq_f16(vt, vn, vln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);
    const float16x8_t ve = vfmaq_f16(vs, vp, vt);
    const float16x8_t vd = vaddq_f16(ve, vone);

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vfmsq_f16(vone, vr, vd);
    vr = vfmaq_f16(vr, vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vmovq_n_f16(0.0f));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(__fp16))) {
      vst1_f16(o, vf_lo); o += 4;
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(__fp16))) {
      vst1_f16(o, vf_lo); o += 2;
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(__fp16))) {
      vst1_lane_f16(o, vf_lo, 0);
    }
  }
}
