// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/neon-p5.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.8000FEp23f);
  const float32x4_t vc5 = vmovq_n_f32(0x1.0F9F9Cp-7f);
  const float32x4_t vc4 = vmovq_n_f32(0x1.573A1Ap-5f);
  const float32x4_t vc3 = vmovq_n_f32(0x1.555A80p-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFDC6p-2f);
  const float32x4_t vc1 = vmovq_n_f32(0x1.FFFFF6p-1f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vc5);
  XNN_FORCE_REALIZATION(vc4);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);  
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vminus_ln2);

  const float32x4_t vi_max = vld1q_dup_f32(max);

  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  float32x4_t vacc2 = vmovq_n_f32(0.0f);
  float32x4_t vacc3 = vmovq_n_f32(0.0f);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vi0123 = vld1q_f32(input); input += 4;
    const float32x4_t vi4567 = vld1q_f32(input); input += 4;
    const float32x4_t vi89AB = vld1q_f32(input); input += 4;
    const float32x4_t viCDEF = vld1q_f32(input); input += 4;

    const float32x4_t vx0123 = vsubq_f32(vi0123, vi_max);
    const float32x4_t vx4567 = vsubq_f32(vi4567, vi_max);
    const float32x4_t vx89AB = vsubq_f32(vi89AB, vi_max);
    const float32x4_t vxCDEF = vsubq_f32(viCDEF, vi_max);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vx0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vx4567, vlog2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vx89AB, vlog2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vxCDEF, vlog2e);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 23));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 23));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 23));
    const float32x4_t vsCDEF = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 23));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vx0123, vn0123, vminus_ln2);
    float32x4_t vt4567 = vfmaq_f32(vx4567, vn4567, vminus_ln2);
    float32x4_t vt89AB = vfmaq_f32(vx89AB, vn89AB, vminus_ln2);
    float32x4_t vtCDEF = vfmaq_f32(vxCDEF, vnCDEF, vminus_ln2);

    float32x4_t vp0123 = vfmaq_f32(vc4, vc5, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc4, vc5, vt4567);
    float32x4_t vp89AB = vfmaq_f32(vc4, vc5, vt89AB);
    float32x4_t vpCDEF = vfmaq_f32(vc4, vc5, vtCDEF);

    vp0123 = vfmaq_f32(vc3, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc3, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc3, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc3, vpCDEF, vtCDEF);

    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc2, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc2, vpCDEF, vtCDEF);

    vp0123 = vfmaq_f32(vc1, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc1, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc1, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc1, vpCDEF, vtCDEF);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vt4567 = vmulq_f32(vt4567, vs4567);
    vt89AB = vmulq_f32(vt89AB, vs89AB);
    vtCDEF = vmulq_f32(vtCDEF, vsCDEF);

    float32x4_t vf0123 = vfmaq_f32(vs0123, vp0123, vt0123);
    float32x4_t vf4567 = vfmaq_f32(vs4567, vp4567, vt4567);
    float32x4_t vf89AB = vfmaq_f32(vs89AB, vp89AB, vt89AB);
    float32x4_t vfCDEF = vfmaq_f32(vsCDEF, vpCDEF, vtCDEF);

    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vx4567, vdenorm_cutoff)));
    vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcltq_f32(vx89AB, vdenorm_cutoff)));
    vfCDEF = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vfCDEF), vcltq_f32(vxCDEF, vdenorm_cutoff)));

    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;
    vst1q_f32(output, vf89AB); output += 4;
    vst1q_f32(output, vfCDEF); output += 4;

    vacc0 = vaddq_f32(vacc0, vf0123);
    vacc0 = vaddq_f32(vacc0, vf4567);
    vacc0 = vaddq_f32(vacc0, vf89AB);
    vacc0 = vaddq_f32(vacc0, vfCDEF);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);
  vacc2 = vaddq_f32(vacc2, vacc3);
  vacc0 = vaddq_f32(vacc0, vacc2);

  float32x4_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vi = vld1q_f32(input); input += 4;

    const float32x4_t vx = vsubq_f32(vi, vi_max);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    vt = vmulq_f32(vt, vs);
    float32x4_t vf = vfmaq_f32(vs, vp, vt);

    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    vst1q_f32(output, vf); output += 4;

    vacc = vaddq_f32(vacc, vf);
  }
#if XNN_ARCH_ARM64
  float vacc_lo = vaddvq_f32(vacc);
#else
  float32x2_t vacc_lo = vadd_f32(vget_high_f32(vacc), vget_low_f32(vacc));
#endif
  if (batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vi = vld1q_f32(input); input += 4;

    const float32x4_t vx = vsubq_f32(vi, vi_max);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    const float32x4_t vs = vreinterpretq_f32_s32(vshlq_n_s32(vreinterpretq_s32_f32(vn), 23));

    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    float32x4_t vp = vfmaq_f32(vc4, vc5, vt);
    vp = vfmaq_f32(vc3, vp, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmaq_f32(vc1, vp, vt);

    vt = vmulq_f32(vt, vs);
    float32x4_t vf = vfmaq_f32(vs, vp, vt);

    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcltq_f32(vx, vdenorm_cutoff)));

    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vf_lo); output += 2;

      #if XNN_ARCH_ARM64
        vacc_lo += vaddv_f32(vf_lo);
      #else
        vacc_lo = vadd_f32(vacc_lo, vf_lo);
      #endif

      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vf_lo, 0);

      #if XNN_ARCH_ARM64
        vacc_lo += vget_lane_f32(vf_lo, 0);
      #else
        vacc_lo = vadd_f32(vacc_lo, vreinterpret_f32_u64(vshl_n_u64(vreinterpret_u64_f32(vf_lo), 32)));
      #endif
    }
  }
#if XNN_ARCH_ARM64
  *sum = vacc_lo;
#else
  vst1_lane_f32(sum, vpadd_f32(vacc_lo, vacc_lo), 0);
#endif
}
