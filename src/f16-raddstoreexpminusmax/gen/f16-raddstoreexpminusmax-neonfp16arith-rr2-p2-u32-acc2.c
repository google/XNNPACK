// Auto-generated file. Do not edit!
//   Template: src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/raddstoreexpminusmax.h"


void xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2(
    size_t batch,
    const void* input,
    const void* max,
    void* output,
    void* sum,
    const union xnn_f16_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  const float16x8_t vlog2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3DC5)));  // 0x1.714p+0h
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_ln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xB98C)));  // -0x1.630p-1h
  const float16x8_t vminus_ln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x0AF4)));  // 0x1.BD0p-13h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C0E)));  // 0x1.038p+0h
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float16x8_t vi_max = vreinterpretq_f16_u16(vld1q_dup_u16(max));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  float16x8_t vacc0 = vreinterpretq_f16_u16(vmovq_n_u16(0));
  float16x8_t vacc1 = vreinterpretq_f16_u16(vmovq_n_u16(0));
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vi0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vi3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx0 = vsubq_f16(vi0, vi_max);
    const float16x8_t vx1 = vsubq_f16(vi1, vi_max);
    const float16x8_t vx2 = vsubq_f16(vi2, vi_max);
    const float16x8_t vx3 = vsubq_f16(vi3, vi_max);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vx0, vlog2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vx1, vlog2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vx2, vlog2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vx3, vlog2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);
    vn2 = vsubq_f16(vn2, vmagic_bias);
    vn3 = vsubq_f16(vn3, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vx0, vn0, vminus_ln2_hi);
    float16x8_t vt1 = vfmaq_f16(vx1, vn1, vminus_ln2_hi);
    float16x8_t vt2 = vfmaq_f16(vx2, vn2, vminus_ln2_hi);
    float16x8_t vt3 = vfmaq_f16(vx3, vn3, vminus_ln2_hi);

    vt0 = vfmaq_f16(vt0, vn0, vminus_ln2_lo);
    vt1 = vfmaq_f16(vt1, vn1, vminus_ln2_lo);
    vt2 = vfmaq_f16(vt2, vn2, vminus_ln2_lo);
    vt3 = vfmaq_f16(vt3, vn3, vminus_ln2_lo);

    const float16x8_t vp0 = vfmaq_f16(vc1, vc2, vt0);
    const float16x8_t vp1 = vfmaq_f16(vc1, vc2, vt1);
    const float16x8_t vp2 = vfmaq_f16(vc1, vc2, vt2);
    const float16x8_t vp3 = vfmaq_f16(vc1, vc2, vt3);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);
    vt2 = vmulq_f16(vt2, vs2);
    vt3 = vmulq_f16(vt3, vs3);

    float16x8_t vf0 = vfmaq_f16(vs0, vp0, vt0);
    const uint16x8_t vm0 = vcltq_f16(vx0, vdenorm_cutoff);
    float16x8_t vf1 = vfmaq_f16(vs1, vp1, vt1);
    const uint16x8_t vm1 = vcltq_f16(vx1, vdenorm_cutoff);
    float16x8_t vf2 = vfmaq_f16(vs2, vp2, vt2);
    const uint16x8_t vm2 = vcltq_f16(vx2, vdenorm_cutoff);
    float16x8_t vf3 = vfmaq_f16(vs3, vp3, vt3);
    const uint16x8_t vm3 = vcltq_f16(vx3, vdenorm_cutoff);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vm0));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vm1));
    vf2 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf2), vm2));
    vf3 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf3), vm3));

    vst1q_u16(o, vreinterpretq_u16_f16(vf0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf3)); o += 8;

    vacc0 = vaddq_f16(vacc0, vf0);
    vacc1 = vaddq_f16(vacc1, vf1);
    vacc0 = vaddq_f16(vacc0, vf2);
    vacc1 = vaddq_f16(vacc1, vf3);
  }
  vacc0 = vaddq_f16(vacc0, vacc1);

  float16x8_t vacc = vacc0;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vx = vsubq_f16(vi, vi_max);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);

    float16x8_t vf = vfmaq_f16(vs, vp, vt);
    const uint16x8_t vm = vcltq_f16(vx, vdenorm_cutoff);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vm));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;

    vacc = vaddq_f16(vacc, vf);
  }
  float16x4_t vacc_lo = vadd_f16(vget_low_f16(vacc), vget_high_f16(vacc));
  if (batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const float16x8_t vi = vreinterpretq_f16_u16(vld1q_u16(i));

    const float16x8_t vx = vsubq_f16(vi, vi_max);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vx, vlog2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);

    float16x8_t vt = vfmaq_f16(vx, vn, vminus_ln2_hi);
    vt = vfmaq_f16(vt, vn, vminus_ln2_lo);

    const float16x8_t vp = vfmaq_f16(vc1, vc2, vt);
    vt = vmulq_f16(vt, vs);

    float16x8_t vf = vfmaq_f16(vs, vp, vt);
    const uint16x8_t vm = vcltq_f16(vx, vdenorm_cutoff);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vm));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vf_lo)); o += 4;
      vacc_lo = vadd_f16(vacc_lo, vf_lo);
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vf_lo), 0); o += 2;
      vacc_lo = vadd_f16(vacc_lo, vreinterpret_f16_u64(vshl_n_u64(vreinterpret_u64_f16(vf_lo), 32)));
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vf_lo), 0);
      vacc_lo = vadd_f16(vacc_lo, vreinterpret_f16_u64(vshl_n_u64(vreinterpret_u64_f16(vf_lo), 48)));
    }
  }
  vacc_lo = vpadd_f16(vacc_lo, vacc_lo);
  vacc_lo = vpadd_f16(vacc_lo, vacc_lo);
  vst1_lane_u16(sum, vreinterpret_u16_f16(vacc_lo), 0);
}
