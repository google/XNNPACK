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

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x660F)));  // 0x1.83Cp+10h
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));  // -0x1.714p+0h
  const float16x8_t vln2_hi = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));  // 0x1.630p-1h
  const float16x8_t vln2_lo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x8AF4)));  // -0x1.BD0p-13h
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x37F9)));  // 0x1.FE4p-2h
  const float16x8_t vc1 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC0E)));  // -0x1.038p+0h
  const float16x8_t vone = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x3C00)));  // 1.0h
  const float16x8_t vdenorm_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xC8DA)));  // -0x1.368p+3h

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vc1);
  XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx2 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx3 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vz0 = vabsq_f16(vx0);
    const float16x8_t vz1 = vabsq_f16(vx1);
    const float16x8_t vz2 = vabsq_f16(vx2);
    const float16x8_t vz3 = vabsq_f16(vx3);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vminus_log2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vminus_log2e);
    float16x8_t vn2 = vfmaq_f16(vmagic_bias, vz2, vminus_log2e);
    float16x8_t vn3 = vfmaq_f16(vmagic_bias, vz3, vminus_log2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    const float16x8_t vs2 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn2), 10));
    const float16x8_t vs3 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn3), 10));

    vn0 = vsubq_f16(vn0, vmagic_bias);
    vn1 = vsubq_f16(vn1, vmagic_bias);
    vn2 = vsubq_f16(vn2, vmagic_bias);
    vn3 = vsubq_f16(vn3, vmagic_bias);

    float16x8_t vt0 = vfmaq_f16(vz0, vn0, vln2_hi);
    float16x8_t vt1 = vfmaq_f16(vz1, vn1, vln2_hi);
    float16x8_t vt2 = vfmaq_f16(vz2, vn2, vln2_hi);
    float16x8_t vt3 = vfmaq_f16(vz3, vn3, vln2_hi);

    vt0 = vfmaq_f16(vt0, vn0, vln2_lo);
    vt1 = vfmaq_f16(vt1, vn1, vln2_lo);
    vt2 = vfmaq_f16(vt2, vn2, vln2_lo);
    vt3 = vfmaq_f16(vt3, vn3, vln2_lo);

    const float16x8_t vp0 = vfmaq_f16(vc1, vc2, vt0);
    const float16x8_t vp1 = vfmaq_f16(vc1, vc2, vt1);
    const float16x8_t vp2 = vfmaq_f16(vc1, vc2, vt2);
    const float16x8_t vp3 = vfmaq_f16(vc1, vc2, vt3);

    vt0 = vmulq_f16(vt0, vs0);
    vt1 = vmulq_f16(vt1, vs1);
    vt2 = vmulq_f16(vt2, vs2);
    vt3 = vmulq_f16(vt3, vs3);

    const float16x8_t ve0 = vfmaq_f16(vs0, vp0, vt0);
    const float16x8_t ve1 = vfmaq_f16(vs1, vp1, vt1);
    const float16x8_t ve2 = vfmaq_f16(vs2, vp2, vt2);
    const float16x8_t ve3 = vfmaq_f16(vs3, vp3, vt3);

    const float16x8_t vd0 = vaddq_f16(ve0, vone);
    const float16x8_t vd1 = vaddq_f16(ve1, vone);
    const float16x8_t vd2 = vaddq_f16(ve2, vone);
    const float16x8_t vd3 = vaddq_f16(ve3, vone);

    float16x8_t vf0 = vdivq_f16(ve0, vd0);
    float16x8_t vf1 = vdivq_f16(ve1, vd1);
    float16x8_t vf2 = vdivq_f16(ve2, vd2);
    float16x8_t vf3 = vdivq_f16(ve3, vd3);

    vf0 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf0), vcagtq_f16(vx0, vdenorm_cutoff)));
    vf1 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf1), vcagtq_f16(vx1, vdenorm_cutoff)));
    vf2 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf2), vcagtq_f16(vx2, vdenorm_cutoff)));
    vf3 = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf3), vcagtq_f16(vx3, vdenorm_cutoff)));

    const uint16x8_t vm0 = vcltq_f16(vx0, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm1 = vcltq_f16(vx1, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm2 = vcltq_f16(vx2, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    const uint16x8_t vm3 = vcltq_f16(vx3, vreinterpretq_f16_u16(vmovq_n_u16(0)));

    vf0 = vbslq_f16(vm0, vf0, vsubq_f16(vone, vf0));
    vf1 = vbslq_f16(vm1, vf1, vsubq_f16(vone, vf1));
    vf2 = vbslq_f16(vm2, vf2, vsubq_f16(vone, vf2));
    vf3 = vbslq_f16(vm3, vf3, vsubq_f16(vone, vf3));

    vst1q_u16(o, vreinterpretq_u16_f16(vf0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf1)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf2)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vf3)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

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

    float16x8_t vf = vdivq_f16(ve, vd);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    vst1q_u16(o, vreinterpretq_u16_f16(vf)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

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

    float16x8_t vf = vdivq_f16(ve, vd);
    vf = vreinterpretq_f16_u16(vbicq_u16(vreinterpretq_u16_f16(vf), vcagtq_f16(vx, vdenorm_cutoff)));
    const uint16x8_t vm = vcltq_f16(vx, vreinterpretq_f16_u16(vmovq_n_u16(0)));
    vf = vbslq_f16(vm, vf, vsubq_f16(vone, vf));

    float16x4_t vf_lo = vget_low_f16(vf);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vf_lo)); o += 4;
      vf_lo = vget_high_f16(vf);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vf_lo), 0); o += 2;
      vf_lo = vext_f16(vf_lo, vf_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vf_lo), 0);
    }
  }
}
