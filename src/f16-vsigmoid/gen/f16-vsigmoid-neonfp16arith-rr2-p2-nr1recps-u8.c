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


void xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_u8(
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

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vrecpsq_f16(vr, vd);
    vr = vmulq_f16(vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
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

    float16x8_t vr = vrecpeq_f16(vd);
    const float16x8_t vadj = vrecpsq_f16(vr, vd);
    vr = vmulq_f16(vr, vadj);

    float16x8_t vf = vmulq_f16(ve, vr);
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
