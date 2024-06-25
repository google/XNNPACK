// Auto-generated file. Do not edit!
//   Template: src/f16-vtanh/neonfp16arith-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

void xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u16(
    size_t n,
    const void* input,
    void* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vsat_cutoff = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4482)));
  const float16x8_t vmagic_bias = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x620F)));
  const float16x8_t vminus_log2e = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBDC5)));
  const float16x8_t vln2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x398C)));
  const float16x8_t vc3 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBD5B)));
  const float16x8_t vc2 = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4008)));
  const float16x8_t vtwo = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0x4000)));
  const float16x8_t vminus_one = vreinterpretq_f16_u16(vmovq_n_u16(UINT16_C(0xBC00)));
  const uint16x8_t vsign_mask = vmovq_n_u16(UINT16_C(0x8000));

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; n >= 2 * sizeof(float16x8_t); n -= 2 * sizeof(float16x8_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vz0 = vabsq_f16(vx0);
    float16x8_t vz1 = vabsq_f16(vx1);

    vz0 = vminq_f16(vz0, vsat_cutoff);
    vz1 = vminq_f16(vz1, vsat_cutoff);

    float16x8_t vn0 = vfmaq_f16(vmagic_bias, vz0, vminus_log2e);
    float16x8_t vn1 = vfmaq_f16(vmagic_bias, vz1, vminus_log2e);

    const float16x8_t vs0 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn0), 10));
    vn0 = vsubq_f16(vn0, vmagic_bias);
    const float16x8_t vs1 = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn1), 10));
    vn1 = vsubq_f16(vn1, vmagic_bias);
    const float16x8_t vt0 = vfmaq_f16(vz0, vn0, vln2);
    const float16x8_t vt1 = vfmaq_f16(vz1, vn1, vln2);
    float16x8_t vp0 = vfmaq_f16(vc2, vc3, vt0);
    float16x8_t vp1 = vfmaq_f16(vc2, vc3, vt1);
    vp0 = vfmsq_f16(vtwo, vp0, vt0);
    vp1 = vfmsq_f16(vtwo, vp1, vt1);

    const float16x8_t vts0 = vmulq_f16(vt0, vs0);
    const float16x8_t vsmo0 = vaddq_f16(vs0, vminus_one);
    const float16x8_t vts1 = vmulq_f16(vt1, vs1);
    const float16x8_t vsmo1 = vaddq_f16(vs1, vminus_one);
    const float16x8_t vemo0 = vfmsq_f16(vsmo0, vp0, vts0);
    const float16x8_t vemo1 = vfmsq_f16(vsmo1, vp1, vts1);
    const float16x8_t vepo0 = vaddq_f16(vemo0, vtwo);
    const float16x8_t vepo1 = vaddq_f16(vemo1, vtwo);

    float16x8_t vrepo0 = vrecpeq_f16(vepo0);
    float16x8_t vrepo1 = vrecpeq_f16(vepo1);
    const float16x8_t verepo0 = vrecpsq_f16(vrepo0, vepo0);
    const float16x8_t verepo1 = vrecpsq_f16(vrepo1, vepo1);
    vrepo0 = vmulq_f16(vrepo0, verepo0);
    vrepo1 = vmulq_f16(vrepo1, verepo1);

    float16x8_t vy0 = vmulq_f16(vemo0, vrepo0);
    float16x8_t vy1 = vmulq_f16(vemo1, vrepo1);


    vy0 = vbslq_f16(vsign_mask, vx0, vy0);
    vy1 = vbslq_f16(vsign_mask, vx1, vy1);

    vst1q_u16(o, vreinterpretq_u16_f16(vy0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy1)); o += 8;
  }
  for (; n >= 1 * sizeof(float16x8_t); n -= 1 * sizeof(float16x8_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float16x8_t vz = vabsq_f16(vx);

    vz = vminq_f16(vz, vsat_cutoff);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);

    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    const float16x8_t vt = vfmaq_f16(vz, vn, vln2);
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vtwo, vp, vt);

    const float16x8_t vts = vmulq_f16(vt, vs);
    const float16x8_t vsmo = vaddq_f16(vs, vminus_one);
    const float16x8_t vemo = vfmsq_f16(vsmo, vp, vts);
    const float16x8_t vepo = vaddq_f16(vemo, vtwo);

    float16x8_t vrepo = vrecpeq_f16(vepo);
    const float16x8_t verepo = vrecpsq_f16(vrepo, vepo);
    vrepo = vmulq_f16(vrepo, verepo);

    float16x8_t vy = vmulq_f16(vemo, vrepo);


    vy = vbslq_f16(vsign_mask, vx, vy);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }

  if (n != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vz = vabsq_f16(vx);
    vz = vminq_f16(vz, vsat_cutoff);

    float16x8_t vn = vfmaq_f16(vmagic_bias, vz, vminus_log2e);
    const float16x8_t vs = vreinterpretq_f16_s16(vshlq_n_s16(vreinterpretq_s16_f16(vn), 10));
    vn = vsubq_f16(vn, vmagic_bias);
    const float16x8_t vt = vfmaq_f16(vz, vn, vln2);
    float16x8_t vp = vfmaq_f16(vc2, vc3, vt);
    vp = vfmsq_f16(vtwo, vp, vt);

    const float16x8_t vts = vmulq_f16(vt, vs);
    const float16x8_t vsmo = vaddq_f16(vs, vminus_one);
    const float16x8_t vemo = vfmsq_f16(vsmo, vp, vts);
    const float16x8_t vepo = vaddq_f16(vemo, vtwo);

    float16x8_t vrepo = vrecpeq_f16(vepo);
    const float16x8_t verepo = vrecpsq_f16(vrepo, vepo);
    vrepo = vmulq_f16(vrepo, verepo);

    float16x8_t vy = vmulq_f16(vemo, vrepo);


    vy = vbslq_f16(vsign_mask, vx, vy);

    float16x4_t vy_lo = vget_low_f16(vy);
    if (n & 4 * sizeof(uint16_t)) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;
      vy_lo = vget_high_f16(vy);
    }
    if (n & 2 * sizeof(uint16_t)) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o+= 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }
    if (n & 1 * sizeof(uint16_t)) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}
