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

void xnn_f16_vtanh_ukernel__neonfp16arith_expm1minus_rr1_p3h2ts_nr1recps_u8(
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
