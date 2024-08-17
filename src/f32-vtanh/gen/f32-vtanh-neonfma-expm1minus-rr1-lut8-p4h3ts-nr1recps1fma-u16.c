// Auto-generated file. Do not edit!
//   Template: src/f32-vtanh/tanh-neon-expm1minus.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"
#include "xnnpack/microparams.h"

extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_8[8];


void xnn_f32_vtanh_ukernel__neonfma_expm1minus_rr1_lut8_p4h3ts_nr1recps1fma_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  const float32x4_t vsat_cutoff = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.sat_cutoff);
  const float32x4_t vminus_log2e = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.minus_log2e);

  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.magic_bias);

  const uint64x2_t vindex_mask = vreinterpretq_u64_u32(vmovq_n_u32(UINT32_C(0x7)));
  const float32x4_t vln2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.ln2);

  const float32x4_t vc4 = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.c4);
  const float32x4_t vc3 = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.c3);
  const float32x4_t vc2 = vld1q_dup_f32(&params->neon_expm1minus_rr1_lut8_p4h3.c2);

  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vtwo = vmovq_n_f32(2.0f);

  const uint32x4_t vsign_mask = vmovq_n_u32(UINT32_C(0x80000000));

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;
    const float32x4_t vx89AB = vld1q_f32(input); input += 4;
    const float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    float32x4_t vz0123 = vabsq_f32(vx0123);
    float32x4_t vz4567 = vabsq_f32(vx4567);
    float32x4_t vz89AB = vabsq_f32(vx89AB);
    float32x4_t vzCDEF = vabsq_f32(vxCDEF);
    vz0123 = vminq_f32(vz0123, vsat_cutoff);
    vz4567 = vminq_f32(vz4567, vsat_cutoff);
    vz89AB = vminq_f32(vz89AB, vsat_cutoff);
    vzCDEF = vminq_f32(vzCDEF, vsat_cutoff);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vminus_log2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vz89AB, vminus_log2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vzCDEF, vminus_log2e);

    const uint32x4_t ve0123 = vshlq_n_u32(vreinterpretq_u32_f32(vn0123), 20);
    const uint64x2_t vidx0123 = vandq_u64(vreinterpretq_u64_f32(vn0123), vindex_mask);
    const uint32x4_t ve4567 = vshlq_n_u32(vreinterpretq_u32_f32(vn4567), 20);
    const uint64x2_t vidx4567 = vandq_u64(vreinterpretq_u64_f32(vn4567), vindex_mask);
    const uint32x4_t ve89AB = vshlq_n_u32(vreinterpretq_u32_f32(vn89AB), 20);
    const uint64x2_t vidx89AB = vandq_u64(vreinterpretq_u64_f32(vn89AB), vindex_mask);
    const uint32x4_t veCDEF = vshlq_n_u32(vreinterpretq_u32_f32(vnCDEF), 20);
    const uint64x2_t vidxCDEF = vandq_u64(vreinterpretq_u64_f32(vnCDEF), vindex_mask);
    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0);
    uint32x2_t vl01 = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx01]);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    uint32x2_t vl45 = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx45]);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    uint32x2_t vl89 = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx89]);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);
    uint32x2_t vlCD = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidxCD]);
    const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1);
    uint32x2_t vl23 = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx23]);
    uint32x2_t vl67 = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx67]);
    uint32x2_t vlAB = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidxAB]);
    uint32x2_t vlEF = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidxEF]);
    vl01 = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx23 >> 32)], vl23, 1);
    vl45 = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx67 >> 32)], vl67, 1);
    vl89 = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx89 >> 32)], vl89, 1);
    vlAB = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidxAB >> 32)], vlAB, 1);
    vlCD = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidxCD >> 32)], vlCD, 1);
    vlEF = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidxEF >> 32)], vlEF, 1);
    const uint32x4_t vl0123 = vcombine_u32(vl01, vl23);
    const uint32x4_t vl4567 = vcombine_u32(vl45, vl67);
    const uint32x4_t vl89AB = vcombine_u32(vl89, vlAB);
    const uint32x4_t vlCDEF = vcombine_u32(vlCD, vlEF);
    const float32x4_t vs0123 = vreinterpretq_f32_u32(vaddq_u32(vl0123, ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_u32(vaddq_u32(vl4567, ve4567));
    const float32x4_t vs89AB = vreinterpretq_f32_u32(vaddq_u32(vl89AB, ve89AB));
    const float32x4_t vsCDEF = vreinterpretq_f32_u32(vaddq_u32(vlCDEF, veCDEF));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);

    const float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vln2);
    const float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vln2);
    const float32x4_t vt89AB = vfmaq_f32(vz89AB, vn89AB, vln2);
    const float32x4_t vtCDEF = vfmaq_f32(vzCDEF, vnCDEF, vln2);

    float32x4_t vp0123 = vfmaq_f32(vc3, vc4, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc3, vc4, vt4567);
    float32x4_t vp89AB = vfmaq_f32(vc3, vc4, vt89AB);
    float32x4_t vpCDEF = vfmaq_f32(vc3, vc4, vtCDEF);
    vp0123 = vfmaq_f32(vc2, vp0123, vt0123);
    vp4567 = vfmaq_f32(vc2, vp4567, vt4567);
    vp89AB = vfmaq_f32(vc2, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vc2, vpCDEF, vtCDEF);
    vp0123 = vfmsq_f32(vtwo, vp0123, vt0123);
    vp4567 = vfmsq_f32(vtwo, vp4567, vt4567);
    vp89AB = vfmsq_f32(vtwo, vp89AB, vt89AB);
    vpCDEF = vfmsq_f32(vtwo, vpCDEF, vtCDEF);

    const float32x4_t vts0123 = vmulq_f32(vt0123, vs0123);
    const float32x4_t vsmo0123 = vsubq_f32(vs0123, vone);
    const float32x4_t vts4567 = vmulq_f32(vt4567, vs4567);
    const float32x4_t vsmo4567 = vsubq_f32(vs4567, vone);
    const float32x4_t vts89AB = vmulq_f32(vt89AB, vs89AB);
    const float32x4_t vsmo89AB = vsubq_f32(vs89AB, vone);
    const float32x4_t vtsCDEF = vmulq_f32(vtCDEF, vsCDEF);
    const float32x4_t vsmoCDEF = vsubq_f32(vsCDEF, vone);
    const float32x4_t vemo0123 = vfmsq_f32(vsmo0123, vp0123, vts0123);
    const float32x4_t vemo4567 = vfmsq_f32(vsmo4567, vp4567, vts4567);
    const float32x4_t vemo89AB = vfmsq_f32(vsmo89AB, vp89AB, vts89AB);
    const float32x4_t vemoCDEF = vfmsq_f32(vsmoCDEF, vpCDEF, vtsCDEF);

    const float32x4_t vepo0123 = vaddq_f32(vemo0123, vtwo);
    const float32x4_t vepo4567 = vaddq_f32(vemo4567, vtwo);
    const float32x4_t vepo89AB = vaddq_f32(vemo89AB, vtwo);
    const float32x4_t vepoCDEF = vaddq_f32(vemoCDEF, vtwo);

    float32x4_t vrepo0123 = vrecpeq_f32(vepo0123);
    float32x4_t vrepo4567 = vrecpeq_f32(vepo4567);
    float32x4_t vrepo89AB = vrecpeq_f32(vepo89AB);
    float32x4_t vrepoCDEF = vrecpeq_f32(vepoCDEF);
    float32x4_t verepo0123 = vrecpsq_f32(vrepo0123, vepo0123);
    float32x4_t verepo4567 = vrecpsq_f32(vrepo4567, vepo4567);
    float32x4_t verepo89AB = vrecpsq_f32(vrepo89AB, vepo89AB);
    float32x4_t verepoCDEF = vrecpsq_f32(vrepoCDEF, vepoCDEF);
    vrepo0123 = vmulq_f32(vrepo0123, verepo0123);
    vrepo4567 = vmulq_f32(vrepo4567, verepo4567);
    vrepo89AB = vmulq_f32(vrepo89AB, verepo89AB);
    vrepoCDEF = vmulq_f32(vrepoCDEF, verepoCDEF);
    verepo0123 = vfmsq_f32(vone, vrepo0123, vepo0123);
    verepo4567 = vfmsq_f32(vone, vrepo4567, vepo4567);
    verepo89AB = vfmsq_f32(vone, vrepo89AB, vepo89AB);
    verepoCDEF = vfmsq_f32(vone, vrepoCDEF, vepoCDEF);
    vrepo0123 = vfmaq_f32(vrepo0123, vrepo0123, verepo0123);
    vrepo4567 = vfmaq_f32(vrepo4567, vrepo4567, verepo4567);
    vrepo89AB = vfmaq_f32(vrepo89AB, vrepo89AB, verepo89AB);
    vrepoCDEF = vfmaq_f32(vrepoCDEF, vrepoCDEF, verepoCDEF);

    float32x4_t vy0123 = vmulq_f32(vemo0123, vrepo0123);
    float32x4_t vy4567 = vmulq_f32(vemo4567, vrepo4567);
    float32x4_t vy89AB = vmulq_f32(vemo89AB, vrepo89AB);
    float32x4_t vyCDEF = vmulq_f32(vemoCDEF, vrepoCDEF);

    vy0123 = vbslq_f32(vsign_mask, vx0123, vy0123);
    vy4567 = vbslq_f32(vsign_mask, vx4567, vy4567);
    vy89AB = vbslq_f32(vsign_mask, vx89AB, vy89AB);
    vyCDEF = vbslq_f32(vsign_mask, vxCDEF, vyCDEF);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
  }

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    const uint32x4_t ve = vshlq_n_u32(vreinterpretq_u32_f32(vn), 20);
    const uint64x2_t vidx = vandq_u64(vreinterpretq_u64_f32(vn), vindex_mask);
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    uint32x2_t vl_lo = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
    uint32x2_t vl_hi = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const uint32x4_t vl = vcombine_u32(vl_lo, vl_hi);
    const float32x4_t vs = vreinterpretq_f32_u32(vaddq_u32(vl, ve));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc3, vc4, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);
    verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);

    float32x4_t vy = vmulq_f32(vemo, vrepo);

    vy = vbslq_f32(vsign_mask, vx, vy);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    float32x4_t vz = vabsq_f32(vx);
    vz = vminq_f32(vz, vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vminus_log2e);

    const uint32x4_t ve = vshlq_n_u32(vreinterpretq_u32_f32(vn), 20);
    const uint64x2_t vidx = vandq_u64(vreinterpretq_u64_f32(vn), vindex_mask);
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    uint32x2_t vl_lo = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx_lo]);
    uint32x2_t vl_hi = vld1_dup_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_u32(&xnn_table_exp2minus_k_over_8[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const uint32x4_t vl = vcombine_u32(vl_lo, vl_hi);
    const float32x4_t vs = vreinterpretq_f32_u32(vaddq_u32(vl, ve));

    vn = vsubq_f32(vn, vmagic_bias);

    const float32x4_t vt = vfmaq_f32(vz, vn, vln2);

    float32x4_t vp = vfmaq_f32(vc3, vc4, vt);
    vp = vfmaq_f32(vc2, vp, vt);
    vp = vfmsq_f32(vtwo, vp, vt);

    const float32x4_t vts = vmulq_f32(vt, vs);
    const float32x4_t vsmo = vsubq_f32(vs, vone);
    const float32x4_t vemo = vfmsq_f32(vsmo, vp, vts);

    const float32x4_t vepo = vaddq_f32(vemo, vtwo);

    float32x4_t vrepo = vrecpeq_f32(vepo);
    float32x4_t verepo = vrecpsq_f32(vrepo, vepo);
    vrepo = vmulq_f32(vrepo, verepo);
    verepo = vfmsq_f32(vone, vrepo, vepo);
    vrepo = vfmaq_f32(vrepo, vrepo, verepo);

    float32x4_t vy = vmulq_f32(vemo, vrepo);

    vy = vbslq_f32(vsign_mask, vx, vy);

    float32x2_t vy_low = vget_low_f32(vy);

    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_low); output += 2;
      vy_low = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_low, 0);
    }
  }
}
