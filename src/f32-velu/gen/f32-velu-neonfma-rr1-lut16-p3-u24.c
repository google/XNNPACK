// Auto-generated file. Do not edit!
//   Template: src/f32-velu/neon-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_u24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vsat_cutoff = vmovq_n_f32(-0x1.154246p+4f);
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p19f);
  const float32x4_t vlog2e = vmovq_n_f32(0x1.715476p+0f);
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0xF));
  const float32x4_t vc3 = vmovq_n_f32(0x1.55561Cp-3f);
  const float32x4_t vc2 = vmovq_n_f32(0x1.0001ECp-1f);
  const float32x4_t vone = vmovq_n_f32(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vminus_ln2);

  const float32x4_t vprescale = vld1q_dup_f32(&params->scalar.prescale);
  const float32x4_t valpha = vld1q_dup_f32(&params->scalar.alpha);
  const float32x4_t vbeta = vld1q_dup_f32(&params->scalar.beta);

  for (; batch >= 24 * sizeof(float); batch -= 24 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;
    float32x4_t vx89AB = vld1q_f32(input); input += 4;
    float32x4_t vxCDEF = vld1q_f32(input); input += 4;
    float32x4_t vxGHIJ = vld1q_f32(input); input += 4;
    float32x4_t vxKLMN = vld1q_f32(input); input += 4;

    const float32x4_t vz0123 = vmaxq_f32(vmulq_f32(vx0123, vprescale), vsat_cutoff);
    const float32x4_t vz4567 = vmaxq_f32(vmulq_f32(vx4567, vprescale), vsat_cutoff);
    const float32x4_t vz89AB = vmaxq_f32(vmulq_f32(vx89AB, vprescale), vsat_cutoff);
    const float32x4_t vzCDEF = vmaxq_f32(vmulq_f32(vxCDEF, vprescale), vsat_cutoff);
    const float32x4_t vzGHIJ = vmaxq_f32(vmulq_f32(vxGHIJ, vprescale), vsat_cutoff);
    const float32x4_t vzKLMN = vmaxq_f32(vmulq_f32(vxKLMN, vprescale), vsat_cutoff);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vz0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vz4567, vlog2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vz89AB, vlog2e);
    float32x4_t vnCDEF = vfmaq_f32(vmagic_bias, vzCDEF, vlog2e);
    float32x4_t vnGHIJ = vfmaq_f32(vmagic_bias, vzGHIJ, vlog2e);
    float32x4_t vnKLMN = vfmaq_f32(vmagic_bias, vzKLMN, vlog2e);

    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask), 2));
    const int32x4_t ven0123 = vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 19);
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask), 2));
    const int32x4_t ven4567 = vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 19);
    const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask), 2));
    const int32x4_t ven89AB = vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 19);
    const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask), 2));
    const int32x4_t venCDEF = vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 19);
    const uint64x2_t vidxGHIJ = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vnGHIJ), vindex_mask), 2));
    const int32x4_t venGHIJ = vshlq_n_s32(vreinterpretq_s32_f32(vnGHIJ), 19);
    const uint64x2_t vidxKLMN = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vnKLMN), vindex_mask), 2));
    const int32x4_t venKLMN = vshlq_n_s32(vreinterpretq_s32_f32(vnKLMN), 19);

    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    int32x2_t vl01 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx01));
    int32x2_t vl23 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx23));
    vl01 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx01 >> 32)), vl01, 1);
    vl23 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx23 >> 32)), vl23, 1);
    const int32x4_t vl0123 = vcombine_s32(vl01, vl23);
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    int32x2_t vl45 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx45));
    int32x2_t vl67 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx67));
    vl45 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx45 >> 32)), vl45, 1);
    vl67 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx67 >> 32)), vl67, 1);
    const int32x4_t vl4567 = vcombine_s32(vl45, vl67);
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);
    int32x2_t vl89 = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx89));
    int32x2_t vlAB = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxAB));
    vl89 = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx89 >> 32)), vl89, 1);
    vlAB = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxAB >> 32)), vlAB, 1);
    const int32x4_t vl89AB = vcombine_s32(vl89, vlAB);
    const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0);
    const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1);
    int32x2_t vlCD = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxCD));
    int32x2_t vlEF = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxEF));
    vlCD = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxCD >> 32)), vlCD, 1);
    vlEF = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxEF >> 32)), vlEF, 1);
    const int32x4_t vlCDEF = vcombine_s32(vlCD, vlEF);
    const uint64_t vidxGH = vgetq_lane_u64(vidxGHIJ, 0);
    const uint64_t vidxIJ = vgetq_lane_u64(vidxGHIJ, 1);
    int32x2_t vlGH = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxGH));
    int32x2_t vlIJ = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxIJ));
    vlGH = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxGH >> 32)), vlGH, 1);
    vlIJ = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxIJ >> 32)), vlIJ, 1);
    const int32x4_t vlGHIJ = vcombine_s32(vlGH, vlIJ);
    const uint64_t vidxKL = vgetq_lane_u64(vidxKLMN, 0);
    const uint64_t vidxMN = vgetq_lane_u64(vidxKLMN, 1);
    int32x2_t vlKL = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxKL));
    int32x2_t vlMN = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxMN));
    vlKL = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxKL >> 32)), vlKL, 1);
    vlMN = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxMN >> 32)), vlMN, 1);
    const int32x4_t vlKLMN = vcombine_s32(vlKL, vlMN);

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vl0123, ven0123));
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vl4567, ven4567));
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vl89AB, ven89AB));
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);
    float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vlCDEF, venCDEF));
    vnGHIJ = vsubq_f32(vnGHIJ, vmagic_bias);
    float32x4_t vsGHIJ = vreinterpretq_f32_s32(vaddq_s32(vlGHIJ, venGHIJ));
    vnKLMN = vsubq_f32(vnKLMN, vmagic_bias);
    float32x4_t vsKLMN = vreinterpretq_f32_s32(vaddq_s32(vlKLMN, venKLMN));

    float32x4_t vt0123 = vfmaq_f32(vz0123, vn0123, vminus_ln2);
    float32x4_t vt4567 = vfmaq_f32(vz4567, vn4567, vminus_ln2);
    float32x4_t vt89AB = vfmaq_f32(vz89AB, vn89AB, vminus_ln2);
    float32x4_t vtCDEF = vfmaq_f32(vzCDEF, vnCDEF, vminus_ln2);
    float32x4_t vtGHIJ = vfmaq_f32(vzGHIJ, vnGHIJ, vminus_ln2);
    float32x4_t vtKLMN = vfmaq_f32(vzKLMN, vnKLMN, vminus_ln2);

    float32x4_t vp0123 = vfmaq_f32(vc2, vc3, vt0123);
    float32x4_t vp4567 = vfmaq_f32(vc2, vc3, vt4567);
    float32x4_t vp89AB = vfmaq_f32(vc2, vc3, vt89AB);
    float32x4_t vpCDEF = vfmaq_f32(vc2, vc3, vtCDEF);
    float32x4_t vpGHIJ = vfmaq_f32(vc2, vc3, vtGHIJ);
    float32x4_t vpKLMN = vfmaq_f32(vc2, vc3, vtKLMN);

    vp0123 = vmulq_f32(vp0123, vt0123);
    vp4567 = vmulq_f32(vp4567, vt4567);
    vp89AB = vmulq_f32(vp89AB, vt89AB);
    vpCDEF = vmulq_f32(vpCDEF, vtCDEF);
    vpGHIJ = vmulq_f32(vpGHIJ, vtGHIJ);
    vpKLMN = vmulq_f32(vpKLMN, vtKLMN);

    vt0123 = vmulq_f32(vt0123, vs0123);
    vs0123 = vsubq_f32(vs0123, vone);
    vt4567 = vmulq_f32(vt4567, vs4567);
    vs4567 = vsubq_f32(vs4567, vone);
    vt89AB = vmulq_f32(vt89AB, vs89AB);
    vs89AB = vsubq_f32(vs89AB, vone);
    vtCDEF = vmulq_f32(vtCDEF, vsCDEF);
    vsCDEF = vsubq_f32(vsCDEF, vone);
    vtGHIJ = vmulq_f32(vtGHIJ, vsGHIJ);
    vsGHIJ = vsubq_f32(vsGHIJ, vone);
    vtKLMN = vmulq_f32(vtKLMN, vsKLMN);
    vsKLMN = vsubq_f32(vsKLMN, vone);

    vp0123 = vfmaq_f32(vt0123, vp0123, vt0123);
    vp4567 = vfmaq_f32(vt4567, vp4567, vt4567);
    vp89AB = vfmaq_f32(vt89AB, vp89AB, vt89AB);
    vpCDEF = vfmaq_f32(vtCDEF, vpCDEF, vtCDEF);
    vpGHIJ = vfmaq_f32(vtGHIJ, vpGHIJ, vtGHIJ);
    vpKLMN = vfmaq_f32(vtKLMN, vpKLMN, vtKLMN);

    const float32x4_t ve0123 = vmulq_f32(vaddq_f32(vp0123, vs0123), valpha);
    const float32x4_t ve4567 = vmulq_f32(vaddq_f32(vp4567, vs4567), valpha);
    const float32x4_t ve89AB = vmulq_f32(vaddq_f32(vp89AB, vs89AB), valpha);
    const float32x4_t veCDEF = vmulq_f32(vaddq_f32(vpCDEF, vsCDEF), valpha);
    const float32x4_t veGHIJ = vmulq_f32(vaddq_f32(vpGHIJ, vsGHIJ), valpha);
    const float32x4_t veKLMN = vmulq_f32(vaddq_f32(vpKLMN, vsKLMN), valpha);

    const uint32x4_t vm0123 = vcltq_f32(vx0123, vmovq_n_f32(0.0f));
    vx0123 = vmulq_f32(vx0123, vbeta);
    const uint32x4_t vm4567 = vcltq_f32(vx4567, vmovq_n_f32(0.0f));
    vx4567 = vmulq_f32(vx4567, vbeta);
    const uint32x4_t vm89AB = vcltq_f32(vx89AB, vmovq_n_f32(0.0f));
    vx89AB = vmulq_f32(vx89AB, vbeta);
    const uint32x4_t vmCDEF = vcltq_f32(vxCDEF, vmovq_n_f32(0.0f));
    vxCDEF = vmulq_f32(vxCDEF, vbeta);
    const uint32x4_t vmGHIJ = vcltq_f32(vxGHIJ, vmovq_n_f32(0.0f));
    vxGHIJ = vmulq_f32(vxGHIJ, vbeta);
    const uint32x4_t vmKLMN = vcltq_f32(vxKLMN, vmovq_n_f32(0.0f));
    vxKLMN = vmulq_f32(vxKLMN, vbeta);

    const float32x4_t vy0123 = vbslq_f32(vm0123, ve0123, vx0123);
    const float32x4_t vy4567 = vbslq_f32(vm4567, ve4567, vx4567);
    const float32x4_t vy89AB = vbslq_f32(vm89AB, ve89AB, vx89AB);
    const float32x4_t vyCDEF = vbslq_f32(vmCDEF, veCDEF, vxCDEF);
    const float32x4_t vyGHIJ = vbslq_f32(vmGHIJ, veGHIJ, vxGHIJ);
    const float32x4_t vyKLMN = vbslq_f32(vmKLMN, veKLMN, vxKLMN);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
    vst1q_f32(output, vy89AB); output += 4;
    vst1q_f32(output, vyCDEF); output += 4;
    vst1q_f32(output, vyGHIJ); output += 4;
    vst1q_f32(output, vyKLMN); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vlog2e);
    const uint64x2_t vidx = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask), 2));
    const int32x4_t ven = vshlq_n_s32(vreinterpretq_s32_f32(vn), 19);

    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    int32x2_t vl_lo = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    int32x2_t vl_hi = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    vl_lo = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)), vl_lo, 1);
    vl_hi = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)), vl_hi, 1);

    vn = vsubq_f32(vn, vmagic_bias);
    const int32x4_t vl = vcombine_s32(vl_lo, vl_hi);

    float32x4_t vt = vfmaq_f32(vz, vn, vminus_ln2);
    float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vl, ven));

    float32x4_t vp = vfmaq_f32(vc2, vc3, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vmaxq_f32(vmulq_f32(vx, vprescale), vsat_cutoff);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vz, vlog2e);
    const uint64x2_t vidx = vreinterpretq_u64_s32(vshlq_n_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask), 2));
    const int32x4_t ven = vshlq_n_s32(vreinterpretq_s32_f32(vn), 19);

    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    int32x2_t vl_lo = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo));
    int32x2_t vl_hi = vld1_dup_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi));
    vl_lo = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32)), vl_lo, 1);
    vl_hi = vld1_lane_s32((const int32_t*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32)), vl_hi, 1);

    vn = vsubq_f32(vn, vmagic_bias);
    const int32x4_t vl = vcombine_s32(vl_lo, vl_hi);

    float32x4_t vt = vfmaq_f32(vz, vn, vminus_ln2);
    float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vl, ven));

    float32x4_t vp = vfmaq_f32(vc2, vc3, vt);
    vp = vmulq_f32(vp, vt);

    vt = vmulq_f32(vt, vs);
    vs = vsubq_f32(vs, vone);
    vp = vfmaq_f32(vt, vp, vt);
    const float32x4_t ve = vmulq_f32(vaddq_f32(vp, vs), valpha);

    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vx = vmulq_f32(vx, vbeta);
    const float32x4_t vy = vbslq_f32(vm, ve, vx);

    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
