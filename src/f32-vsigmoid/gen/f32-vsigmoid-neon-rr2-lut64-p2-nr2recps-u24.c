// Auto-generated file. Do not edit!
//   Template: src/f32-vsigmoid/neon-lut64-p2.c.in
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


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_64[64];

void xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_u24(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p17f);
  const float32x4_t vminus_log2e = vmovq_n_f32(-0x1.715476p0f);
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);
  const float32x4_t vone = vmovq_n_f32(1.0f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(0x1.5D589Ep+6f);

  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vminus_log2e);
  // XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vc2);
  // XNN_FORCE_REALIZATION(vone);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float32x4_t vln2_hi = vmovq_n_f32(0x1.630000p-1f);
  const float32x4_t vln2_lo = vmovq_n_f32(-0x1.BD0106p-13f);
  XNN_FORCE_REALIZATION(vln2_hi);
  XNN_FORCE_REALIZATION(vln2_lo);

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

    float32x4_t vn0123 = vmlaq_f32(vmagic_bias, vz0123, vminus_log2e);
    float32x4_t vn4567 = vmlaq_f32(vmagic_bias, vz4567, vminus_log2e);
    float32x4_t vn89AB = vmlaq_f32(vmagic_bias, vz89AB, vminus_log2e);
    float32x4_t vnCDEF = vmlaq_f32(vmagic_bias, vzCDEF, vminus_log2e);
    float32x4_t vnGHIJ = vmlaq_f32(vmagic_bias, vzGHIJ, vminus_log2e);
    float32x4_t vnKLMN = vmlaq_f32(vmagic_bias, vzKLMN, vminus_log2e);

    const int32x4_t ve0123 = vshlq_n_s32(vreinterpretq_s32_f32(vn0123), 17);
    const int32x4_t ve4567 = vshlq_n_s32(vreinterpretq_s32_f32(vn4567), 17);
    const int32x4_t ve89AB = vshlq_n_s32(vreinterpretq_s32_f32(vn89AB), 17);
    const int32x4_t veCDEF = vshlq_n_s32(vreinterpretq_s32_f32(vnCDEF), 17);
    const int32x4_t veGHIJ = vshlq_n_s32(vreinterpretq_s32_f32(vnGHIJ), 17);
    const int32x4_t veKLMN = vshlq_n_s32(vreinterpretq_s32_f32(vnKLMN), 17);

    // Use bits 0:6 bits of batch, as integer, as an index for table lookup of l := 2**(batch % 64).
    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask));
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask));
    const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask));
    const uint64x2_t vidxCDEF = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnCDEF), vindex_mask));
    const uint64x2_t vidxGHIJ = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnGHIJ), vindex_mask));
    const uint64x2_t vidxKLMN = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vnKLMN), vindex_mask));

    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    float32x2_t vl01 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx23]);
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    float32x2_t vl45 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx45]);
    float32x2_t vl67 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx67]);
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);
    float32x2_t vl89 = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx89]);
    float32x2_t vlAB = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxAB]);
    const uint64_t vidxCD = vgetq_lane_u64(vidxCDEF, 0);
    const uint64_t vidxEF = vgetq_lane_u64(vidxCDEF, 1);
    float32x2_t vlCD = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxCD]);
    float32x2_t vlEF = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxEF]);
    const uint64_t vidxGH = vgetq_lane_u64(vidxGHIJ, 0);
    const uint64_t vidxIJ = vgetq_lane_u64(vidxGHIJ, 1);
    float32x2_t vlGH = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxGH]);
    float32x2_t vlIJ = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxIJ]);
    const uint64_t vidxKL = vgetq_lane_u64(vidxKLMN, 0);
    const uint64_t vidxMN = vgetq_lane_u64(vidxKLMN, 1);
    float32x2_t vlKL = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxKL]);
    float32x2_t vlMN = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidxMN]);

    vl01 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl0123 = vcombine_f32(vl01, vl23);
    vl45 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1);
    const float32x4_t vl4567 = vcombine_f32(vl45, vl67);
    vl89 = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx89 >> 32)], vl89, 1);
    vlAB = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxAB >> 32)], vlAB, 1);
    const float32x4_t vl89AB = vcombine_f32(vl89, vlAB);
    vlCD = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxCD >> 32)], vlCD, 1);
    vlEF = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxEF >> 32)], vlEF, 1);
    const float32x4_t vlCDEF = vcombine_f32(vlCD, vlEF);
    vlGH = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxGH >> 32)], vlGH, 1);
    vlIJ = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxIJ >> 32)], vlIJ, 1);
    const float32x4_t vlGHIJ = vcombine_f32(vlGH, vlIJ);
    vlKL = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxKL >> 32)], vlKL, 1);
    vlMN = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidxMN >> 32)], vlMN, 1);
    const float32x4_t vlKLMN = vcombine_f32(vlKL, vlMN);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl89AB), ve89AB));
    const float32x4_t vsCDEF = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlCDEF), veCDEF));
    const float32x4_t vsGHIJ = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlGHIJ), veGHIJ));
    const float32x4_t vsKLMN = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vlKLMN), veKLMN));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);
    vnCDEF = vsubq_f32(vnCDEF, vmagic_bias);
    vnGHIJ = vsubq_f32(vnGHIJ, vmagic_bias);
    vnKLMN = vsubq_f32(vnKLMN, vmagic_bias);

    float32x4_t vt0123 = vmlaq_f32(vz0123, vn0123, vln2_hi);
    float32x4_t vt4567 = vmlaq_f32(vz4567, vn4567, vln2_hi);
    float32x4_t vt89AB = vmlaq_f32(vz89AB, vn89AB, vln2_hi);
    float32x4_t vtCDEF = vmlaq_f32(vzCDEF, vnCDEF, vln2_hi);
    float32x4_t vtGHIJ = vmlaq_f32(vzGHIJ, vnGHIJ, vln2_hi);
    float32x4_t vtKLMN = vmlaq_f32(vzKLMN, vnKLMN, vln2_hi);

    vt0123 = vmlaq_f32(vt0123, vn0123, vln2_lo);
    vt4567 = vmlaq_f32(vt4567, vn4567, vln2_lo);
    vt89AB = vmlaq_f32(vt89AB, vn89AB, vln2_lo);
    vtCDEF = vmlaq_f32(vtCDEF, vnCDEF, vln2_lo);
    vtGHIJ = vmlaq_f32(vtGHIJ, vnGHIJ, vln2_lo);
    vtKLMN = vmlaq_f32(vtKLMN, vnKLMN, vln2_lo);

    float32x4_t vp0123 = vmulq_f32(vt0123, vc2);
    float32x4_t vp4567 = vmulq_f32(vt4567, vc2);
    float32x4_t vp89AB = vmulq_f32(vt89AB, vc2);
    float32x4_t vpCDEF = vmulq_f32(vtCDEF, vc2);
    float32x4_t vpGHIJ = vmulq_f32(vtGHIJ, vc2);
    float32x4_t vpKLMN = vmulq_f32(vtKLMN, vc2);

    vp0123 = vmlsq_f32(vt0123, vp0123, vt0123);
    vp4567 = vmlsq_f32(vt4567, vp4567, vt4567);
    vp89AB = vmlsq_f32(vt89AB, vp89AB, vt89AB);
    vpCDEF = vmlsq_f32(vtCDEF, vpCDEF, vtCDEF);
    vpGHIJ = vmlsq_f32(vtGHIJ, vpGHIJ, vtGHIJ);
    vpKLMN = vmlsq_f32(vtKLMN, vpKLMN, vtKLMN);

    const float32x4_t vy0123 = vmlsq_f32(vs0123, vs0123, vp0123);
    const float32x4_t vy4567 = vmlsq_f32(vs4567, vs4567, vp4567);
    const float32x4_t vy89AB = vmlsq_f32(vs89AB, vs89AB, vp89AB);
    const float32x4_t vyCDEF = vmlsq_f32(vsCDEF, vsCDEF, vpCDEF);
    const float32x4_t vyGHIJ = vmlsq_f32(vsGHIJ, vsGHIJ, vpGHIJ);
    const float32x4_t vyKLMN = vmlsq_f32(vsKLMN, vsKLMN, vpKLMN);

    const float32x4_t vd0123 = vaddq_f32(vy0123, vone);
    const float32x4_t vd4567 = vaddq_f32(vy4567, vone);
    const float32x4_t vd89AB = vaddq_f32(vy89AB, vone);
    const float32x4_t vdCDEF = vaddq_f32(vyCDEF, vone);
    const float32x4_t vdGHIJ = vaddq_f32(vyGHIJ, vone);
    const float32x4_t vdKLMN = vaddq_f32(vyKLMN, vone);

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

    float32x4_t vf0123 = vmulq_f32(vy0123, vr0123);
    float32x4_t vf4567 = vmulq_f32(vy4567, vr4567);
    float32x4_t vf89AB = vmulq_f32(vy89AB, vr89AB);
    float32x4_t vfCDEF = vmulq_f32(vyCDEF, vrCDEF);
    float32x4_t vfGHIJ = vmulq_f32(vyGHIJ, vrGHIJ);
    float32x4_t vfKLMN = vmulq_f32(vyKLMN, vrKLMN);

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

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);
    const int32x4_t ve = vshlq_n_s32(vreinterpretq_s32_f32(vn), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);

    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vmlaq_f32(vz, vn, vln2_hi);
    vt = vmlaq_f32(vt, vn, vln2_lo);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vmlsq_f32(vt, vp, vt);

    const float32x4_t vy = vmlsq_f32(vs, vs, vp);
    const float32x4_t vd = vaddq_f32(vy, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    float32x4_t vf = vmulq_f32(vy, vr);
    vf = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf), vcagtq_f32(vx, vdenorm_cutoff)));
    const uint32x4_t vm = vcltq_f32(vx, vmovq_n_f32(0.0f));
    vf = vbslq_f32(vm, vf, vsubq_f32(vone, vf));

    vst1q_f32(output, vf); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);

    const float32x4_t vz = vabsq_f32(vx);

    float32x4_t vn = vmlaq_f32(vmagic_bias, vz, vminus_log2e);
    const int32x4_t ve = vshlq_n_s32(vreinterpretq_s32_f32(vn), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2minus_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);

    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));
    vn = vsubq_f32(vn, vmagic_bias);
    float32x4_t vt = vmlaq_f32(vz, vn, vln2_hi);
    vt = vmlaq_f32(vt, vn, vln2_lo);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vmlsq_f32(vt, vp, vt);

    const float32x4_t vy = vmlsq_f32(vs, vs, vp);
    const float32x4_t vd = vaddq_f32(vy, vone);

    float32x4_t vr = vrecpeq_f32(vd);
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));
    vr = vmulq_f32(vr, vrecpsq_f32(vr, vd));

    float32x4_t vf = vmulq_f32(vy, vr);
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
