// Auto-generated file. Do not edit!
//   Template: src/f32-raddstoreexpminusmax/neon-lut64-p2.c.in
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


extern XNN_INTERNAL const float xnn_table_exp2_k_over_64[64];

void xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc2(
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
  const float32x4_t vmagic_bias = vmovq_n_f32(0x1.800000p17f);
  const int32x4_t vindex_mask = vmovq_n_s32(INT32_C(0x3F));
  const float32x4_t vc2 = vmovq_n_f32(0x1.FFFF0Ap-2f);
  const float32x4_t vdenorm_cutoff = vmovq_n_f32(-0x1.5D589Ep6f);

  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vdenorm_cutoff);

  const float32x4_t vminus_ln2 = vmovq_n_f32(-0x1.62E430p-1f);
  XNN_FORCE_REALIZATION(vminus_ln2);

  const float32x4_t vi_max = vld1q_dup_f32(max);

  float32x4_t vacc0 = vmovq_n_f32(0.0f);
  float32x4_t vacc1 = vmovq_n_f32(0.0f);
  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const float32x4_t vi0123 = vld1q_f32(input); input += 4;
    const float32x4_t vi4567 = vld1q_f32(input); input += 4;
    const float32x4_t vi89AB = vld1q_f32(input); input += 4;

    const float32x4_t vx0123 = vsubq_f32(vi0123, vi_max);
    const float32x4_t vx4567 = vsubq_f32(vi4567, vi_max);
    const float32x4_t vx89AB = vsubq_f32(vi89AB, vi_max);

    float32x4_t vn0123 = vfmaq_f32(vmagic_bias, vx0123, vlog2e);
    float32x4_t vn4567 = vfmaq_f32(vmagic_bias, vx4567, vlog2e);
    float32x4_t vn89AB = vfmaq_f32(vmagic_bias, vx89AB, vlog2e);

    const int32x4_t ve0123 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn0123), vmovq_n_s32(INT32_C(0x3F))), 17);
    const int32x4_t ve4567 = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn4567), vmovq_n_s32(INT32_C(0x3F))), 17);
    const int32x4_t ve89AB = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn89AB), vmovq_n_s32(INT32_C(0x3F))), 17);

    const uint64x2_t vidx0123 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn0123), vindex_mask));
    const uint64_t vidx01 = vgetq_lane_u64(vidx0123, 0);
    const uint64_t vidx23 = vgetq_lane_u64(vidx0123, 1);
    const uint64x2_t vidx4567 = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn4567), vindex_mask));
    const uint64_t vidx45 = vgetq_lane_u64(vidx4567, 0);
    const uint64_t vidx67 = vgetq_lane_u64(vidx4567, 1);
    const uint64x2_t vidx89AB = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn89AB), vindex_mask));
    const uint64_t vidx89 = vgetq_lane_u64(vidx89AB, 0);
    const uint64_t vidxAB = vgetq_lane_u64(vidx89AB, 1);

    float32x2_t vl01 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx01]);
    float32x2_t vl23 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx23]);
    float32x2_t vl45 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx45]);
    float32x2_t vl67 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx67]);
    float32x2_t vl89 = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx89]);
    float32x2_t vlAB = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidxAB]);

    vl01 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx01 >> 32)], vl01, 1);
    vl23 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx23 >> 32)], vl23, 1);
    const float32x4_t vl0123 = vcombine_f32(vl01, vl23);
    vl45 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx45 >> 32)], vl45, 1);
    vl67 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx67 >> 32)], vl67, 1);
    const float32x4_t vl4567 = vcombine_f32(vl45, vl67);
    vl89 = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx89 >> 32)], vl89, 1);
    vlAB = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidxAB >> 32)], vlAB, 1);
    const float32x4_t vl89AB = vcombine_f32(vl89, vlAB);

    const float32x4_t vs0123 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl0123), ve0123));
    const float32x4_t vs4567 = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl4567), ve4567));
    const float32x4_t vs89AB = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl89AB), ve89AB));

    vn0123 = vsubq_f32(vn0123, vmagic_bias);
    vn4567 = vsubq_f32(vn4567, vmagic_bias);
    vn89AB = vsubq_f32(vn89AB, vmagic_bias);

    float32x4_t vt0123 = vfmaq_f32(vx0123, vn0123, vminus_ln2);
    float32x4_t vt4567 = vfmaq_f32(vx4567, vn4567, vminus_ln2);
    float32x4_t vt89AB = vfmaq_f32(vx89AB, vn89AB, vminus_ln2);

    float32x4_t vp0123 = vmulq_f32(vt0123, vc2);
    float32x4_t vp4567 = vmulq_f32(vt4567, vc2);
    float32x4_t vp89AB = vmulq_f32(vt89AB, vc2);

    vp0123 = vfmaq_f32(vt0123, vt0123, vp0123);
    vp4567 = vfmaq_f32(vt4567, vt4567, vp4567);
    vp89AB = vfmaq_f32(vt89AB, vt89AB, vp89AB);

    float32x4_t vf0123 = vfmaq_f32(vs0123, vs0123, vp0123);
    float32x4_t vf4567 = vfmaq_f32(vs4567, vs4567, vp4567);
    float32x4_t vf89AB = vfmaq_f32(vs89AB, vs89AB, vp89AB);

    vf0123 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf0123), vcltq_f32(vx0123, vdenorm_cutoff)));
    vf4567 = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf4567), vcltq_f32(vx4567, vdenorm_cutoff)));
    vf89AB = vreinterpretq_f32_u32(vbicq_u32(vreinterpretq_u32_f32(vf89AB), vcltq_f32(vx89AB, vdenorm_cutoff)));

    vst1q_f32(output, vf0123); output += 4;
    vst1q_f32(output, vf4567); output += 4;
    vst1q_f32(output, vf89AB); output += 4;

    vacc0 = vaddq_f32(vacc0, vf0123);
    vacc0 = vaddq_f32(vacc0, vf4567);
    vacc0 = vaddq_f32(vacc0, vf89AB);
  }
  vacc0 = vaddq_f32(vacc0, vacc1);

  float32x4_t vacc = vacc0;
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vi = vld1q_f32(input); input += 4;

    const float32x4_t vx = vsubq_f32(vi, vi_max);

    float32x4_t vn = vfmaq_f32(vmagic_bias, vx, vlog2e);

    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmaq_f32(vt, vt, vp);

    float32x4_t vf = vfmaq_f32(vs, vs, vp);

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

    const int32x4_t ve = vshlq_n_s32(vbicq_s32(vreinterpretq_s32_f32(vn), vmovq_n_s32(INT32_C(0x3F))), 17);

    const uint64x2_t vidx = vreinterpretq_u64_s32(vandq_s32(vreinterpretq_s32_f32(vn), vindex_mask));
    const uint64_t vidx_lo = vgetq_lane_u64(vidx, 0);
    const uint64_t vidx_hi = vgetq_lane_u64(vidx, 1);
    float32x2_t vl_lo = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_lo]);
    float32x2_t vl_hi = vld1_dup_f32(&xnn_table_exp2_k_over_64[(uint32_t) vidx_hi]);
    vl_lo = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_lo >> 32)], vl_lo, 1);
    vl_hi = vld1_lane_f32(&xnn_table_exp2_k_over_64[(uint32_t) (vidx_hi >> 32)], vl_hi, 1);
    const float32x4_t vl = vcombine_f32(vl_lo, vl_hi);
    const float32x4_t vs = vreinterpretq_f32_s32(vaddq_s32(vreinterpretq_s32_f32(vl), ve));

    vn = vsubq_f32(vn, vmagic_bias);

    float32x4_t vt = vfmaq_f32(vx, vn, vminus_ln2);

    float32x4_t vp = vmulq_f32(vt, vc2);
    vp = vfmaq_f32(vt, vt, vp);

    float32x4_t vf = vfmaq_f32(vs, vs, vp);

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
