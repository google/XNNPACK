// Copyright 2026 Google LLC
//
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_mr(void) {
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
  return api.get_step(NULL).m;
}
size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_nr(void) {
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
  return api.get_step(NULL).n;
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  assert(m == 1);
  assert(dst_stride_col == sizeof(int8_t));

  const int32_t output_zero_point =
      minmax_params->fp32_scalar.output_zero_point;
  const int32_t output_min = minmax_params->fp32_scalar.output_min;
  const int32_t output_max = minmax_params->fp32_scalar.output_max;
  struct kai_matmul_uker_args args = {0};
  args.flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP;
  args.shape.m = m;
  args.shape.n = n;
  args.shape.k = k / sizeof(int8_t);
  args.operand.lhs.ptr = lhs_packed;
  args.operand.rhs.ptr = rhs_packed;
  args.operand.dst.ptr = dst;
  args.operand.dst.stride.m = dst_stride_row;
  args.operand.bias.scale_bias_global.ptr = &output_zero_point;
  args.activation.clamp.min_ptr = &output_min;
  args.activation.clamp.max_ptr = &output_max;

  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8_qsi4cxp8vsx4sf32bi32_1x64vs_sme2_dot();
  api.run(NULL, &args);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_mr(void) {
  return 1;
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_nr(void) {
  return 16 * svcntsw();
}

__arm_locally_streaming static void xnn_pqs8_qc4w_gemm_1x64vs_sme_dot_impl(
    size_t m, size_t n, size_t k_elements, const int8_t* lhs,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    int32_t output_zero_point, int32_t output_min, int32_t output_max) {
  const size_t vl = svcntw();
  const size_t n_panel = 2 * vl;
  const size_t rounded_k = (k_elements + 31) & ~(size_t)31;
  const size_t panel_bytes =
      n_panel * (sizeof(int32_t) + (rounded_k / 2) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svint32_t vout_zp = svdup_n_s32(output_zero_point);
  const svint32_t vmin = svdup_n_s32(output_min);
  const svint32_t vmax = svdup_n_s32(output_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* lhs_row = lhs + m_idx * k_elements;
    int8_t* dst_row = (int8_t*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 4 * n_panel <= n; n_idx += 4 * n_panel) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = p0 + panel_bytes;
      const uint8_t* p2 = p1 + panel_bytes;
      const uint8_t* p3 = p2 + panel_bytes;

      svint32_t vacc0 = svdup_n_s32(0);
      svint32_t vacc1 = svdup_n_s32(0);
      svint32_t vacc2 = svdup_n_s32(0);
      svint32_t vacc3 = svdup_n_s32(0);
      svint32_t vacc4 = svdup_n_s32(0);
      svint32_t vacc5 = svdup_n_s32(0);
      svint32_t vacc6 = svdup_n_s32(0);
      svint32_t vacc7 = svdup_n_s32(0);

      const int8_t* w0 = (const int8_t*)(p0 + n_panel * sizeof(int32_t));
      const int8_t* w1 = (const int8_t*)(p1 + n_panel * sizeof(int32_t));
      const int8_t* w2 = (const int8_t*)(p2 + n_panel * sizeof(int32_t));
      const int8_t* w3 = (const int8_t*)(p3 + n_panel * sizeof(int32_t));

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
#pragma clang loop unroll(full)
        for (int s = 0; s < 4; ++s) {
          const svint8_t vp0 = svld1_vnum_s8(pg8, w0, s);
          const svint8_t vp1 = svld1_vnum_s8(pg8, w1, s);
          const svint8_t vp2 = svld1_vnum_s8(pg8, w2, s);
          const svint8_t vp3 = svld1_vnum_s8(pg8, w3, s);
          const svint8_t vlo0 = svlsl_n_s8_x(pg8, vp0, 4);
          const svint8_t vhi0 = svand_n_s8_x(pg8, vp0, (int8_t)0xF0);
          const svint8_t vlo1 = svlsl_n_s8_x(pg8, vp1, 4);
          const svint8_t vhi1 = svand_n_s8_x(pg8, vp1, (int8_t)0xF0);
          const svint8_t vlo2 = svlsl_n_s8_x(pg8, vp2, 4);
          const svint8_t vhi2 = svand_n_s8_x(pg8, vp2, (int8_t)0xF0);
          const svint8_t vlo3 = svlsl_n_s8_x(pg8, vp3, 4);
          const svint8_t vhi3 = svand_n_s8_x(pg8, vp3, (int8_t)0xF0);
          const svint8_t vb0 = svzip1_s8(vlo0, vhi0);
          const svint8_t vb1 = svzip2_s8(vlo0, vhi0);
          const svint8_t vb2 = svzip1_s8(vlo1, vhi1);
          const svint8_t vb3 = svzip2_s8(vlo1, vhi1);
          const svint8_t vb4 = svzip1_s8(vlo2, vhi2);
          const svint8_t vb5 = svzip2_s8(vlo2, vhi2);
          const svint8_t vb6 = svzip1_s8(vlo3, vhi3);
          const svint8_t vb7 = svzip2_s8(vlo3, vhi3);
          if (s == 0) {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 0);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 0);
            vacc2 = svdot_lane_s32(vacc2, vb2, va, 0);
            vacc3 = svdot_lane_s32(vacc3, vb3, va, 0);
            vacc4 = svdot_lane_s32(vacc4, vb4, va, 0);
            vacc5 = svdot_lane_s32(vacc5, vb5, va, 0);
            vacc6 = svdot_lane_s32(vacc6, vb6, va, 0);
            vacc7 = svdot_lane_s32(vacc7, vb7, va, 0);
          } else if (s == 1) {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 1);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 1);
            vacc2 = svdot_lane_s32(vacc2, vb2, va, 1);
            vacc3 = svdot_lane_s32(vacc3, vb3, va, 1);
            vacc4 = svdot_lane_s32(vacc4, vb4, va, 1);
            vacc5 = svdot_lane_s32(vacc5, vb5, va, 1);
            vacc6 = svdot_lane_s32(vacc6, vb6, va, 1);
            vacc7 = svdot_lane_s32(vacc7, vb7, va, 1);
          } else if (s == 2) {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 2);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 2);
            vacc2 = svdot_lane_s32(vacc2, vb2, va, 2);
            vacc3 = svdot_lane_s32(vacc3, vb3, va, 2);
            vacc4 = svdot_lane_s32(vacc4, vb4, va, 2);
            vacc5 = svdot_lane_s32(vacc5, vb5, va, 2);
            vacc6 = svdot_lane_s32(vacc6, vb6, va, 2);
            vacc7 = svdot_lane_s32(vacc7, vb7, va, 2);
          } else {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 3);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 3);
            vacc2 = svdot_lane_s32(vacc2, vb2, va, 3);
            vacc3 = svdot_lane_s32(vacc3, vb3, va, 3);
            vacc4 = svdot_lane_s32(vacc4, vb4, va, 3);
            vacc5 = svdot_lane_s32(vacc5, vb5, va, 3);
            vacc6 = svdot_lane_s32(vacc6, vb6, va, 3);
            vacc7 = svdot_lane_s32(vacc7, vb7, va, 3);
          }
        }
        w0 += 8 * n_panel;
        w1 += 8 * n_panel;
        w2 += 8 * n_panel;
        w3 += 8 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        const svint8_t vp0 = svld1_s8(pg8, w0);
        const svint8_t vp1 = svld1_s8(pg8, w1);
        const svint8_t vp2 = svld1_s8(pg8, w2);
        const svint8_t vp3 = svld1_s8(pg8, w3);
        const svint8_t vlo0 = svlsl_n_s8_x(pg8, vp0, 4);
        const svint8_t vhi0 = svand_n_s8_x(pg8, vp0, (int8_t)0xF0);
        const svint8_t vlo1 = svlsl_n_s8_x(pg8, vp1, 4);
        const svint8_t vhi1 = svand_n_s8_x(pg8, vp1, (int8_t)0xF0);
        const svint8_t vlo2 = svlsl_n_s8_x(pg8, vp2, 4);
        const svint8_t vhi2 = svand_n_s8_x(pg8, vp2, (int8_t)0xF0);
        const svint8_t vlo3 = svlsl_n_s8_x(pg8, vp3, 4);
        const svint8_t vhi3 = svand_n_s8_x(pg8, vp3, (int8_t)0xF0);
        vacc0 = svdot_lane_s32(vacc0, svzip1_s8(vlo0, vhi0), va, 0);
        vacc1 = svdot_lane_s32(vacc1, svzip2_s8(vlo0, vhi0), va, 0);
        vacc2 = svdot_lane_s32(vacc2, svzip1_s8(vlo1, vhi1), va, 0);
        vacc3 = svdot_lane_s32(vacc3, svzip2_s8(vlo1, vhi1), va, 0);
        vacc4 = svdot_lane_s32(vacc4, svzip1_s8(vlo2, vhi2), va, 0);
        vacc5 = svdot_lane_s32(vacc5, svzip2_s8(vlo2, vhi2), va, 0);
        vacc6 = svdot_lane_s32(vacc6, svzip1_s8(vlo3, vhi3), va, 0);
        vacc7 = svdot_lane_s32(vacc7, svzip2_s8(vlo3, vhi3), va, 0);
        w0 += 2 * n_panel;
        w1 += 2 * n_panel;
        w2 += 2 * n_panel;
        w3 += 2 * n_panel;
      }

      vacc0 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc0, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p0, 0));
      vacc1 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc1, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p0, 1));
      vacc2 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc2, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p1, 0));
      vacc3 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc3, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p1, 1));
      vacc4 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc4, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p2, 0));
      vacc5 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc5, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p2, 1));
      vacc6 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc6, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p3, 0));
      vacc7 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc7, 4),
                          svld1_vnum_s32(pg32, (const int32_t*)p3, 1));

      const size_t scale_off = n_panel * (sizeof(int32_t) + (rounded_k / 2));
      const float* s0 = (const float*)(p0 + scale_off);
      const float* s1 = (const float*)(p1 + scale_off);
      const float* s2 = (const float*)(p2 + scale_off);
      const float* s3 = (const float*)(p3 + scale_off);

      svint32_t vq0 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc0),
                                          svld1_vnum_f32(pg32, s0, 0))));
      svint32_t vq1 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc1),
                                          svld1_vnum_f32(pg32, s0, 1))));
      svint32_t vq2 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc2),
                                          svld1_vnum_f32(pg32, s1, 0))));
      svint32_t vq3 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc3),
                                          svld1_vnum_f32(pg32, s1, 1))));
      svint32_t vq4 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc4),
                                          svld1_vnum_f32(pg32, s2, 0))));
      svint32_t vq5 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc5),
                                          svld1_vnum_f32(pg32, s2, 1))));
      svint32_t vq6 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc6),
                                          svld1_vnum_f32(pg32, s3, 0))));
      svint32_t vq7 = svcvt_s32_f32_x(
          pg32,
          svrintn_f32_x(pg32, svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc7),
                                          svld1_vnum_f32(pg32, s3, 1))));

      vq0 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq0, vout_zp), vmin), vmax);
      vq1 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq1, vout_zp), vmin), vmax);
      vq2 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq2, vout_zp), vmin), vmax);
      vq3 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq3, vout_zp), vmin), vmax);
      vq4 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq4, vout_zp), vmin), vmax);
      vq5 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq5, vout_zp), vmin), vmax);
      vq6 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq6, vout_zp), vmin), vmax);
      vq7 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq7, vout_zp), vmin), vmax);

      const svint16_t vh01 =
          svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
      const svint16_t vh23 =
          svuzp1_s16(svreinterpret_s16_s32(vq2), svreinterpret_s16_s32(vq3));
      const svint16_t vh45 =
          svuzp1_s16(svreinterpret_s16_s32(vq4), svreinterpret_s16_s32(vq5));
      const svint16_t vh67 =
          svuzp1_s16(svreinterpret_s16_s32(vq6), svreinterpret_s16_s32(vq7));
      const svint8_t vb0123 =
          svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh23));
      const svint8_t vb4567 =
          svuzp1_s8(svreinterpret_s8_s16(vh45), svreinterpret_s8_s16(vh67));
      svst1_s8(pg8, dst_row + n_idx, vb0123);
      svst1_s8(pg8, dst_row + n_idx + 4 * vl, vb4567);
    }

    for (; n_idx < n; n_idx += n_panel) {
      const uint8_t* panel =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const int32_t* bias_ptr = (const int32_t*)panel;
      const int8_t* w_ptr = (const int8_t*)(panel + n_panel * sizeof(int32_t));
      const float* scale_ptr =
          (const float*)(panel + n_panel * (sizeof(int32_t) + (rounded_k / 2)));

      svint32_t vacc0 = svdup_n_s32(0);
      svint32_t vacc1 = svdup_n_s32(0);

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
#pragma clang loop unroll(full)
        for (int s = 0; s < 4; ++s) {
          const svint8_t vpacked = svld1_vnum_s8(pg8, w_ptr, s);
          const svint8_t vlo = svlsl_n_s8_x(pg8, vpacked, 4);
          const svint8_t vhi = svand_n_s8_x(pg8, vpacked, (int8_t)0xF0);
          const svint8_t vb0 = svzip1_s8(vlo, vhi);
          const svint8_t vb1 = svzip2_s8(vlo, vhi);
          if (s == 0) {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 0);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 0);
          } else if (s == 1) {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 1);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 1);
          } else if (s == 2) {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 2);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 2);
          } else {
            vacc0 = svdot_lane_s32(vacc0, vb0, va, 3);
            vacc1 = svdot_lane_s32(vacc1, vb1, va, 3);
          }
        }
        w_ptr += 8 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        const svint8_t vpacked = svld1_s8(pg8, w_ptr);
        const svint8_t vlo = svlsl_n_s8_x(pg8, vpacked, 4);
        const svint8_t vhi = svand_n_s8_x(pg8, vpacked, (int8_t)0xF0);
        vacc0 = svdot_lane_s32(vacc0, svzip1_s8(vlo, vhi), va, 0);
        vacc1 = svdot_lane_s32(vacc1, svzip2_s8(vlo, vhi), va, 0);
        w_ptr += 2 * n_panel;
      }

      vacc0 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc0, 4),
                          svld1_vnum_s32(pg32, bias_ptr, 0));
      vacc1 = svadd_s32_x(pg32, svasr_n_s32_x(pg32, vacc1, 4),
                          svld1_vnum_s32(pg32, bias_ptr, 1));

      const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr, 0);
      const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr, 1);
      const svfloat32_t vf0 =
          svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
      const svfloat32_t vf1 =
          svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
      svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
      svint32_t vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
      vq0 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq0, vout_zp), vmin), vmax);
      vq1 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq1, vout_zp), vmin), vmax);

      const size_t n_rem = n - n_idx < n_panel ? n - n_idx : n_panel;
      const svint16_t vh01 =
          svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
      const svint8_t vb01 =
          svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh01));
      svst1_s8(svwhilelt_b8_u64(0, n_rem), dst_row + n_idx, vb01);
    }
  }
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  assert(m == 1);
  assert(dst_stride_col == sizeof(int8_t));
  (void)dst_stride_col;
  xnn_pqs8_qc4w_gemm_1x64vs_sme_dot_impl(
      m, n, k / sizeof(int8_t), (const int8_t*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->fp32_scalar.output_zero_point,
      minmax_params->fp32_scalar.output_min,
      minmax_params->fp32_scalar.output_max);
}
#else
size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_mr(void) {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}
size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2_get_nr(void) {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
