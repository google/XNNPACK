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
#include "src/xnnpack/gemm.h"
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

static const int8_t xnn_qc4w_lut[64] = {
    0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0, 0, 3,  0, 0, 0,
    4,  0, 0, 0, 5,  0, 0, 0, 6,  0, 0, 0, 7,  0, 0, 0,
    -8, 0, 0, 0, -7, 0, 0, 0, -6, 0, 0, 0, -5, 0, 0, 0,
    -4, 0, 0, 0, -3, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0};

static XNN_INLINE svint8x4_t xnn_luti4_s8_x4(svuint8_t v0,
                                             svuint8_t v1) __arm_streaming
    __arm_in("zt0") __arm_preserves("za") {
  const svint8x2_t vlo = svluti4_lane_zt_s8_x2(0, v0, 0);
  const svint8x2_t vhi = svluti4_lane_zt_s8_x2(0, v1, 0);
  return svcreate4_s8(svget2_s8(vlo, 0), svget2_s8(vlo, 1), svget2_s8(vhi, 0),
                      svget2_s8(vhi, 1));
}

static XNN_INLINE svint32x4_t xnn_pqs8_qc4w_load_bias_x4(
    const uint8_t* p0, const uint8_t* p1) __arm_streaming
    __arm_preserves("za", "zt0") {
  const svcount_t pc32 = svptrue_c32();
  const svint32x2_t b0 = svld1_s32_x2(pc32, (const int32_t*)p0);
  const svint32x2_t b1 = svld1_s32_x2(pc32, (const int32_t*)p1);
  return svcreate4_s32(svget2_s32(b0, 0), svget2_s32(b0, 1), svget2_s32(b1, 0),
                       svget2_s32(b1, 1));
}

static XNN_INLINE void xnn_pqs8_qc4w_epilogue_4vl(
    svint32x4_t vacc, const uint8_t* p0, const uint8_t* p1, size_t scale_off,
    svint32_t vout_zp, svint32_t vmin, svint32_t vmax, int8_t* dst_row,
    size_t n_idx, size_t n) __arm_streaming __arm_preserves("za", "zt0") {
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc32 = svptrue_c32();
  const svfloat32x2_t s0 = svld1_f32_x2(pc32, (const float*)(p0 + scale_off));
  const svfloat32x2_t s1 = svld1_f32_x2(pc32, (const float*)(p1 + scale_off));
  const svfloat32x4_t vf = svcvt_f32_s32_x4(vacc);
  const svfloat32x4_t vscaled = svcreate4_f32(
      svmul_f32_x(pg32, svget4_f32(vf, 0), svget2_f32(s0, 0)),
      svmul_f32_x(pg32, svget4_f32(vf, 1), svget2_f32(s0, 1)),
      svmul_f32_x(pg32, svget4_f32(vf, 2), svget2_f32(s1, 0)),
      svmul_f32_x(pg32, svget4_f32(vf, 3), svget2_f32(s1, 1)));
  const svint32x4_t vq = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled));
  const svint32x4_t vclamped = svclamp_single_s32_x4(
      svcreate4_s32(svqadd_s32(svget4_s32(vq, 0), vout_zp),
                    svqadd_s32(svget4_s32(vq, 1), vout_zp),
                    svqadd_s32(svget4_s32(vq, 2), vout_zp),
                    svqadd_s32(svget4_s32(vq, 3), vout_zp)),
      vmin, vmax);
  const svint16_t vh01 = svuzp1_s16(
      svreinterpret_s16_s32(svget4_s32(vclamped, 0)),
      svreinterpret_s16_s32(svget4_s32(vclamped, 1)));
  const svint16_t vh23 = svuzp1_s16(
      svreinterpret_s16_s32(svget4_s32(vclamped, 2)),
      svreinterpret_s16_s32(svget4_s32(vclamped, 3)));
  const svint8_t vb0123 =
      svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh23));
  svst1_s8(svwhilelt_b8_u64(n_idx, n), dst_row + n_idx, vb0123);
}

__arm_locally_streaming __arm_new("za", "zt0") static XNN_NOINLINE void
xnn_pqs8_qc4w_gemm_1x64vs_sme_dot_impl(
    size_t m, size_t n, size_t k_elements, const int8_t* lhs,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    int32_t output_zero_point, int32_t output_min, int32_t output_max) {
  const size_t vl = svcntw();
  const size_t n_panel = 2 * vl;
  const size_t rounded_k = (k_elements + 31) & ~(size_t)31;
  const size_t w_off = n_panel * sizeof(int32_t);
  const size_t scale_off = w_off + n_panel * (rounded_k / 2);
  const size_t panel_bytes = scale_off + n_panel * sizeof(float);
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc8 = svptrue_c8();
  const svcount_t pc32 = svptrue_c32();
  const svint32_t vout_zp = svdup_n_s32(output_zero_point);
  const svint32_t vmin = svdup_n_s32(output_min);
  const svint32_t vmax = svdup_n_s32(output_max);
  svldr_zt(0, xnn_qc4w_lut);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* lhs_row = lhs + m_idx * k_elements;
    int8_t* dst_row = (int8_t*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 6 * n_panel < n; n_idx += 8 * n_panel) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = p0 + panel_bytes;
      const uint8_t* p2 = p1 + panel_bytes;
      const uint8_t* p3 = p2 + panel_bytes;
      const uint8_t* p4 = p3 + panel_bytes;
      const uint8_t* p5 = p4 + panel_bytes;
      const uint8_t* p6 = p5 + panel_bytes;
      const uint8_t* p7 = (n_idx + 7 * n_panel < n) ? (p6 + panel_bytes) : p6;

      svwrite_za32_s32_vg1x4(0, xnn_pqs8_qc4w_load_bias_x4(p0, p1));
      svwrite_za32_s32_vg1x4(1, xnn_pqs8_qc4w_load_bias_x4(p2, p3));
      svwrite_za32_s32_vg1x4(2, xnn_pqs8_qc4w_load_bias_x4(p4, p5));
      svwrite_za32_s32_vg1x4(3, xnn_pqs8_qc4w_load_bias_x4(p6, p7));

      const uint8_t* w0 = p0 + w_off;
      const uint8_t* w1 = p1 + w_off;
      const uint8_t* w2 = p2 + w_off;
      const uint8_t* w3 = p3 + w_off;
      const uint8_t* w4 = p4 + w_off;
      const uint8_t* w5 = p5 + w_off;
      const uint8_t* w6 = p6 + w_off;
      const uint8_t* w7 = p7 + w_off;

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        const svuint8x4_t vr1 = svld1_u8_x4(pc8, w1);
        const svuint8x4_t vr2 = svld1_u8_x4(pc8, w2);
        const svuint8x4_t vr3 = svld1_u8_x4(pc8, w3);
        const svuint8x4_t vr4 = svld1_u8_x4(pc8, w4);
        const svuint8x4_t vr5 = svld1_u8_x4(pc8, w5);
        const svuint8x4_t vr6 = svld1_u8_x4(pc8, w6);
        const svuint8x4_t vr7 = svld1_u8_x4(pc8, w7);

        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 0), svget4_u8(vr1, 0)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 0), svget4_u8(vr3, 0)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            2, xnn_luti4_s8_x4(svget4_u8(vr4, 0), svget4_u8(vr5, 0)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            3, xnn_luti4_s8_x4(svget4_u8(vr6, 0), svget4_u8(vr7, 0)), va, 0);

        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 1), svget4_u8(vr1, 1)), va, 1);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 1), svget4_u8(vr3, 1)), va, 1);
        svdot_lane_za32_s8_vg1x4(
            2, xnn_luti4_s8_x4(svget4_u8(vr4, 1), svget4_u8(vr5, 1)), va, 1);
        svdot_lane_za32_s8_vg1x4(
            3, xnn_luti4_s8_x4(svget4_u8(vr6, 1), svget4_u8(vr7, 1)), va, 1);

        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 2), svget4_u8(vr1, 2)), va, 2);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 2), svget4_u8(vr3, 2)), va, 2);
        svdot_lane_za32_s8_vg1x4(
            2, xnn_luti4_s8_x4(svget4_u8(vr4, 2), svget4_u8(vr5, 2)), va, 2);
        svdot_lane_za32_s8_vg1x4(
            3, xnn_luti4_s8_x4(svget4_u8(vr6, 2), svget4_u8(vr7, 2)), va, 2);

        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 3), svget4_u8(vr1, 3)), va, 3);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 3), svget4_u8(vr3, 3)), va, 3);
        svdot_lane_za32_s8_vg1x4(
            2, xnn_luti4_s8_x4(svget4_u8(vr4, 3), svget4_u8(vr5, 3)), va, 3);
        svdot_lane_za32_s8_vg1x4(
            3, xnn_luti4_s8_x4(svget4_u8(vr6, 3), svget4_u8(vr7, 3)), va, 3);

        w0 += 8 * n_panel;
        w1 += 8 * n_panel;
        w2 += 8 * n_panel;
        w3 += 8 * n_panel;
        w4 += 8 * n_panel;
        w5 += 8 * n_panel;
        w6 += 8 * n_panel;
        w7 += 8 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svld1_u8(pg8, w0), svld1_u8(pg8, w1)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svld1_u8(pg8, w2), svld1_u8(pg8, w3)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            2, xnn_luti4_s8_x4(svld1_u8(pg8, w4), svld1_u8(pg8, w5)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            3, xnn_luti4_s8_x4(svld1_u8(pg8, w6), svld1_u8(pg8, w7)), va, 0);
        w0 += 2 * n_panel;
        w1 += 2 * n_panel;
        w2 += 2 * n_panel;
        w3 += 2 * n_panel;
        w4 += 2 * n_panel;
        w5 += 2 * n_panel;
        w6 += 2 * n_panel;
        w7 += 2 * n_panel;
      }

      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(0), p0, p1, scale_off,
                                 vout_zp, vmin, vmax, dst_row,
                                 n_idx + 0 * n_panel, n);
      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(1), p2, p3, scale_off,
                                 vout_zp, vmin, vmax, dst_row,
                                 n_idx + 2 * n_panel, n);
      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(2), p4, p5, scale_off,
                                 vout_zp, vmin, vmax, dst_row,
                                 n_idx + 4 * n_panel, n);
      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(3), p6, p7, scale_off,
                                 vout_zp, vmin, vmax, dst_row,
                                 n_idx + 6 * n_panel, n);
    }

    for (; n_idx + 2 * n_panel < n; n_idx += 4 * n_panel) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = p0 + panel_bytes;
      const uint8_t* p2 = p1 + panel_bytes;
      const uint8_t* p3 = (n_idx + 3 * n_panel < n) ? (p2 + panel_bytes) : p2;

      svwrite_za32_s32_vg1x4(0, xnn_pqs8_qc4w_load_bias_x4(p0, p1));
      svwrite_za32_s32_vg1x4(1, xnn_pqs8_qc4w_load_bias_x4(p2, p3));

      const uint8_t* w0 = p0 + w_off;
      const uint8_t* w1 = p1 + w_off;
      const uint8_t* w2 = p2 + w_off;
      const uint8_t* w3 = p3 + w_off;

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        const svuint8x4_t vr1 = svld1_u8_x4(pc8, w1);
        const svuint8x4_t vr2 = svld1_u8_x4(pc8, w2);
        const svuint8x4_t vr3 = svld1_u8_x4(pc8, w3);

        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 0), svget4_u8(vr1, 0)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 0), svget4_u8(vr3, 0)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 1), svget4_u8(vr1, 1)), va, 1);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 1), svget4_u8(vr3, 1)), va, 1);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 2), svget4_u8(vr1, 2)), va, 2);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 2), svget4_u8(vr3, 2)), va, 2);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 3), svget4_u8(vr1, 3)), va, 3);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svget4_u8(vr2, 3), svget4_u8(vr3, 3)), va, 3);

        w0 += 8 * n_panel;
        w1 += 8 * n_panel;
        w2 += 8 * n_panel;
        w3 += 8 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svld1_u8(pg8, w0), svld1_u8(pg8, w1)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            1, xnn_luti4_s8_x4(svld1_u8(pg8, w2), svld1_u8(pg8, w3)), va, 0);
        w0 += 2 * n_panel;
        w1 += 2 * n_panel;
        w2 += 2 * n_panel;
        w3 += 2 * n_panel;
      }

      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(0), p0, p1, scale_off,
                                 vout_zp, vmin, vmax, dst_row,
                                 n_idx + 0 * n_panel, n);
      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(1), p2, p3, scale_off,
                                 vout_zp, vmin, vmax, dst_row,
                                 n_idx + 2 * n_panel, n);
    }

    if (n_idx + n_panel < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = p0 + panel_bytes;
      svwrite_za32_s32_vg1x4(0, xnn_pqs8_qc4w_load_bias_x4(p0, p1));
      const uint8_t* w0 = p0 + w_off;
      const uint8_t* w1 = p1 + w_off;

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        const svuint8x4_t vr1 = svld1_u8_x4(pc8, w1);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 0), svget4_u8(vr1, 0)), va, 0);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 1), svget4_u8(vr1, 1)), va, 1);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 2), svget4_u8(vr1, 2)), va, 2);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svget4_u8(vr0, 3), svget4_u8(vr1, 3)), va, 3);
        w0 += 8 * n_panel;
        w1 += 8 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(
            0, xnn_luti4_s8_x4(svld1_u8(pg8, w0), svld1_u8(pg8, w1)), va, 0);
        w0 += 2 * n_panel;
        w1 += 2 * n_panel;
      }

      xnn_pqs8_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(0), p0, p1, scale_off,
                                 vout_zp, vmin, vmax, dst_row, n_idx, n);
      n_idx += 2 * n_panel;
    }

    if (n_idx < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      svwrite_za32_s32_vg1x2(0, svld1_s32_x2(pc32, (const int32_t*)p0));
      const uint8_t* w0 = p0 + w_off;

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        svdot_lane_za32_s8_vg1x2(
            0, svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x2(
            0, svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x2(
            0, svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x2(
            0, svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 3), 0), va, 3);
        w0 += 8 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x2(0, svluti4_lane_zt_s8_x2(0, svld1_u8(pg8, w0), 0),
                                 va, 0);
        w0 += 2 * n_panel;
      }

      const svint32x2_t vacc = svread_za32_s32_vg1x2(0);
      const svfloat32x2_t s0 =
          svld1_f32_x2(pc32, (const float*)(p0 + scale_off));
      const svfloat32x2_t vf = svcvt_f32_s32_x2(vacc);
      const svfloat32x2_t vscaled = svcreate2_f32(
          svmul_f32_x(pg32, svget2_f32(vf, 0), svget2_f32(s0, 0)),
          svmul_f32_x(pg32, svget2_f32(vf, 1), svget2_f32(s0, 1)));
      const svfloat32x2_t vrnd = svrintn_f32_x2(vscaled);
      svint32_t vq0 = svcvt_s32_f32_x(pg32, svget2_f32(vrnd, 0));
      svint32_t vq1 = svcvt_s32_f32_x(pg32, svget2_f32(vrnd, 1));
      vq0 = svmin_s32_x(
          pg32, svmax_s32_x(pg32, svqadd_s32(vq0, vout_zp), vmin), vmax);
      vq1 = svmin_s32_x(
          pg32, svmax_s32_x(pg32, svqadd_s32(vq1, vout_zp), vmin), vmax);
      const svint16_t vh01 =
          svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
      const svint8_t vb01 =
          svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh01));
      svst1_s8(svwhilelt_b8_u64(n_idx, n), dst_row + n_idx, vb01);
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
