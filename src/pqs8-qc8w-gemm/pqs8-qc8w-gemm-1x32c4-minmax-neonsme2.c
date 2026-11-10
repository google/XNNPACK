// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8_qsi8cxp/kai_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot.h"

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_mr() {
  return kai_get_m_step_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot();
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot();
}

// Wraps the
// `kai_run_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot` GEMM
// microkernel with a name that is compatible with our tooling.
void xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = minmax_params->fp32_scalar.output_zero_point;
  kai_params.min_value = minmax_params->fp32_scalar.output_min;
  kai_params.max_value = minmax_params->fp32_scalar.output_max;

  kai_run_matmul_clamp_qai8_qai8_qsi8cxp2vlx4sb_1x16vl_sme2_dot(
      m, n, k / sizeof(int8_t), lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(int8_t), &kai_params);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <assert.h>
#include <stdint.h>

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_mr() { return 1; }

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_nr() {
  return 2 * svcntsw();
}

static XNN_INLINE svint32x4_t
xnn_pqs8_load_bias_x4(const uint8_t* p0,
                      const uint8_t* p1) __arm_streaming __arm_preserves("za") {
  const svcount_t pc32 = svptrue_c32();
  const svint32x2_t b0 = svld1_s32_x2(pc32, (const int32_t*)p0);
  const svint32x2_t b1 = svld1_s32_x2(pc32, (const int32_t*)p1);
  return svcreate4_s32(svget2_s32(b0, 0), svget2_s32(b0, 1), svget2_s32(b1, 0),
                       svget2_s32(b1, 1));
}

static XNN_INLINE svint8x4_t
xnn_pqs8_load_w_2panels(const int8_t* wa, const int8_t* wb,
                        int64_t vnum) __arm_streaming __arm_preserves("za") {
  const svcount_t pc8 = svptrue_c8();
  const svint8x2_t va = svld1_vnum_s8_x2(pc8, wa, vnum);
  const svint8x2_t vb = svld1_vnum_s8_x2(pc8, wb, vnum);
  return svcreate4_s8(svget2_s8(va, 0), svget2_s8(va, 1), svget2_s8(vb, 0),
                      svget2_s8(vb, 1));
}

static XNN_INLINE void xnn_pqs8_epilogue_4vl(
    svint32x4_t vacc, const uint8_t* p0, const uint8_t* p1, size_t scale_off,
    svint32_t vout_zp, svint32_t vmin, svint32_t vmax, int8_t* dst_row,
    size_t n_idx, size_t n) __arm_streaming __arm_preserves("za") {
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc32 = svptrue_c32();
  const svfloat32x2_t s0 = svld1_f32_x2(pc32, (const float*)(p0 + scale_off));
  const svfloat32x2_t s1 = svld1_f32_x2(pc32, (const float*)(p1 + scale_off));
  const svfloat32x4_t vf = svcvt_f32_s32_x4(vacc);
  const svfloat32x4_t vscaled =
      svcreate4_f32(svmul_f32_x(pg32, svget4_f32(vf, 0), svget2_f32(s0, 0)),
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
  const svint16_t vh01 =
      svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vclamped, 0)),
                 svreinterpret_s16_s32(svget4_s32(vclamped, 1)));
  const svint16_t vh23 =
      svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vclamped, 2)),
                 svreinterpret_s16_s32(svget4_s32(vclamped, 3)));
  const svint8_t vb0123 =
      svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh23));
  svst1_s8(svwhilelt_b8_u64(n_idx, n), dst_row + n_idx, vb0123);
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pqs8_qc8w_gemm_1x16vl_sme_dot_impl(
    size_t m, size_t n, size_t k_elements, const int8_t* lhs,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    int32_t output_zero_point, int32_t output_min, int32_t output_max) {
  const size_t vl = svcntw();
  const size_t n_panel = 2 * vl;
  const size_t k_rounded = (k_elements + 3) & ~(size_t)3;
  const size_t w_off = n_panel * sizeof(int32_t);
  const size_t scale_off = w_off + n_panel * k_rounded * sizeof(int8_t);
  const size_t panel_bytes = scale_off + n_panel * sizeof(float);
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc8 = svptrue_c8();
  const svcount_t pc32 = svptrue_c32();
  const svint32_t vout_zp = svdup_n_s32(output_zero_point);
  const svint32_t vmin = svdup_n_s32(output_min);
  const svint32_t vmax = svdup_n_s32(output_max);

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

      svwrite_za32_s32_vg1x4(0, xnn_pqs8_load_bias_x4(p0, p1));
      svwrite_za32_s32_vg1x4(1, xnn_pqs8_load_bias_x4(p2, p3));
      svwrite_za32_s32_vg1x4(2, xnn_pqs8_load_bias_x4(p4, p5));
      svwrite_za32_s32_vg1x4(3, xnn_pqs8_load_bias_x4(p6, p7));

      const int8_t* w0 = (const int8_t*)(p0 + w_off);
      const int8_t* w1 = (const int8_t*)(p1 + w_off);
      const int8_t* w2 = (const int8_t*)(p2 + w_off);
      const int8_t* w3 = (const int8_t*)(p3 + w_off);
      const int8_t* w4 = (const int8_t*)(p4 + w_off);
      const int8_t* w5 = (const int8_t*)(p5 + w_off);
      const int8_t* w6 = (const int8_t*)(p6 + w_off);
      const int8_t* w7 = (const int8_t*)(p7 + w_off);

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(2, xnn_pqs8_load_w_2panels(w4, w5, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(3, xnn_pqs8_load_w_2panels(w6, w7, 0), va, 0);

        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(2, xnn_pqs8_load_w_2panels(w4, w5, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(3, xnn_pqs8_load_w_2panels(w6, w7, 2), va, 1);

        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(2, xnn_pqs8_load_w_2panels(w4, w5, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(3, xnn_pqs8_load_w_2panels(w6, w7, 4), va, 2);

        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(2, xnn_pqs8_load_w_2panels(w4, w5, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(3, xnn_pqs8_load_w_2panels(w6, w7, 6), va, 3);

        w0 += 16 * n_panel;
        w1 += 16 * n_panel;
        w2 += 16 * n_panel;
        w3 += 16 * n_panel;
        w4 += 16 * n_panel;
        w5 += 16 * n_panel;
        w6 += 16 * n_panel;
        w7 += 16 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(2, xnn_pqs8_load_w_2panels(w4, w5, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(3, xnn_pqs8_load_w_2panels(w6, w7, 0), va, 0);
        w0 += 4 * n_panel;
        w1 += 4 * n_panel;
        w2 += 4 * n_panel;
        w3 += 4 * n_panel;
        w4 += 4 * n_panel;
        w5 += 4 * n_panel;
        w6 += 4 * n_panel;
        w7 += 4 * n_panel;
      }

      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(0), p0, p1, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx + 0 * n_panel,
                            n);
      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(1), p2, p3, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx + 2 * n_panel,
                            n);
      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(2), p4, p5, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx + 4 * n_panel,
                            n);
      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(3), p6, p7, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx + 6 * n_panel,
                            n);
    }

    for (; n_idx + 2 * n_panel < n; n_idx += 4 * n_panel) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = p0 + panel_bytes;
      const uint8_t* p2 = p1 + panel_bytes;
      const uint8_t* p3 = (n_idx + 3 * n_panel < n) ? (p2 + panel_bytes) : p2;

      svwrite_za32_s32_vg1x4(0, xnn_pqs8_load_bias_x4(p0, p1));
      svwrite_za32_s32_vg1x4(1, xnn_pqs8_load_bias_x4(p2, p3));

      const int8_t* w0 = (const int8_t*)(p0 + w_off);
      const int8_t* w1 = (const int8_t*)(p1 + w_off);
      const int8_t* w2 = (const int8_t*)(p2 + w_off);
      const int8_t* w3 = (const int8_t*)(p3 + w_off);

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 6), va, 3);

        w0 += 16 * n_panel;
        w1 += 16 * n_panel;
        w2 += 16 * n_panel;
        w3 += 16 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, xnn_pqs8_load_w_2panels(w2, w3, 0), va, 0);
        w0 += 4 * n_panel;
        w1 += 4 * n_panel;
        w2 += 4 * n_panel;
        w3 += 4 * n_panel;
      }

      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(0), p0, p1, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx + 0 * n_panel,
                            n);
      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(1), p2, p3, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx + 2 * n_panel,
                            n);
    }

    if (n_idx + n_panel < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = p0 + panel_bytes;
      svwrite_za32_s32_vg1x4(0, xnn_pqs8_load_bias_x4(p0, p1));
      const int8_t* w0 = (const int8_t*)(p0 + w_off);
      const int8_t* w1 = (const int8_t*)(p1 + w_off);

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 6), va, 3);
        w0 += 16 * n_panel;
        w1 += 16 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_pqs8_load_w_2panels(w0, w1, 0), va, 0);
        w0 += 4 * n_panel;
        w1 += 4 * n_panel;
      }

      xnn_pqs8_epilogue_4vl(svread_za32_s32_vg1x4(0), p0, p1, scale_off,
                            vout_zp, vmin, vmax, dst_row, n_idx, n);
      n_idx += 2 * n_panel;
    }

    if (n_idx < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      svwrite_za32_s32_vg1x2(0, svld1_s32_x2(pc32, (const int32_t*)p0));
      const int8_t* w0 = (const int8_t*)(p0 + w_off);

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x2(0, svld1_vnum_s8_x2(pc8, w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x2(0, svld1_vnum_s8_x2(pc8, w0, 2), va, 1);
        svdot_lane_za32_s8_vg1x2(0, svld1_vnum_s8_x2(pc8, w0, 4), va, 2);
        svdot_lane_za32_s8_vg1x2(0, svld1_vnum_s8_x2(pc8, w0, 6), va, 3);
        w0 += 16 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x2(0, svld1_s8_x2(pc8, w0), va, 0);
        w0 += 4 * n_panel;
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
      svst1b_s16(svwhilelt_b16_u64(n_idx, n), dst_row + n_idx, vh01);
    }
  }
}

void xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_pqs8_qc8w_gemm_1x16vl_sme_dot_impl(
      m, n, k / sizeof(int8_t), (const int8_t*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->fp32_scalar.output_zero_point,
      minmax_params->fp32_scalar.output_min,
      minmax_params->fp32_scalar.output_max);
}
#else
#include <assert.h>

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_mr() {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2_get_nr() {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pqs8_qc8w_gemm_minmax_ukernel_1x32c4__neonsme2(
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
