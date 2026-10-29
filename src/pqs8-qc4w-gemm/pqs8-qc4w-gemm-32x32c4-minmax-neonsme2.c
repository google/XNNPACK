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

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa();
  return api.get_step(NULL).m;
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa();
  return api.get_step(NULL).n;
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
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
      kai_matmul_clamp_qai8_qai8p8vsx4_qsi4cxp8vsx4sf32bi32_8vsx8vs_sme2_mopa();
  api.run(NULL, &args);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
  return 2 * svcntsw();
}

static const int8_t xnn_qc4w_lut[64] = {
    0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0, 0, 3,  0, 0, 0,
    4,  0, 0, 0, 5,  0, 0, 0, 6,  0, 0, 0, 7,  0, 0, 0,
    -8, 0, 0, 0, -7, 0, 0, 0, -6, 0, 0, 0, -5, 0, 0, 0,
    -4, 0, 0, 0, -3, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0};

static XNN_INLINE svint32x4_t xnn_pqs8_qc4w_requant_tile_4rows(
    svint32x4_t vacc, svfloat32_t vscale, svint32_t vout_zp, svint32_t vmin,
    svint32_t vmax) __arm_streaming __arm_preserves("za", "zt0") {
  const svbool_t pg32 = svptrue_b32();
  const svfloat32x4_t vf = svcvt_f32_s32_x4(vacc);
  const svfloat32x4_t vscaled =
      svcreate4_f32(svmul_f32_x(pg32, svget4_f32(vf, 0), vscale),
                    svmul_f32_x(pg32, svget4_f32(vf, 1), vscale),
                    svmul_f32_x(pg32, svget4_f32(vf, 2), vscale),
                    svmul_f32_x(pg32, svget4_f32(vf, 3), vscale));
  const svint32x4_t vq = svcvt_s32_f32_x4(svrintn_f32_x4(vscaled));
  return svclamp_single_s32_x4(
      svcreate4_s32(svqadd_s32(svget4_s32(vq, 0), vout_zp),
                    svqadd_s32(svget4_s32(vq, 1), vout_zp),
                    svqadd_s32(svget4_s32(vq, 2), vout_zp),
                    svqadd_s32(svget4_s32(vq, 3), vout_zp)),
      vmin, vmax);
}

static XNN_INLINE void xnn_pqs8_qc4w_store_2vl_4rows(
    svint32x4_t za_rows0, svint32x4_t za_rows1, svfloat32_t vscale0,
    svfloat32_t vscale1, svint32_t vout_zp, svint32_t vmin, svint32_t vmax,
    svbool_t p_out16, int8_t* dst_base, size_t dst_stride_row) __arm_streaming
    __arm_preserves("za", "zt0") {
  const svint32x4_t vq0 =
      xnn_pqs8_qc4w_requant_tile_4rows(za_rows0, vscale0, vout_zp, vmin, vmax);
  const svint32x4_t vq1 =
      xnn_pqs8_qc4w_requant_tile_4rows(za_rows1, vscale1, vout_zp, vmin, vmax);

  const svint16_t vh0 = svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vq0, 0)),
                                   svreinterpret_s16_s32(svget4_s32(vq1, 0)));
  const svint16_t vh1 = svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vq0, 1)),
                                   svreinterpret_s16_s32(svget4_s32(vq1, 1)));
  const svint16_t vh2 = svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vq0, 2)),
                                   svreinterpret_s16_s32(svget4_s32(vq1, 2)));
  const svint16_t vh3 = svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vq0, 3)),
                                   svreinterpret_s16_s32(svget4_s32(vq1, 3)));

  svst1b_s16(p_out16, dst_base, vh0);
  svst1b_s16(p_out16, dst_base + dst_stride_row, vh1);
  svst1b_s16(p_out16, dst_base + 2 * dst_stride_row, vh2);
  svst1b_s16(p_out16, dst_base + 3 * dst_stride_row, vh3);
}

__arm_locally_streaming __arm_new("za", "zt0") static XNN_NOINLINE void
xnn_pqs8_qc4w_gemm_8vsx8vs_sme_mopa_impl(
    size_t m, size_t n, size_t k_elements, const int8_t* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    int32_t output_zero_point, int32_t output_min, int32_t output_max) {
  const size_t vl = svcntw();
  const size_t m_panel = 2 * vl;
  const size_t n_panel = 2 * vl;
  const size_t k_padded = (k_elements + 3) & ~(size_t)3;
  const size_t rounded_k = (k_elements + 31) & ~(size_t)31;
  const size_t lhs_panel_bytes = m_panel * k_padded;
  const size_t rhs_panel_bytes =
      n_panel * (sizeof(int32_t) + (rounded_k / 2) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc8 = svptrue_c8();
  const svint32_t vout_zp = svdup_n_s32(output_zero_point);
  const svint32_t vmin = svdup_n_s32(output_min);
  const svint32_t vmax = svdup_n_s32(output_max);
  svldr_zt(0, xnn_qc4w_lut);

  for (size_t m_idx = 0; m_idx < m; m_idx += m_panel) {
    const size_t m_rem = m - m_idx < m_panel ? m - m_idx : m_panel;
    const int8_t* lhs_p = lhs_packed + (m_idx / m_panel) * lhs_panel_bytes;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_panel) {
      if (m_rem <= vl && n_idx + n_panel < n) {
        const size_t n_rem4 = n - n_idx;
        const uint8_t* rhs_p0 =
            (const uint8_t*)rhs_packed + (n_idx / n_panel) * rhs_panel_bytes;
        const uint8_t* rhs_p1 = rhs_p0 + rhs_panel_bytes;
        const int32_t* bias_ptr0 = (const int32_t*)rhs_p0;
        const int32_t* bias_ptr1 = (const int32_t*)rhs_p1;
        const uint8_t* b0_ptr = rhs_p0 + n_panel * sizeof(int32_t);
        const uint8_t* b1_ptr = rhs_p1 + n_panel * sizeof(int32_t);
        const float* scale_ptr0 =
            (const float*)(rhs_p0 +
                           n_panel * (sizeof(int32_t) + (rounded_k / 2)));
        const float* scale_ptr1 =
            (const float*)(rhs_p1 +
                           n_panel * (sizeof(int32_t) + (rounded_k / 2)));

        svzero_za();
        svaddha_za32_s32_m(0, pg32, pg32, svld1_vnum_s32(pg32, bias_ptr0, 0));
        svaddha_za32_s32_m(1, pg32, pg32, svld1_vnum_s32(pg32, bias_ptr0, 1));
        svaddha_za32_s32_m(2, pg32, pg32, svld1_vnum_s32(pg32, bias_ptr1, 0));
        svaddha_za32_s32_m(3, pg32, pg32, svld1_vnum_s32(pg32, bias_ptr1, 1));

        const int8_t* a_ptr = lhs_p;
        size_t k_idx = 0;
        for (; k_idx + 16 <= k_padded; k_idx += 16) {
          const svuint8x4_t vr0 = svld1_u8_x4(pc8, b0_ptr);
          const svuint8x4_t vr1 = svld1_u8_x4(pc8, b1_ptr);
#define XNN_STEP(step)                                                      \
  do {                                                                      \
    const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, (step) * 2);             \
    const svint8x2_t vb0 =                                                  \
        svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, (step)), 0);                \
    const svint8x2_t vb1 =                                                  \
        svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, (step)), 0);                \
    svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb0, 0));                  \
    svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb0, 1));                  \
    svmopa_za32_s8_m(2, pg8, pg8, va0, svget2_s8(vb1, 0));                  \
    svmopa_za32_s8_m(3, pg8, pg8, va0, svget2_s8(vb1, 1));                  \
  } while (0)
          XNN_STEP(0);
          XNN_STEP(1);
          XNN_STEP(2);
          XNN_STEP(3);
#undef XNN_STEP
          a_ptr += 16 * m_panel;
          b0_ptr += 8 * n_panel;
          b1_ptr += 8 * n_panel;
        }
        for (; k_idx < k_padded; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8x2_t vb0 =
              svluti4_lane_zt_s8_x2(0, svld1_u8(pg8, b0_ptr), 0);
          const svint8x2_t vb1 =
              svluti4_lane_zt_s8_x2(0, svld1_u8(pg8, b1_ptr), 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb0, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb0, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svget2_s8(vb1, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svget2_s8(vb1, 1));
          a_ptr += 4 * m_panel;
          b0_ptr += 2 * n_panel;
          b1_ptr += 2 * n_panel;
        }

        const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr0, 0);
        const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr0, 1);
        const svfloat32_t vscale2 = svld1_vnum_f32(pg32, scale_ptr1, 0);
        const svfloat32_t vscale3 = svld1_vnum_f32(pg32, scale_ptr1, 1);
        const svbool_t p_out8 = svwhilelt_b8_u64(0, n_rem4);

        size_t row = 0;
        for (; row + 4 <= m_rem; row += 4) {
          const svint32x4_t vq0 = xnn_pqs8_qc4w_requant_tile_4rows(
              svread_hor_za32_s32_vg4(0, (uint32_t)row), vscale0, vout_zp, vmin,
              vmax);
          const svint32x4_t vq1 = xnn_pqs8_qc4w_requant_tile_4rows(
              svread_hor_za32_s32_vg4(1, (uint32_t)row), vscale1, vout_zp, vmin,
              vmax);
          const svint32x4_t vq2 = xnn_pqs8_qc4w_requant_tile_4rows(
              svread_hor_za32_s32_vg4(2, (uint32_t)row), vscale2, vout_zp, vmin,
              vmax);
          const svint32x4_t vq3 = xnn_pqs8_qc4w_requant_tile_4rows(
              svread_hor_za32_s32_vg4(3, (uint32_t)row), vscale3, vout_zp, vmin,
              vmax);
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) +
              n_idx;
#define XNN_PQS8_QC4W_STORE_4VL_ROW(R)                                         \
  do {                                                                         \
    const svint16_t vh01 =                                                     \
        svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vq0, (R))),                \
                   svreinterpret_s16_s32(svget4_s32(vq1, (R))));               \
    const svint16_t vh23 =                                                     \
        svuzp1_s16(svreinterpret_s16_s32(svget4_s32(vq2, (R))),                \
                   svreinterpret_s16_s32(svget4_s32(vq3, (R))));               \
    const svint8_t vb =                                                        \
        svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh23));     \
    svst1_s8(p_out8, dst_ptr + (R) * dst_stride_row, vb);                      \
  } while (0)
          XNN_PQS8_QC4W_STORE_4VL_ROW(0);
          XNN_PQS8_QC4W_STORE_4VL_ROW(1);
          XNN_PQS8_QC4W_STORE_4VL_ROW(2);
          XNN_PQS8_QC4W_STORE_4VL_ROW(3);
#undef XNN_PQS8_QC4W_STORE_4VL_ROW
        }
        for (; row < m_rem; ++row) {
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) +
              n_idx;
          const svfloat32x4_t vf = svcvt_f32_s32_x4(svcreate4_s32(
              svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row),
              svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row),
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row),
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row)));
          const svint32x4_t vq = svcvt_s32_f32_x4(svrintn_f32_x4(svcreate4_f32(
              svmul_f32_x(pg32, svget4_f32(vf, 0), vscale0),
              svmul_f32_x(pg32, svget4_f32(vf, 1), vscale1),
              svmul_f32_x(pg32, svget4_f32(vf, 2), vscale2),
              svmul_f32_x(pg32, svget4_f32(vf, 3), vscale3))));
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
          svst1_s8(p_out8, dst_ptr, vb0123);
        }
        n_idx += n_panel;
        continue;
      }

      const size_t n_rem = n - n_idx < n_panel ? n - n_idx : n_panel;
      const uint8_t* rhs_p =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * rhs_panel_bytes;
      const int32_t* bias_ptr = (const int32_t*)rhs_p;
      const uint8_t* b_ptr = rhs_p + n_panel * sizeof(int32_t);
      const float* scale_ptr =
          (const float*)(rhs_p + n_panel * (sizeof(int32_t) + (rounded_k / 2)));

      const svint32_t vbias0 = svld1_vnum_s32(pg32, bias_ptr, 0);
      const svint32_t vbias1 = svld1_vnum_s32(pg32, bias_ptr, 1);
      svzero_za();
      svaddha_za32_s32_m(0, pg32, pg32, vbias0);
      svaddha_za32_s32_m(1, pg32, pg32, vbias1);

      const int8_t* a_ptr = lhs_p;
      size_t k_idx = 0;
      if (m_rem <= vl) {
        for (; k_idx + 16 <= k_padded; k_idx += 16) {
          const svuint8x4_t vr = svld1_u8_x4(pc8, b_ptr);
#define XNN_STEP_1VL(step)                                                 \
  do {                                                                     \
    const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, (step) * 2);            \
    const svint8x2_t vb =                                                  \
        svluti4_lane_zt_s8_x2(0, svget4_u8(vr, (step)), 0);                \
    svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb, 0));                  \
    svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb, 1));                  \
  } while (0)
          XNN_STEP_1VL(0);
          XNN_STEP_1VL(1);
          XNN_STEP_1VL(2);
          XNN_STEP_1VL(3);
#undef XNN_STEP_1VL
          a_ptr += 16 * m_panel;
          b_ptr += 8 * n_panel;
        }
        for (; k_idx < k_padded; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8x2_t vb =
              svluti4_lane_zt_s8_x2(0, svld1_u8(pg8, b_ptr), 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb, 1));
          a_ptr += 4 * m_panel;
          b_ptr += 2 * n_panel;
        }
      } else {
        svaddha_za32_s32_m(2, pg32, pg32, vbias0);
        svaddha_za32_s32_m(3, pg32, pg32, vbias1);
        for (; k_idx + 16 <= k_padded; k_idx += 16) {
          const svuint8x4_t vr = svld1_u8_x4(pc8, b_ptr);
#define XNN_STEP(step)                                                     \
  do {                                                                     \
    const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, (step) * 2);            \
    const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, (step) * 2 + 1);        \
    const svint8x2_t vb =                                                  \
        svluti4_lane_zt_s8_x2(0, svget4_u8(vr, (step)), 0);                \
    svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb, 0));                  \
    svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb, 1));                  \
    svmopa_za32_s8_m(2, pg8, pg8, va1, svget2_s8(vb, 0));                  \
    svmopa_za32_s8_m(3, pg8, pg8, va1, svget2_s8(vb, 1));                  \
  } while (0)
          XNN_STEP(0);
          XNN_STEP(1);
          XNN_STEP(2);
          XNN_STEP(3);
#undef XNN_STEP
          a_ptr += 16 * m_panel;
          b_ptr += 8 * n_panel;
        }
        for (; k_idx < k_padded; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 1);
          const svint8x2_t vb =
              svluti4_lane_zt_s8_x2(0, svld1_u8(pg8, b_ptr), 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svget2_s8(vb, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svget2_s8(vb, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va1, svget2_s8(vb, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va1, svget2_s8(vb, 1));
          a_ptr += 4 * m_panel;
          b_ptr += 2 * n_panel;
        }
      }

      const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr, 0);
      const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr, 1);
      const svbool_t p_out16 = svwhilelt_b16_u64(0, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      size_t row = 0;
      for (; row + 4 <= rows0; row += 4) {
        int8_t* dst_ptr =
            (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
        xnn_pqs8_qc4w_store_2vl_4rows(
            svread_hor_za32_s32_vg4(0, (uint32_t)row),
            svread_hor_za32_s32_vg4(1, (uint32_t)row), vscale0, vscale1,
            vout_zp, vmin, vmax, p_out16, dst_ptr, dst_stride_row);
      }
      for (; row < rows0; ++row) {
        int8_t* dst_ptr =
            (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
        const svint32_t vacc0 =
            svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
        const svint32_t vacc1 =
            svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
        const svfloat32_t vf0 =
            svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
        const svfloat32_t vf1 =
            svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
        svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
        svint32_t vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
        vq0 = svmin_s32_x(
            pg32, svmax_s32_x(pg32, svqadd_s32(vq0, vout_zp), vmin), vmax);
        vq1 = svmin_s32_x(
            pg32, svmax_s32_x(pg32, svqadd_s32(vq1, vout_zp), vmin), vmax);
        const svint16_t vh01 =
            svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
        svst1b_s16(p_out16, dst_ptr, vh01);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        size_t row1 = 0;
        for (; row1 + 4 <= rows1; row1 += 4) {
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + vl + row1) * dst_stride_row) +
              n_idx;
          xnn_pqs8_qc4w_store_2vl_4rows(
              svread_hor_za32_s32_vg4(2, (uint32_t)row1),
              svread_hor_za32_s32_vg4(3, (uint32_t)row1), vscale0, vscale1,
              vout_zp, vmin, vmax, p_out16, dst_ptr, dst_stride_row);
        }
        for (; row1 < rows1; ++row1) {
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + vl + row1) * dst_stride_row) +
              n_idx;
          const svint32_t vacc0 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row1);
          const svint32_t vacc1 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row1);
          const svfloat32_t vf0 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
          const svfloat32_t vf1 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
          svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
          svint32_t vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
          vq0 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svqadd_s32(vq0, vout_zp), vmin), vmax);
          vq1 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svqadd_s32(vq1, vout_zp), vmin), vmax);
          const svint16_t vh01 = svuzp1_s16(svreinterpret_s16_s32(vq0),
                                            svreinterpret_s16_s32(vq1));
          svst1b_s16(p_out16, dst_ptr, vh01);
        }
      }
    }
  }
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  assert(dst_stride_col == sizeof(int8_t));
  (void)dst_stride_col;
  xnn_pqs8_qc4w_gemm_8vsx8vs_sme_mopa_impl(
      m, n, k / sizeof(int8_t), (const int8_t*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->fp32_scalar.output_zero_point,
      minmax_params->fp32_scalar.output_min,
      minmax_params->fp32_scalar.output_max);
}
#else
size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_mr(void) {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2_get_nr(void) {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pqs8_qc4w_gemm_minmax_fp32_ukernel_32x32c4__neonsme2(
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
