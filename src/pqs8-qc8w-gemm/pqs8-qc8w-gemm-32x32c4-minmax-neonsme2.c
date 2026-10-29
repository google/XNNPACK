// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa.h"

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa();
}

// Wraps the
// `kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa` GEMM
// microkernel with a name that is compatible with our tooling.
void xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = minmax_params->fp32_scalar.output_zero_point;
  kai_params.min_value = minmax_params->fp32_scalar.output_min;
  kai_params.max_value = minmax_params->fp32_scalar.output_max;

  kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxpsb2vlx4_2vlx2vl_sme2_mopa(
      m, n, k / sizeof(int8_t), lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(int8_t), &kai_params);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <assert.h>
#include <stdint.h>

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pqs8_qc8w_gemm_2vlx2vl_sme2_mopa_impl(size_t m,
                                                                size_t n,
                                                                size_t
                                                                    k_elements,
                                                                const int8_t*
                                                                    lhs_packed,
                                                                const void*
                                                                    rhs_packed,
                                                                void* dst,
                                                                size_t
                                                                    dst_stride_row,
                                                                int32_t
                                                                    output_zero_point,
                                                                int32_t
                                                                    output_min,
                                                                int32_t
                                                                    output_max) {
  const size_t vl = svcntw();
  const size_t m_panel = 2 * vl;
  const size_t n_panel = 2 * vl;
  const size_t k_rounded = (k_elements + 3) & ~(size_t)3;
  const size_t lhs_panel_bytes = m_panel * k_rounded;
  const size_t rhs_panel_bytes =
      n_panel * (sizeof(int32_t) + k_rounded * sizeof(int8_t) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svint32_t vout_zp = svdup_n_s32(output_zero_point);
  const svint32_t vmin = svdup_n_s32(output_min);
  const svint32_t vmax = svdup_n_s32(output_max);

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
        const int8_t* b0_ptr =
            (const int8_t*)(rhs_p0 + n_panel * sizeof(int32_t));
        const int8_t* b1_ptr =
            (const int8_t*)(rhs_p1 + n_panel * sizeof(int32_t));
        const float* scale_ptr0 = (const float*)(b0_ptr + n_panel * k_rounded);
        const float* scale_ptr1 = (const float*)(b1_ptr + n_panel * k_rounded);

        const svint32_t vbias0 = svld1_vnum_s32(pg32, bias_ptr0, 0);
        const svint32_t vbias1 = svld1_vnum_s32(pg32, bias_ptr0, 1);
        const svint32_t vbias2 = svld1_vnum_s32(pg32, bias_ptr1, 0);
        const svint32_t vbias3 = svld1_vnum_s32(pg32, bias_ptr1, 1);
        svzero_za();
        svaddha_za32_s32_m(0, pg32, pg32, vbias0);
        svaddha_za32_s32_m(1, pg32, pg32, vbias1);
        svaddha_za32_s32_m(2, pg32, pg32, vbias2);
        svaddha_za32_s32_m(3, pg32, pg32, vbias3);

        const int8_t* a_ptr = lhs_p;
        size_t k_idx = 0;
        for (; k_idx + 16 <= k_rounded; k_idx += 16) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 2);
          const svint8_t va2 = svld1_vnum_s8(pg8, a_ptr, 4);
          const svint8_t va3 = svld1_vnum_s8(pg8, a_ptr, 6);

          svmopa_za32_s8_m(0, pg8, pg8, va0, svld1_vnum_s8(pg8, b0_ptr, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svld1_vnum_s8(pg8, b0_ptr, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svld1_vnum_s8(pg8, b1_ptr, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svld1_vnum_s8(pg8, b1_ptr, 1));

          svmopa_za32_s8_m(0, pg8, pg8, va1, svld1_vnum_s8(pg8, b0_ptr, 2));
          svmopa_za32_s8_m(1, pg8, pg8, va1, svld1_vnum_s8(pg8, b0_ptr, 3));
          svmopa_za32_s8_m(2, pg8, pg8, va1, svld1_vnum_s8(pg8, b1_ptr, 2));
          svmopa_za32_s8_m(3, pg8, pg8, va1, svld1_vnum_s8(pg8, b1_ptr, 3));

          svmopa_za32_s8_m(0, pg8, pg8, va2, svld1_vnum_s8(pg8, b0_ptr, 4));
          svmopa_za32_s8_m(1, pg8, pg8, va2, svld1_vnum_s8(pg8, b0_ptr, 5));
          svmopa_za32_s8_m(2, pg8, pg8, va2, svld1_vnum_s8(pg8, b1_ptr, 4));
          svmopa_za32_s8_m(3, pg8, pg8, va2, svld1_vnum_s8(pg8, b1_ptr, 5));

          svmopa_za32_s8_m(0, pg8, pg8, va3, svld1_vnum_s8(pg8, b0_ptr, 6));
          svmopa_za32_s8_m(1, pg8, pg8, va3, svld1_vnum_s8(pg8, b0_ptr, 7));
          svmopa_za32_s8_m(2, pg8, pg8, va3, svld1_vnum_s8(pg8, b1_ptr, 6));
          svmopa_za32_s8_m(3, pg8, pg8, va3, svld1_vnum_s8(pg8, b1_ptr, 7));

          a_ptr += 16 * m_panel;
          b0_ptr += 16 * n_panel;
          b1_ptr += 16 * n_panel;
        }
        for (; k_idx < k_rounded; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          svmopa_za32_s8_m(0, pg8, pg8, va0, svld1_vnum_s8(pg8, b0_ptr, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svld1_vnum_s8(pg8, b0_ptr, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svld1_vnum_s8(pg8, b1_ptr, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svld1_vnum_s8(pg8, b1_ptr, 1));
          a_ptr += 4 * m_panel;
          b0_ptr += 4 * n_panel;
          b1_ptr += 4 * n_panel;
        }

        const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr0, 0);
        const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr0, 1);
        const svfloat32_t vscale2 = svld1_vnum_f32(pg32, scale_ptr1, 0);
        const svfloat32_t vscale3 = svld1_vnum_f32(pg32, scale_ptr1, 1);
        const svbool_t p_out8 = svwhilelt_b8_u64(0, n_rem4);

        for (size_t row = 0; row < m_rem; ++row) {
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) +
              n_idx;
          const svint32_t vacc0 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
          const svint32_t vacc1 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
          const svint32_t vacc2 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row);
          const svint32_t vacc3 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row);
          const svfloat32_t vf0 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
          const svfloat32_t vf1 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
          const svfloat32_t vf2 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc2), vscale2);
          const svfloat32_t vf3 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc3), vscale3);
          svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
          svint32_t vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
          svint32_t vq2 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf2));
          svint32_t vq3 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf3));
          vq0 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq0, vout_zp), vmin),
              vmax);
          vq1 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq1, vout_zp), vmin),
              vmax);
          vq2 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq2, vout_zp), vmin),
              vmax);
          vq3 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq3, vout_zp), vmin),
              vmax);
          const svint16_t vh01 = svuzp1_s16(svreinterpret_s16_s32(vq0),
                                            svreinterpret_s16_s32(vq1));
          const svint16_t vh23 = svuzp1_s16(svreinterpret_s16_s32(vq2),
                                            svreinterpret_s16_s32(vq3));
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
      const int8_t* b_ptr = (const int8_t*)(rhs_p + n_panel * sizeof(int32_t));
      const float* scale_ptr = (const float*)(b_ptr + n_panel * k_rounded);

      const svint32_t vbias0 = svld1_vnum_s32(pg32, bias_ptr, 0);
      const svint32_t vbias1 = svld1_vnum_s32(pg32, bias_ptr, 1);
      svzero_za();
      svaddha_za32_s32_m(0, pg32, pg32, vbias0);
      svaddha_za32_s32_m(1, pg32, pg32, vbias1);
      svaddha_za32_s32_m(2, pg32, pg32, vbias0);
      svaddha_za32_s32_m(3, pg32, pg32, vbias1);

      const int8_t* a_ptr = lhs_p;
      size_t k_idx = 0;
      for (; k_idx + 16 <= k_rounded; k_idx += 16) {
        const svint8_t va00 = svld1_vnum_s8(pg8, a_ptr, 0);
        const svint8_t va01 = svld1_vnum_s8(pg8, a_ptr, 1);
        const svint8_t vb00 = svld1_vnum_s8(pg8, b_ptr, 0);
        const svint8_t vb01 = svld1_vnum_s8(pg8, b_ptr, 1);
        svmopa_za32_s8_m(0, pg8, pg8, va00, vb00);
        svmopa_za32_s8_m(1, pg8, pg8, va00, vb01);
        svmopa_za32_s8_m(2, pg8, pg8, va01, vb00);
        svmopa_za32_s8_m(3, pg8, pg8, va01, vb01);

        const svint8_t va10 = svld1_vnum_s8(pg8, a_ptr, 2);
        const svint8_t va11 = svld1_vnum_s8(pg8, a_ptr, 3);
        const svint8_t vb10 = svld1_vnum_s8(pg8, b_ptr, 2);
        const svint8_t vb11 = svld1_vnum_s8(pg8, b_ptr, 3);
        svmopa_za32_s8_m(0, pg8, pg8, va10, vb10);
        svmopa_za32_s8_m(1, pg8, pg8, va10, vb11);
        svmopa_za32_s8_m(2, pg8, pg8, va11, vb10);
        svmopa_za32_s8_m(3, pg8, pg8, va11, vb11);

        const svint8_t va20 = svld1_vnum_s8(pg8, a_ptr, 4);
        const svint8_t va21 = svld1_vnum_s8(pg8, a_ptr, 5);
        const svint8_t vb20 = svld1_vnum_s8(pg8, b_ptr, 4);
        const svint8_t vb21 = svld1_vnum_s8(pg8, b_ptr, 5);
        svmopa_za32_s8_m(0, pg8, pg8, va20, vb20);
        svmopa_za32_s8_m(1, pg8, pg8, va20, vb21);
        svmopa_za32_s8_m(2, pg8, pg8, va21, vb20);
        svmopa_za32_s8_m(3, pg8, pg8, va21, vb21);

        const svint8_t va30 = svld1_vnum_s8(pg8, a_ptr, 6);
        const svint8_t va31 = svld1_vnum_s8(pg8, a_ptr, 7);
        const svint8_t vb30 = svld1_vnum_s8(pg8, b_ptr, 6);
        const svint8_t vb31 = svld1_vnum_s8(pg8, b_ptr, 7);
        svmopa_za32_s8_m(0, pg8, pg8, va30, vb30);
        svmopa_za32_s8_m(1, pg8, pg8, va30, vb31);
        svmopa_za32_s8_m(2, pg8, pg8, va31, vb30);
        svmopa_za32_s8_m(3, pg8, pg8, va31, vb31);

        a_ptr += 16 * m_panel;
        b_ptr += 16 * n_panel;
      }
      for (; k_idx < k_rounded; k_idx += 4) {
        const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
        const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 1);
        const svint8_t vb0 = svld1_vnum_s8(pg8, b_ptr, 0);
        const svint8_t vb1 = svld1_vnum_s8(pg8, b_ptr, 1);
        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
        svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
        a_ptr += 4 * m_panel;
        b_ptr += 4 * n_panel;
      }

      const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr, 0);
      const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr, 1);
      const svbool_t p_out8 = svwhilelt_b8_u64(0, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
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
            pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq0, vout_zp), vmin),
            vmax);
        vq1 = svmin_s32_x(
            pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq1, vout_zp), vmin),
            vmax);
        const svint16_t vh01 =
            svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
        const svint8_t vb01 =
            svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh01));
        svst1_s8(p_out8, dst_ptr, vb01);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row = 0; row < rows1; ++row) {
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + vl + row) * dst_stride_row) +
              n_idx;
          const svint32_t vacc0 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row);
          const svint32_t vacc1 =
              svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row);
          const svfloat32_t vf0 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
          const svfloat32_t vf1 =
              svmul_f32_x(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
          svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
          svint32_t vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
          vq0 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq0, vout_zp), vmin),
              vmax);
          vq1 = svmin_s32_x(
              pg32, svmax_s32_x(pg32, svadd_s32_x(pg32, vq1, vout_zp), vmin),
              vmax);
          const svint16_t vh01 = svuzp1_s16(svreinterpret_s16_s32(vq0),
                                            svreinterpret_s16_s32(vq1));
          const svint8_t vb01 =
              svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh01));
          svst1_s8(p_out8, dst_ptr, vb01);
        }
      }
    }
  }
}

void xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col,
    const union xnn_qs8_qc8w_conv_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_pqs8_qc8w_gemm_2vlx2vl_sme2_mopa_impl(
      m, n, k / sizeof(int8_t), (const int8_t*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->fp32_scalar.output_zero_point,
      minmax_params->fp32_scalar.output_min,
      minmax_params->fp32_scalar.output_max);
}
#else
#include <assert.h>

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_mr() {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2_get_nr() {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pqs8_qc8w_gemm_minmax_ukernel_32x32c4__neonsme2(
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
