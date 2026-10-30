// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
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

__arm_locally_streaming __arm_new(
    "za") static void xnn_pqs8_qc8w_gemm_1x16vl_sme_dot_impl(size_t m, size_t n,
                                                             size_t k_elements,
                                                             const int8_t* lhs,
                                                             const void*
                                                                 rhs_packed,
                                                             void* dst,
                                                             size_t
                                                                 dst_stride_row,
                                                             int32_t
                                                                 output_zero_point,
                                                             int32_t output_min,
                                                             int32_t
                                                                 output_max) {
  const size_t vl = svcntw();
  const size_t n_panel = 2 * vl;
  const size_t k_rounded = (k_elements + 3) & ~(size_t)3;
  const size_t panel_bytes =
      n_panel * (sizeof(int32_t) + k_rounded * sizeof(int8_t) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svbool_t p_row0 = svwhilelt_b8_u64(0, 4);
  const svbool_t p_row1 = sveor_b_z(pg8, svwhilelt_b8_u64(0, 8), p_row0);
  const svbool_t p_row2 =
      sveor_b_z(pg8, svwhilelt_b8_u64(0, 12), svwhilelt_b8_u64(0, 8));
  const svbool_t p_row3 =
      sveor_b_z(pg8, svwhilelt_b8_u64(0, 16), svwhilelt_b8_u64(0, 12));
  const svint32_t vout_zp = svdup_n_s32(output_zero_point);
  const svint32_t vmin = svdup_n_s32(output_min);
  const svint32_t vmax = svdup_n_s32(output_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* lhs_row = lhs + m_idx * k_elements;
    int8_t* dst_row = (int8_t*)((uintptr_t)dst + m_idx * dst_stride_row);

    for (size_t n_idx = 0; n_idx < n; n_idx += 2 * n_panel) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / n_panel) * panel_bytes;
      const uint8_t* p1 = (n_idx + n_panel < n) ? (p0 + panel_bytes) : p0;
      const int8_t* w0 = (const int8_t*)(p0 + n_panel * sizeof(int32_t));
      const int8_t* w1 = (const int8_t*)(p1 + n_panel * sizeof(int32_t));

      svzero_za();

      size_t k_idx = 0;
      for (; k_idx + 16 <= k_elements; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);

        svmopa_za32_s8_m(0, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 0));
        svmopa_za32_s8_m(1, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 1));
        svmopa_za32_s8_m(2, p_row0, pg8, va, svld1_vnum_s8(pg8, w1, 0));
        svmopa_za32_s8_m(3, p_row0, pg8, va, svld1_vnum_s8(pg8, w1, 1));

        svmopa_za32_s8_m(0, p_row1, pg8, va, svld1_vnum_s8(pg8, w0, 2));
        svmopa_za32_s8_m(1, p_row1, pg8, va, svld1_vnum_s8(pg8, w0, 3));
        svmopa_za32_s8_m(2, p_row1, pg8, va, svld1_vnum_s8(pg8, w1, 2));
        svmopa_za32_s8_m(3, p_row1, pg8, va, svld1_vnum_s8(pg8, w1, 3));

        svmopa_za32_s8_m(0, p_row2, pg8, va, svld1_vnum_s8(pg8, w0, 4));
        svmopa_za32_s8_m(1, p_row2, pg8, va, svld1_vnum_s8(pg8, w0, 5));
        svmopa_za32_s8_m(2, p_row2, pg8, va, svld1_vnum_s8(pg8, w1, 4));
        svmopa_za32_s8_m(3, p_row2, pg8, va, svld1_vnum_s8(pg8, w1, 5));

        svmopa_za32_s8_m(0, p_row3, pg8, va, svld1_vnum_s8(pg8, w0, 6));
        svmopa_za32_s8_m(1, p_row3, pg8, va, svld1_vnum_s8(pg8, w0, 7));
        svmopa_za32_s8_m(2, p_row3, pg8, va, svld1_vnum_s8(pg8, w1, 6));
        svmopa_za32_s8_m(3, p_row3, pg8, va, svld1_vnum_s8(pg8, w1, 7));

        w0 += 16 * n_panel;
        w1 += 16 * n_panel;
      }

      for (; k_idx < k_elements; k_idx += 4) {
        const svint8_t va =
            svld1rq_s8(svwhilelt_b8_u64(k_idx, k_elements), lhs_row + k_idx);
        svmopa_za32_s8_m(0, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 0));
        svmopa_za32_s8_m(1, p_row0, pg8, va, svld1_vnum_s8(pg8, w0, 1));
        svmopa_za32_s8_m(2, p_row0, pg8, va, svld1_vnum_s8(pg8, w1, 0));
        svmopa_za32_s8_m(3, p_row0, pg8, va, svld1_vnum_s8(pg8, w1, 1));
        w0 += 4 * n_panel;
        w1 += 4 * n_panel;
      }

      svint32_t vacc0 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 3)));
      svint32_t vacc1 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 3)));
      svint32_t vacc2 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 3)));
      svint32_t vacc3 = svadd_s32_x(
          pg32,
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 0),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 1)),
          svadd_s32_x(pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 2),
                      svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 3)));

      vacc0 =
          svadd_s32_x(pg32, vacc0, svld1_vnum_s32(pg32, (const int32_t*)p0, 0));
      vacc1 =
          svadd_s32_x(pg32, vacc1, svld1_vnum_s32(pg32, (const int32_t*)p0, 1));
      vacc2 =
          svadd_s32_x(pg32, vacc2, svld1_vnum_s32(pg32, (const int32_t*)p1, 0));
      vacc3 =
          svadd_s32_x(pg32, vacc3, svld1_vnum_s32(pg32, (const int32_t*)p1, 1));

      const size_t scale_off =
          n_panel * (sizeof(int32_t) + k_rounded * sizeof(int8_t));
      const float* s0 = (const float*)(p0 + scale_off);
      const float* s1 = (const float*)(p1 + scale_off);

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

      vq0 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq0, vout_zp), vmin), vmax);
      vq1 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq1, vout_zp), vmin), vmax);
      vq2 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq2, vout_zp), vmin), vmax);
      vq3 = svmin_s32_m(
          pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq3, vout_zp), vmin), vmax);

      const svint16_t vh01 =
          svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
      const svint16_t vh23 =
          svuzp1_s16(svreinterpret_s16_s32(vq2), svreinterpret_s16_s32(vq3));
      const svint8_t vb0123 =
          svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh23));
      svst1_s8(svwhilelt_b8_u64(0, n - n_idx), dst_row + n_idx, vb0123);
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
