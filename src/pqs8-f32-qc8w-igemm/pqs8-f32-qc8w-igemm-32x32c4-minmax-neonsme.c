// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_qai8_qai8p_qsi8cxp/kai_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa.h"
#include "kai/ukernels/matmul/pack/kai_lhs_imatmul_pack_x8p2vlx4_x8p_sme.h"

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme_get_mr(void) {
  return kai_get_mr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa();
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme_get_nr(void) {
  return kai_get_nr_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa();
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  const size_t kai_kr = 4;
  const size_t k = ks * round_up(kc, kai_kr);

  // Repackage the params.
  struct kai_matmul_requantize32_params kai_params;
  kai_params.output_zero_point = params->fp32_scalar.output_zero_point;
  kai_params.min_value = (int8_t)params->fp32_scalar.output_min;
  kai_params.max_value = (int8_t)params->fp32_scalar.output_max;

  kai_run_matmul_clamp_qai8_qai8p2vlx4_qsi8cxp2vlx4sb_2vlx2vl_sme_mopa(
      mr, nc, k, packed_lhs, w, c, cm_stride, sizeof(int8_t), &kai_params);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <assert.h>
#include <stdint.h>

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme_get_mr(void) {
  return 2 * svcntsw();
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme_get_nr(void) {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pqs8_qc8w_igemm_2vlx2vl_sme_mopa_impl(size_t m,
                                                                size_t n,
                                                                size_t
                                                                    k_rounded,
                                                                const int8_t*
                                                                    lhs_packed,
                                                                const void*
                                                                    rhs_packed,
                                                                int8_t* dst,
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
        const int8_t* rhs_w0 =
            (const int8_t*)(rhs_p0 + n_panel * sizeof(int32_t));
        const int8_t* rhs_w1 =
            (const int8_t*)(rhs_p1 + n_panel * sizeof(int32_t));
        const float* scale_ptr0 =
            (const float*)(rhs_p0 + n_panel * (sizeof(int32_t) +
                                               k_rounded * sizeof(int8_t)));
        const float* scale_ptr1 =
            (const float*)(rhs_p1 + n_panel * (sizeof(int32_t) +
                                               k_rounded * sizeof(int8_t)));
        svzero_za();

        for (size_t k_idx = 0; k_idx < k_rounded; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, lhs_p + k_idx * m_panel, 0);
          const svint8_t vb0 = svld1_vnum_s8(pg8, rhs_w0 + k_idx * n_panel, 0);
          const svint8_t vb1 = svld1_vnum_s8(pg8, rhs_w0 + k_idx * n_panel, 1);
          const svint8_t vb2 = svld1_vnum_s8(pg8, rhs_w1 + k_idx * n_panel, 0);
          const svint8_t vb3 = svld1_vnum_s8(pg8, rhs_w1 + k_idx * n_panel, 1);

          svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
          svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
          svmopa_za32_s8_m(2, pg8, pg8, va0, vb2);
          svmopa_za32_s8_m(3, pg8, pg8, va0, vb3);
        }

        const svint32_t vbias0 = svld1_vnum_s32(pg32, bias_ptr0, 0);
        const svint32_t vbias1 = svld1_vnum_s32(pg32, bias_ptr0, 1);
        const svint32_t vbias2 = svld1_vnum_s32(pg32, bias_ptr1, 0);
        const svint32_t vbias3 = svld1_vnum_s32(pg32, bias_ptr1, 1);
        const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr0, 0);
        const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr0, 1);
        const svfloat32_t vscale2 = svld1_vnum_f32(pg32, scale_ptr1, 0);
        const svfloat32_t vscale3 = svld1_vnum_f32(pg32, scale_ptr1, 1);
        const svbool_t p_out8 = svwhilelt_b8_u64(0, n_rem4);

        for (size_t row = 0; row < m_rem; ++row) {
          int8_t* dst_ptr =
              (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) +
              n_idx;
          svint32_t vacc0 = svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
          svint32_t vacc1 = svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
          svint32_t vacc2 = svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row);
          svint32_t vacc3 = svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row);
          vacc0 = svadd_s32_m(pg32, vacc0, vbias0);
          vacc1 = svadd_s32_m(pg32, vacc1, vbias1);
          vacc2 = svadd_s32_m(pg32, vacc2, vbias2);
          vacc3 = svadd_s32_m(pg32, vacc3, vbias3);
          const svfloat32_t vf0 =
              svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
          const svfloat32_t vf1 =
              svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
          const svfloat32_t vf2 =
              svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc2), vscale2);
          const svfloat32_t vf3 =
              svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc3), vscale3);
          svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
          svint32_t vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
          svint32_t vq2 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf2));
          svint32_t vq3 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf3));
          vq0 = svmin_s32_m(
              pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq0, vout_zp), vmin),
              vmax);
          vq1 = svmin_s32_m(
              pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq1, vout_zp), vmin),
              vmax);
          vq2 = svmin_s32_m(
              pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq2, vout_zp), vmin),
              vmax);
          vq3 = svmin_s32_m(
              pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq3, vout_zp), vmin),
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
      const int8_t* rhs_w = (const int8_t*)(rhs_p + n_panel * sizeof(int32_t));
      const float* scale_ptr =
          (const float*)(rhs_p + n_panel * (sizeof(int32_t) +
                                            k_rounded * sizeof(int8_t)));
      svzero_za();

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_rounded; k_idx += 8) {
        const svint8_t va0_0 = svld1_vnum_s8(pg8, lhs_p + k_idx * m_panel, 0);
        const svint8_t va1_0 = svld1_vnum_s8(pg8, lhs_p + k_idx * m_panel, 1);
        const svint8_t vb0_0 = svld1_vnum_s8(pg8, rhs_w + k_idx * n_panel, 0);
        const svint8_t vb1_0 = svld1_vnum_s8(pg8, rhs_w + k_idx * n_panel, 1);
        const svint8_t va0_1 =
            svld1_vnum_s8(pg8, lhs_p + (k_idx + 4) * m_panel, 0);
        const svint8_t va1_1 =
            svld1_vnum_s8(pg8, lhs_p + (k_idx + 4) * m_panel, 1);
        const svint8_t vb0_1 =
            svld1_vnum_s8(pg8, rhs_w + (k_idx + 4) * n_panel, 0);
        const svint8_t vb1_1 =
            svld1_vnum_s8(pg8, rhs_w + (k_idx + 4) * n_panel, 1);

        svmopa_za32_s8_m(0, pg8, pg8, va0_0, vb0_0);
        svmopa_za32_s8_m(1, pg8, pg8, va0_0, vb1_0);
        svmopa_za32_s8_m(2, pg8, pg8, va1_0, vb0_0);
        svmopa_za32_s8_m(3, pg8, pg8, va1_0, vb1_0);
        svmopa_za32_s8_m(0, pg8, pg8, va0_1, vb0_1);
        svmopa_za32_s8_m(1, pg8, pg8, va0_1, vb1_1);
        svmopa_za32_s8_m(2, pg8, pg8, va1_1, vb0_1);
        svmopa_za32_s8_m(3, pg8, pg8, va1_1, vb1_1);
      }
      for (; k_idx < k_rounded; k_idx += 4) {
        const svint8_t va0 = svld1_vnum_s8(pg8, lhs_p + k_idx * m_panel, 0);
        const svint8_t va1 = svld1_vnum_s8(pg8, lhs_p + k_idx * m_panel, 1);
        const svint8_t vb0 = svld1_vnum_s8(pg8, rhs_w + k_idx * n_panel, 0);
        const svint8_t vb1 = svld1_vnum_s8(pg8, rhs_w + k_idx * n_panel, 1);

        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
        svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
      }

      const svint32_t vbias0 = svld1_vnum_s32(pg32, bias_ptr, 0);
      const svint32_t vbias1 = svld1_vnum_s32(pg32, bias_ptr, 1);
      const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr, 0);
      const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr, 1);
      const svbool_t p_out8 = svwhilelt_b8_u64(0, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        int8_t* dst_ptr = dst + (m_idx + row) * dst_stride_row + n_idx;
        svint32_t vacc0 = svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
        vacc0 = svadd_s32_m(pg32, vacc0, vbias0);
        const svfloat32_t vf0 =
            svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
        svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
        vq0 = svmin_s32_m(
            pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq0, vout_zp), vmin),
            vmax);

        svint32_t vq1 = vq0;
        if (n_rem > vl) {
          svint32_t vacc1 = svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
          vacc1 = svadd_s32_m(pg32, vacc1, vbias1);
          const svfloat32_t vf1 =
              svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
          vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
          vq1 = svmin_s32_m(
              pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq1, vout_zp), vmin),
              vmax);
        }
        const svint16_t vh01 =
            svuzp1_s16(svreinterpret_s16_s32(vq0), svreinterpret_s16_s32(vq1));
        const svint8_t vb01 =
            svuzp1_s8(svreinterpret_s8_s16(vh01), svreinterpret_s8_s16(vh01));
        svst1_s8(p_out8, dst_ptr, vb01);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row = 0; row < rows1; ++row) {
          int8_t* dst_ptr = dst + (m_idx + vl + row) * dst_stride_row + n_idx;
          svint32_t vacc0 = svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row);
          vacc0 = svadd_s32_m(pg32, vacc0, vbias0);
          const svfloat32_t vf0 =
              svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc0), vscale0);
          svint32_t vq0 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf0));
          vq0 = svmin_s32_m(
              pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq0, vout_zp), vmin),
              vmax);

          svint32_t vq1 = vq0;
          if (n_rem > vl) {
            svint32_t vacc1 =
                svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row);
            vacc1 = svadd_s32_m(pg32, vacc1, vbias1);
            const svfloat32_t vf1 =
                svmul_f32_m(pg32, svcvt_f32_s32_x(pg32, vacc1), vscale1);
            vq1 = svcvt_s32_f32_x(pg32, svrintn_f32_x(pg32, vf1));
            vq1 = svmin_s32_m(
                pg32, svmax_s32_m(pg32, svadd_s32_m(pg32, vq1, vout_zp), vmin),
                vmax);
          }
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

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  const size_t k_rounded = ks * round_up_po2(kc, 4);
  xnn_pqs8_qc8w_igemm_2vlx2vl_sme_mopa_impl(
      mr, nc, k_rounded, (const int8_t*)packed_lhs, w, c, cm_stride,
      params->fp32_scalar.output_zero_point,
      (int8_t)params->fp32_scalar.output_min,
      (int8_t)params->fp32_scalar.output_max);
}
#else
#include <assert.h>

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme_get_mr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme_get_nr(void) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pqs8_qc8w_igemm_minmax_fp32_ukernel_32x32c4__neonsme(
    size_t mr, size_t nc, size_t kc, size_t ks, const void* packed_lhs,
    const void* restrict w, int8_t* restrict c, size_t cm_stride,
    const union xnn_qs8_qc8w_conv_minmax_params* params) {
  assert(
      "Calling wrapped KleidiAI function, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
