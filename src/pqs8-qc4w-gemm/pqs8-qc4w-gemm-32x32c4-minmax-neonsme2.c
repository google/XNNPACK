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

__arm_locally_streaming __arm_new(
    "za") static void xnn_pqs8_qc4w_gemm_8vsx8vs_sme_mopa_impl(size_t m,
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
  const size_t k_padded = (k_elements + 3) & ~(size_t)3;
  const size_t rounded_k = (k_elements + 31) & ~(size_t)31;
  const size_t lhs_panel_bytes = m_panel * k_padded;
  const size_t rhs_panel_bytes =
      n_panel * (sizeof(int32_t) + (rounded_k / 2) + sizeof(float));
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
        const float* scale_ptr0 =
            (const float*)(rhs_p0 +
                           n_panel * (sizeof(int32_t) + (rounded_k / 2)));
        const float* scale_ptr1 =
            (const float*)(rhs_p1 +
                           n_panel * (sizeof(int32_t) + (rounded_k / 2)));
        svzero_za();

        const int8_t* a_ptr = lhs_p;
        size_t k_idx = 0;
        for (; k_idx + 16 <= k_padded; k_idx += 16) {
          for (int step = 0; step < 4; ++step) {
            const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, step * 2);
            const svint8_t vpacked0 = svld1_vnum_s8(pg8, b0_ptr, step);
            const svint8_t vpacked1 = svld1_vnum_s8(pg8, b1_ptr, step);
            const svint8_t vlo0 = svlsl_n_s8_x(pg8, vpacked0, 4);
            const svint8_t vhi0 = svand_n_s8_x(pg8, vpacked0, (int8_t)0xF0);
            const svint8_t vlo1 = svlsl_n_s8_x(pg8, vpacked1, 4);
            const svint8_t vhi1 = svand_n_s8_x(pg8, vpacked1, (int8_t)0xF0);
            svmopa_za32_s8_m(0, pg8, pg8, va0, svzip1_s8(vlo0, vhi0));
            svmopa_za32_s8_m(1, pg8, pg8, va0, svzip2_s8(vlo0, vhi0));
            svmopa_za32_s8_m(2, pg8, pg8, va0, svzip1_s8(vlo1, vhi1));
            svmopa_za32_s8_m(3, pg8, pg8, va0, svzip2_s8(vlo1, vhi1));
          }
          a_ptr += 16 * m_panel;
          b0_ptr += 8 * n_panel;
          b1_ptr += 8 * n_panel;
        }
        for (; k_idx < k_padded; k_idx += 4) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
          const svint8_t vpacked0 = svld1_s8(pg8, b0_ptr);
          const svint8_t vpacked1 = svld1_s8(pg8, b1_ptr);
          const svint8_t vlo0 = svlsl_n_s8_x(pg8, vpacked0, 4);
          const svint8_t vhi0 = svand_n_s8_x(pg8, vpacked0, (int8_t)0xF0);
          const svint8_t vlo1 = svlsl_n_s8_x(pg8, vpacked1, 4);
          const svint8_t vhi1 = svand_n_s8_x(pg8, vpacked1, (int8_t)0xF0);

          svmopa_za32_s8_m(0, pg8, pg8, va0, svzip1_s8(vlo0, vhi0));
          svmopa_za32_s8_m(1, pg8, pg8, va0, svzip2_s8(vlo0, vhi0));
          svmopa_za32_s8_m(2, pg8, pg8, va0, svzip1_s8(vlo1, vhi1));
          svmopa_za32_s8_m(3, pg8, pg8, va0, svzip2_s8(vlo1, vhi1));
          a_ptr += 4 * m_panel;
          b0_ptr += 2 * n_panel;
          b1_ptr += 2 * n_panel;
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
          svint32_t vacc0 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), 4);
          svint32_t vacc1 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), 4);
          svint32_t vacc2 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row), 4);
          svint32_t vacc3 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row), 4);
          vacc0 = svadd_s32_x(pg32, vacc0, vbias0);
          vacc1 = svadd_s32_x(pg32, vacc1, vbias1);
          vacc2 = svadd_s32_x(pg32, vacc2, vbias2);
          vacc3 = svadd_s32_x(pg32, vacc3, vbias3);
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
      const float* scale_ptr =
          (const float*)(rhs_p + n_panel * (sizeof(int32_t) + (rounded_k / 2)));
      svzero_za();

      const int8_t* a_ptr = lhs_p;
      size_t k_idx = 0;
      for (; k_idx + 16 <= k_padded; k_idx += 16) {
        for (int step = 0; step < 4; ++step) {
          const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, step * 2);
          const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, step * 2 + 1);
          const svint8_t vpacked = svld1_vnum_s8(pg8, b_ptr, step);
          const svint8_t vlo = svlsl_n_s8_x(pg8, vpacked, 4);
          const svint8_t vhi = svand_n_s8_x(pg8, vpacked, (int8_t)0xF0);
          const svint8_t vb0 = svzip1_s8(vlo, vhi);
          const svint8_t vb1 = svzip2_s8(vlo, vhi);
          svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
          svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
          svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
          svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
        }
        a_ptr += 16 * m_panel;
        b_ptr += 8 * n_panel;
      }
      for (; k_idx < k_padded; k_idx += 4) {
        const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
        const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 1);
        const svint8_t vpacked = svld1_s8(pg8, b_ptr);
        const svint8_t vlo = svlsl_n_s8_x(pg8, vpacked, 4);
        const svint8_t vhi = svand_n_s8_x(pg8, vpacked, (int8_t)0xF0);
        const svint8_t vb0 = svzip1_s8(vlo, vhi);
        const svint8_t vb1 = svzip2_s8(vlo, vhi);

        svmopa_za32_s8_m(0, pg8, pg8, va0, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va0, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va1, vb0);
        svmopa_za32_s8_m(3, pg8, pg8, va1, vb1);
        a_ptr += 4 * m_panel;
        b_ptr += 2 * n_panel;
      }

      const svint32_t vbias0 = svld1_vnum_s32(pg32, bias_ptr, 0);
      const svint32_t vbias1 = svld1_vnum_s32(pg32, bias_ptr, 1);
      const svfloat32_t vscale0 = svld1_vnum_f32(pg32, scale_ptr, 0);
      const svfloat32_t vscale1 = svld1_vnum_f32(pg32, scale_ptr, 1);
      const svbool_t p_out8 = svwhilelt_b8_u64(0, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        int8_t* dst_ptr =
            (int8_t*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
        svint32_t vacc0 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), 4);
        svint32_t vacc1 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), 4);
        vacc0 = svadd_s32_x(pg32, vacc0, vbias0);
        vacc1 = svadd_s32_x(pg32, vacc1, vbias1);
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
          svint32_t vacc0 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row), 4);
          svint32_t vacc1 = svasr_n_s32_x(
              pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row), 4);
          vacc0 = svadd_s32_x(pg32, vacc0, vbias0);
          vacc1 = svadd_s32_x(pg32, vacc1, vbias1);
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
