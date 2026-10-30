// Copyright 2026 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa.h"

static const int32_t xnn_qc2w_signed_lut[4] = {0, 1, -2, -1};

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr(void) {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa();
}

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr(void) {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa();
}

void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsu2cxp4vlx4_1vlx4vl_sme2_mopa(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max, xnn_qc2w_signed_lut);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

#include "src/xnnpack/math.h"

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr(void) {
  return svcntsw();
}

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr(void) {
  return 4 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_qp8_f32_qc2w_gemm_1vlx4vl_sme2_mopa_impl(size_t m,
                                                                   size_t n,
                                                                   size_t k,
                                                                   const void*
                                                                       lhs_packed,
                                                                   const void*
                                                                       rhs_packed,
                                                                   float* dst,
                                                                   size_t
                                                                       dst_stride_row,
                                                                   float
                                                                       clamp_min,
                                                                   float
                                                                       clamp_max) {
  const size_t vl = svcntw();
  const size_t mr = vl;
  const size_t nr = 4 * vl;
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t k_bytes = k_internal / 4;
  const size_t lhs_block_stride =
      mr * (k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float));
  const size_t rhs_block_stride =
      nr * (k_bytes + sizeof(int32_t) + sizeof(float) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; m_idx += mr) {
    const size_t m_rem = m - m_idx < mr ? m - m_idx : mr;
    const uint8_t* lhs_block =
        (const uint8_t*)lhs_packed + (m_idx / mr) * lhs_block_stride;
    const int32_t* lhs_neg_zps = (const int32_t*)(lhs_block + mr * k_internal);
    const float* lhs_scales = (const float*)(lhs_neg_zps + mr);

    for (size_t n_idx = 0; n_idx < n; n_idx += nr) {
      const uint8_t* rhs_block =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_block_stride;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 4) {
        const svint8_t va =
            svld1_s8(pg8, (const int8_t*)(lhs_block + k_idx * mr));

        const svint8_t vraw =
            svld1_s8(pg8, (const int8_t*)(rhs_block + (k_idx / 4) * nr));
        const svint8_t vc0 = svlsl_n_s8_x(pg8, vraw, 6);
        const svint8_t vc1 =
            svand_n_s8_x(pg8, svlsl_n_s8_x(pg8, vraw, 4), (int8_t)0xC0);
        const svint8_t vc2 =
            svand_n_s8_x(pg8, svlsl_n_s8_x(pg8, vraw, 2), (int8_t)0xC0);
        const svint8_t vc3 = svand_n_s8_x(pg8, vraw, (int8_t)0xC0);

        const svint8_t v02_lo = svzip1_s8(vc0, vc2);
        const svint8_t v02_hi = svzip2_s8(vc0, vc2);
        const svint8_t v13_lo = svzip1_s8(vc1, vc3);
        const svint8_t v13_hi = svzip2_s8(vc1, vc3);

        const svint8_t vb0 = svzip1_s8(v02_lo, v13_lo);
        const svint8_t vb1 = svzip2_s8(v02_lo, v13_lo);
        const svint8_t vb2 = svzip1_s8(v02_hi, v13_hi);
        const svint8_t vb3 = svzip2_s8(v02_hi, v13_hi);

        svmopa_za32_s8_m(0, pg8, pg8, va, vb0);
        svmopa_za32_s8_m(1, pg8, pg8, va, vb1);
        svmopa_za32_s8_m(2, pg8, pg8, va, vb2);
        svmopa_za32_s8_m(3, pg8, pg8, va, vb3);
      }

      const int32_t* rhs_sums = (const int32_t*)(rhs_block + nr * k_bytes);
      const float* rhs_scales = (const float*)(rhs_sums + nr);
      const float* rhs_biases = rhs_scales + nr;

      const svint32_t vrsum0 = svld1_vnum_s32(pg32, rhs_sums, 0);
      const svint32_t vrsum1 = svld1_vnum_s32(pg32, rhs_sums, 1);
      const svint32_t vrsum2 = svld1_vnum_s32(pg32, rhs_sums, 2);
      const svint32_t vrsum3 = svld1_vnum_s32(pg32, rhs_sums, 3);

      const svfloat32_t vrscale0 = svld1_vnum_f32(pg32, rhs_scales, 0);
      const svfloat32_t vrscale1 = svld1_vnum_f32(pg32, rhs_scales, 1);
      const svfloat32_t vrscale2 = svld1_vnum_f32(pg32, rhs_scales, 2);
      const svfloat32_t vrscale3 = svld1_vnum_f32(pg32, rhs_scales, 3);

      const svfloat32_t vrbias0 = svld1_vnum_f32(pg32, rhs_biases, 0);
      const svfloat32_t vrbias1 = svld1_vnum_f32(pg32, rhs_biases, 1);
      const svfloat32_t vrbias2 = svld1_vnum_f32(pg32, rhs_biases, 2);
      const svfloat32_t vrbias3 = svld1_vnum_f32(pg32, rhs_biases, 3);

      const size_t n_rem = n - n_idx;
      const svbool_t p_out0 = svwhilelt_b32_u64(0 * vl, n_rem);
      const svbool_t p_out1 = svwhilelt_b32_u64(1 * vl, n_rem);
      const svbool_t p_out2 = svwhilelt_b32_u64(2 * vl, n_rem);
      const svbool_t p_out3 = svwhilelt_b32_u64(3 * vl, n_rem);

      for (size_t row = 0; row < m_rem; ++row) {
        const svint32_t vlhs_zp = svdup_n_s32(lhs_neg_zps[row]);
        const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scales[row]);

        svint32_t vacc0 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row), 6);
        svint32_t vacc1 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row), 6);
        svint32_t vacc2 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row), 6);
        svint32_t vacc3 = svasr_n_s32_x(
            pg32, svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row), 6);

        vacc0 = svmla_s32_m(pg32, vacc0, vrsum0, vlhs_zp);
        vacc1 = svmla_s32_m(pg32, vacc1, vrsum1, vlhs_zp);
        vacc2 = svmla_s32_m(pg32, vacc2, vrsum2, vlhs_zp);
        vacc3 = svmla_s32_m(pg32, vacc3, vrsum3, vlhs_zp);

        svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
        svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);
        svfloat32_t vf2 = svcvt_f32_s32_x(pg32, vacc2);
        svfloat32_t vf3 = svcvt_f32_s32_x(pg32, vacc3);

        vf0 = svmla_f32_m(pg32, vrbias0, vf0,
                          svmul_f32_x(pg32, vlhs_scale, vrscale0));
        vf1 = svmla_f32_m(pg32, vrbias1, vf1,
                          svmul_f32_x(pg32, vlhs_scale, vrscale1));
        vf2 = svmla_f32_m(pg32, vrbias2, vf2,
                          svmul_f32_x(pg32, vlhs_scale, vrscale2));
        vf3 = svmla_f32_m(pg32, vrbias3, vf3,
                          svmul_f32_x(pg32, vlhs_scale, vrscale3));

        vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
        vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
        vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
        vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);

        float* dst_ptr =
            (float*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
        svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
        svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
        svst1_vnum_f32(p_out2, dst_ptr, 2, vf2);
        svst1_vnum_f32(p_out3, dst_ptr, 3, vf3);
      }
    }
  }
}

void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_qp8_f32_qc2w_gemm_1vlx4vl_sme2_mopa_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr(void) {
  return 0;
}
size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr(void) {
  return 0;
}
void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
