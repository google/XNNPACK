// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa.h"

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
}

// Wraps the
// `kai_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa` GEMM
// microkernel with a name that is compatible with our tooling.
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  const struct xnn_f32_minmax_params* minmax_params = params;
  kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

#include "src/xnnpack/math.h"

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr() {
  return svcntsw();
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr() {
  return 4 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_qp8_f32_qc8w_gemm_1vlx4vl_sme2_mopa_impl(size_t m,
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
  const size_t lhs_block_stride =
      mr * (k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float));
  const size_t rhs_block_stride =
      nr * (k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float) +
            sizeof(float));
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

      const int8_t* a_ptr = (const int8_t*)lhs_block;
      const int8_t* b_ptr = (const int8_t*)rhs_block;
      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va0 = svld1_vnum_s8(pg8, a_ptr, 0);
        const svint8_t va1 = svld1_vnum_s8(pg8, a_ptr, 1);
        const svint8_t va2 = svld1_vnum_s8(pg8, a_ptr, 2);
        const svint8_t va3 = svld1_vnum_s8(pg8, a_ptr, 3);

        svmopa_za32_s8_m(0, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 0));
        svmopa_za32_s8_m(1, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 1));
        svmopa_za32_s8_m(2, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 2));
        svmopa_za32_s8_m(3, pg8, pg8, va0, svld1_vnum_s8(pg8, b_ptr, 3));

        svmopa_za32_s8_m(0, pg8, pg8, va1, svld1_vnum_s8(pg8, b_ptr, 4));
        svmopa_za32_s8_m(1, pg8, pg8, va1, svld1_vnum_s8(pg8, b_ptr, 5));
        svmopa_za32_s8_m(2, pg8, pg8, va1, svld1_vnum_s8(pg8, b_ptr, 6));
        svmopa_za32_s8_m(3, pg8, pg8, va1, svld1_vnum_s8(pg8, b_ptr, 7));

        svmopa_za32_s8_m(0, pg8, pg8, va2, svld1_vnum_s8(pg8, b_ptr, 8));
        svmopa_za32_s8_m(1, pg8, pg8, va2, svld1_vnum_s8(pg8, b_ptr, 9));
        svmopa_za32_s8_m(2, pg8, pg8, va2, svld1_vnum_s8(pg8, b_ptr, 10));
        svmopa_za32_s8_m(3, pg8, pg8, va2, svld1_vnum_s8(pg8, b_ptr, 11));

        svmopa_za32_s8_m(0, pg8, pg8, va3, svld1_vnum_s8(pg8, b_ptr, 12));
        svmopa_za32_s8_m(1, pg8, pg8, va3, svld1_vnum_s8(pg8, b_ptr, 13));
        svmopa_za32_s8_m(2, pg8, pg8, va3, svld1_vnum_s8(pg8, b_ptr, 14));
        svmopa_za32_s8_m(3, pg8, pg8, va3, svld1_vnum_s8(pg8, b_ptr, 15));

        a_ptr += 16 * mr;
        b_ptr += 16 * nr;
      }

      const int32_t* rhs_sums = (const int32_t*)(rhs_block + nr * k_internal);
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

        svint32_t vacc0 = svread_hor_za32_s32_m(svundef_s32(), pg32, 0, row);
        svint32_t vacc1 = svread_hor_za32_s32_m(svundef_s32(), pg32, 1, row);
        svint32_t vacc2 = svread_hor_za32_s32_m(svundef_s32(), pg32, 2, row);
        svint32_t vacc3 = svread_hor_za32_s32_m(svundef_s32(), pg32, 3, row);

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

void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  (void)dst_stride_col;
  const struct xnn_f32_minmax_params* minmax_params = params;
  xnn_qp8_f32_qc8w_gemm_1vlx4vl_sme2_mopa_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr() {
  return 0;
}
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr() {
  return 0;
}
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  assert(0);
}
#endif  // XNN_ENABLE_KLEIDIAI
