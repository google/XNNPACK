// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot.h"

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot();
}

void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  const struct xnn_f32_qc4w_minmax_params* minmax_params = params;
  kai_run_matmul_clamp_f32_qai8dxp1x4_qsi4cxp4vlx4_1x4vl_sme2_sdot(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

#include "src/xnnpack/math.h"

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return 1;
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return 4 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_qp8_f32_qc4w_gemm_1x4vl_sme2_dot_impl(size_t m,
                                                                size_t n,
                                                                size_t k,
                                                                const void*
                                                                    lhs_packed,
                                                                const void*
                                                                    rhs_packed,
                                                                float* dst,
                                                                size_t
                                                                    dst_stride_row,
                                                                float clamp_min,
                                                                float
                                                                    clamp_max) {
  const size_t vl = svcntw();
  const size_t nr = 4 * vl;
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t lhs_row_stride =
      k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float);
  const size_t rhs_row_stride =
      nr * ((k_internal / 2) + sizeof(int32_t) + sizeof(float) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t p_row0 = svwhilelt_b8_u64(0, 4);
  const svbool_t p_row1 = sveor_b_z(pg8, svwhilelt_b8_u64(0, 8), p_row0);
  const svbool_t p_row2 =
      sveor_b_z(pg8, svwhilelt_b8_u64(0, 12), svwhilelt_b8_u64(0, 8));
  const svbool_t p_row3 =
      sveor_b_z(pg8, svwhilelt_b8_u64(0, 16), svwhilelt_b8_u64(0, 12));
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* lhs_row =
        (const int8_t*)((uintptr_t)lhs_packed + m_idx * lhs_row_stride);
    const int32_t lhs_neg_zp = *(const int32_t*)(lhs_row + k_internal);
    const float lhs_scale =
        *(const float*)(lhs_row + k_internal + sizeof(int32_t));
    const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scale);
    const svint32_t vlhs_neg_zp = svdup_n_s32(lhs_neg_zp);
    float* dst_row = (float*)((uintptr_t)dst + m_idx * dst_stride_row);

    for (size_t n_idx = 0; n_idx < n; n_idx += nr) {
      const uint8_t* rhs_block =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      svzero_za();

      const int8_t* w_ptr = (const int8_t*)rhs_block;
      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
#pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) {
          const svbool_t p_row =
              u == 0 ? p_row0 : (u == 1 ? p_row1 : (u == 2 ? p_row2 : p_row3));
          const svint8_t vraw01 = svld1_vnum_s8(pg8, w_ptr, 2 * u + 0);
          const svint8_t vraw23 = svld1_vnum_s8(pg8, w_ptr, 2 * u + 1);
          const svint8_t vlo01 = svlsl_n_s8_x(pg8, vraw01, 4);
          const svint8_t vhi01 = svand_n_s8_x(pg8, vraw01, (int8_t)0xF0);
          const svint8_t vlo23 = svlsl_n_s8_x(pg8, vraw23, 4);
          const svint8_t vhi23 = svand_n_s8_x(pg8, vraw23, (int8_t)0xF0);

          svmopa_za32_s8_m(0, p_row, pg8, va, svzip1_s8(vlo01, vhi01));
          svmopa_za32_s8_m(1, p_row, pg8, va, svzip2_s8(vlo01, vhi01));
          svmopa_za32_s8_m(2, p_row, pg8, va, svzip1_s8(vlo23, vhi23));
          svmopa_za32_s8_m(3, p_row, pg8, va, svzip2_s8(vlo23, vhi23));
        }
        w_ptr += 8 * nr;
      }

      svint32_t vacc0 = svasr_n_s32_x(
          pg32,
          svadd_s32_x(
              pg32,
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 0),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 1)),
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 2),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 0, 3))),
          4);
      svint32_t vacc1 = svasr_n_s32_x(
          pg32,
          svadd_s32_x(
              pg32,
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 0),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 1)),
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 2),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 1, 3))),
          4);
      svint32_t vacc2 = svasr_n_s32_x(
          pg32,
          svadd_s32_x(
              pg32,
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 0),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 1)),
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 2),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 2, 3))),
          4);
      svint32_t vacc3 = svasr_n_s32_x(
          pg32,
          svadd_s32_x(
              pg32,
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 0),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 1)),
              svadd_s32_x(pg32,
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 2),
                          svread_hor_za32_s32_m(svundef_s32(), pg32, 3, 3))),
          4);

      const int32_t* rhs_sums =
          (const int32_t*)(rhs_block + nr * (k_internal / 2));
      const float* rhs_scales = (const float*)(rhs_sums + nr);
      const float* rhs_biases = rhs_scales + nr;

      vacc0 = svmla_s32_m(pg32, vacc0, svld1_vnum_s32(pg32, rhs_sums, 0),
                          vlhs_neg_zp);
      vacc1 = svmla_s32_m(pg32, vacc1, svld1_vnum_s32(pg32, rhs_sums, 1),
                          vlhs_neg_zp);
      vacc2 = svmla_s32_m(pg32, vacc2, svld1_vnum_s32(pg32, rhs_sums, 2),
                          vlhs_neg_zp);
      vacc3 = svmla_s32_m(pg32, vacc3, svld1_vnum_s32(pg32, rhs_sums, 3),
                          vlhs_neg_zp);

      svfloat32_t vf0 = svcvt_f32_s32_x(pg32, vacc0);
      svfloat32_t vf1 = svcvt_f32_s32_x(pg32, vacc1);
      svfloat32_t vf2 = svcvt_f32_s32_x(pg32, vacc2);
      svfloat32_t vf3 = svcvt_f32_s32_x(pg32, vacc3);

      vf0 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases, 0), vf0,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales, 0)));
      vf1 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases, 1), vf1,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales, 1)));
      vf2 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases, 2), vf2,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales, 2)));
      vf3 = svmla_f32_m(
          pg32, svld1_vnum_f32(pg32, rhs_biases, 3), vf3,
          svmul_f32_x(pg32, vlhs_scale, svld1_vnum_f32(pg32, rhs_scales, 3)));

      vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
      vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
      vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
      vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);

      const size_t n_rem = n - n_idx;
      svst1_vnum_f32(svwhilelt_b32_u64(0 * vl, n_rem), dst_row + n_idx, 0, vf0);
      svst1_vnum_f32(svwhilelt_b32_u64(1 * vl, n_rem), dst_row + n_idx, 1, vf1);
      svst1_vnum_f32(svwhilelt_b32_u64(2 * vl, n_rem), dst_row + n_idx, 2, vf2);
      svst1_vnum_f32(svwhilelt_b32_u64(3 * vl, n_rem), dst_row + n_idx, 3, vf3);
    }
  }
}

void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  (void)dst_stride_col;
  const struct xnn_f32_qc4w_minmax_params* minmax_params = params;
  xnn_qp8_f32_qc4w_gemm_1x4vl_sme2_dot_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return 0;
}
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return 0;
}
void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  assert(0);
}
#endif  // XNN_ENABLE_KLEIDIAI
