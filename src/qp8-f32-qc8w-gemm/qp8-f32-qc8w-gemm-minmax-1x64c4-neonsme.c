// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot.h"

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme_get_mr() {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot();
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot();
}

// Wraps the `kai_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  const struct xnn_f32_minmax_params* minmax_params = params;
  kai_run_matmul_clamp_f32_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme_dot(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/math.h"

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme_get_mr() {
  return 1;
}

size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme_get_nr() {
  return 4 * svcntsw();
}

static XNN_INLINE void xnn_qp8_f32_sme_epilogue_4vl(
    svint32_t vacc0, svint32_t vacc1, svint32_t vacc2, svint32_t vacc3,
    const uint8_t* rhs_meta, size_t nr, size_t vl, svint32_t vlhs_neg_zp,
    svfloat32_t vlhs_scale, svfloat32_t vmin, svfloat32_t vmax, float* dst_row,
    size_t n_idx, size_t n) __arm_streaming {
  const svbool_t pg32 = svptrue_b32();
  const int32_t* rhs_sums = (const int32_t*)rhs_meta;
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

__arm_locally_streaming static XNN_NOINLINE void
xnn_qp8_f32_qc8w_gemm_1x4vl_sme_dot_impl(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row, float clamp_min,
    float clamp_max) {
  const size_t vl = svcntw();
  const size_t nr = 4 * vl;
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t lhs_row_stride =
      k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float);
  const size_t rhs_meta_offset = nr * k_internal * sizeof(int8_t);
  const size_t rhs_row_stride =
      rhs_meta_offset + nr * (sizeof(int32_t) + sizeof(float) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* lhs_row =
        (const int8_t*)((uintptr_t)lhs_packed + m_idx * lhs_row_stride);
    int32_t lhs_neg_zp;
    float lhs_scale;
    __builtin_memcpy(&lhs_neg_zp, lhs_row + k_internal, sizeof(lhs_neg_zp));
    __builtin_memcpy(&lhs_scale, lhs_row + k_internal + sizeof(int32_t),
                     sizeof(lhs_scale));
    const svint32_t vlhs_neg_zp = svdup_n_s32(lhs_neg_zp);
    const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scale);
    float* dst_row = (float*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + nr < n; n_idx += 2 * nr) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const uint8_t* p1 = p0 + rhs_row_stride;
      const int8_t* w0 = (const int8_t*)p0;
      const int8_t* w1 = (const int8_t*)p1;

      svint32_t vacc00 = svdup_n_s32(0);
      svint32_t vacc01 = svdup_n_s32(0);
      svint32_t vacc02 = svdup_n_s32(0);
      svint32_t vacc03 = svdup_n_s32(0);
      svint32_t vacc10 = svdup_n_s32(0);
      svint32_t vacc11 = svdup_n_s32(0);
      svint32_t vacc12 = svdup_n_s32(0);
      svint32_t vacc13 = svdup_n_s32(0);

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        vacc00 = svdot_lane_s32(vacc00, svld1_vnum_s8(pg8, w0, 0), va, 0);
        vacc01 = svdot_lane_s32(vacc01, svld1_vnum_s8(pg8, w0, 1), va, 0);
        vacc02 = svdot_lane_s32(vacc02, svld1_vnum_s8(pg8, w0, 2), va, 0);
        vacc03 = svdot_lane_s32(vacc03, svld1_vnum_s8(pg8, w0, 3), va, 0);
        vacc10 = svdot_lane_s32(vacc10, svld1_vnum_s8(pg8, w1, 0), va, 0);
        vacc11 = svdot_lane_s32(vacc11, svld1_vnum_s8(pg8, w1, 1), va, 0);
        vacc12 = svdot_lane_s32(vacc12, svld1_vnum_s8(pg8, w1, 2), va, 0);
        vacc13 = svdot_lane_s32(vacc13, svld1_vnum_s8(pg8, w1, 3), va, 0);

        vacc00 = svdot_lane_s32(vacc00, svld1_vnum_s8(pg8, w0, 4), va, 1);
        vacc01 = svdot_lane_s32(vacc01, svld1_vnum_s8(pg8, w0, 5), va, 1);
        vacc02 = svdot_lane_s32(vacc02, svld1_vnum_s8(pg8, w0, 6), va, 1);
        vacc03 = svdot_lane_s32(vacc03, svld1_vnum_s8(pg8, w0, 7), va, 1);
        vacc10 = svdot_lane_s32(vacc10, svld1_vnum_s8(pg8, w1, 4), va, 1);
        vacc11 = svdot_lane_s32(vacc11, svld1_vnum_s8(pg8, w1, 5), va, 1);
        vacc12 = svdot_lane_s32(vacc12, svld1_vnum_s8(pg8, w1, 6), va, 1);
        vacc13 = svdot_lane_s32(vacc13, svld1_vnum_s8(pg8, w1, 7), va, 1);

        vacc00 = svdot_lane_s32(vacc00, svld1_vnum_s8(pg8, w0, 8), va, 2);
        vacc01 = svdot_lane_s32(vacc01, svld1_vnum_s8(pg8, w0, 9), va, 2);
        vacc02 = svdot_lane_s32(vacc02, svld1_vnum_s8(pg8, w0, 10), va, 2);
        vacc03 = svdot_lane_s32(vacc03, svld1_vnum_s8(pg8, w0, 11), va, 2);
        vacc10 = svdot_lane_s32(vacc10, svld1_vnum_s8(pg8, w1, 8), va, 2);
        vacc11 = svdot_lane_s32(vacc11, svld1_vnum_s8(pg8, w1, 9), va, 2);
        vacc12 = svdot_lane_s32(vacc12, svld1_vnum_s8(pg8, w1, 10), va, 2);
        vacc13 = svdot_lane_s32(vacc13, svld1_vnum_s8(pg8, w1, 11), va, 2);

        vacc00 = svdot_lane_s32(vacc00, svld1_vnum_s8(pg8, w0, 12), va, 3);
        vacc01 = svdot_lane_s32(vacc01, svld1_vnum_s8(pg8, w0, 13), va, 3);
        vacc02 = svdot_lane_s32(vacc02, svld1_vnum_s8(pg8, w0, 14), va, 3);
        vacc03 = svdot_lane_s32(vacc03, svld1_vnum_s8(pg8, w0, 15), va, 3);
        vacc10 = svdot_lane_s32(vacc10, svld1_vnum_s8(pg8, w1, 12), va, 3);
        vacc11 = svdot_lane_s32(vacc11, svld1_vnum_s8(pg8, w1, 13), va, 3);
        vacc12 = svdot_lane_s32(vacc12, svld1_vnum_s8(pg8, w1, 14), va, 3);
        vacc13 = svdot_lane_s32(vacc13, svld1_vnum_s8(pg8, w1, 15), va, 3);

        w0 += 16 * nr;
        w1 += 16 * nr;
      }

      xnn_qp8_f32_sme_epilogue_4vl(vacc00, vacc01, vacc02, vacc03,
                                   p0 + rhs_meta_offset, nr, vl, vlhs_neg_zp,
                                   vlhs_scale, vmin, vmax, dst_row, n_idx, n);
      xnn_qp8_f32_sme_epilogue_4vl(vacc10, vacc11, vacc12, vacc13,
                                   p1 + rhs_meta_offset, nr, vl, vlhs_neg_zp,
                                   vlhs_scale, vmin, vmax, dst_row, n_idx + nr,
                                   n);
    }

    if (n_idx < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const int8_t* w0 = (const int8_t*)p0;

      svint32_t vacc0 = svdup_n_s32(0);
      svint32_t vacc1 = svdup_n_s32(0);
      svint32_t vacc2 = svdup_n_s32(0);
      svint32_t vacc3 = svdup_n_s32(0);
      svint32_t vacc4 = svdup_n_s32(0);
      svint32_t vacc5 = svdup_n_s32(0);
      svint32_t vacc6 = svdup_n_s32(0);
      svint32_t vacc7 = svdup_n_s32(0);

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        vacc0 = svdot_lane_s32(vacc0, svld1_vnum_s8(pg8, w0, 0), va, 0);
        vacc1 = svdot_lane_s32(vacc1, svld1_vnum_s8(pg8, w0, 1), va, 0);
        vacc2 = svdot_lane_s32(vacc2, svld1_vnum_s8(pg8, w0, 2), va, 0);
        vacc3 = svdot_lane_s32(vacc3, svld1_vnum_s8(pg8, w0, 3), va, 0);

        vacc4 = svdot_lane_s32(vacc4, svld1_vnum_s8(pg8, w0, 4), va, 1);
        vacc5 = svdot_lane_s32(vacc5, svld1_vnum_s8(pg8, w0, 5), va, 1);
        vacc6 = svdot_lane_s32(vacc6, svld1_vnum_s8(pg8, w0, 6), va, 1);
        vacc7 = svdot_lane_s32(vacc7, svld1_vnum_s8(pg8, w0, 7), va, 1);

        vacc0 = svdot_lane_s32(vacc0, svld1_vnum_s8(pg8, w0, 8), va, 2);
        vacc1 = svdot_lane_s32(vacc1, svld1_vnum_s8(pg8, w0, 9), va, 2);
        vacc2 = svdot_lane_s32(vacc2, svld1_vnum_s8(pg8, w0, 10), va, 2);
        vacc3 = svdot_lane_s32(vacc3, svld1_vnum_s8(pg8, w0, 11), va, 2);

        vacc4 = svdot_lane_s32(vacc4, svld1_vnum_s8(pg8, w0, 12), va, 3);
        vacc5 = svdot_lane_s32(vacc5, svld1_vnum_s8(pg8, w0, 13), va, 3);
        vacc6 = svdot_lane_s32(vacc6, svld1_vnum_s8(pg8, w0, 14), va, 3);
        vacc7 = svdot_lane_s32(vacc7, svld1_vnum_s8(pg8, w0, 15), va, 3);

        w0 += 16 * nr;
      }

      vacc0 = svadd_s32_x(pg32, vacc0, vacc4);
      vacc1 = svadd_s32_x(pg32, vacc1, vacc5);
      vacc2 = svadd_s32_x(pg32, vacc2, vacc6);
      vacc3 = svadd_s32_x(pg32, vacc3, vacc7);

      xnn_qp8_f32_sme_epilogue_4vl(vacc0, vacc1, vacc2, vacc3,
                                   p0 + rhs_meta_offset, nr, vl, vlhs_neg_zp,
                                   vlhs_scale, vmin, vmax, dst_row, n_idx, n);
    }
  }
}

void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  (void)dst_stride_col;
  const struct xnn_f32_minmax_params* minmax_params = params;
  xnn_qp8_f32_qc8w_gemm_1x4vl_sme_dot_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme_get_mr() {
  return 0;
}
size_t xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme_get_nr() {
  return 0;
}
void xnn_qp8_f32_qc8w_gemm_minmax_ukernel_1x64c4__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  assert(0);
}
#endif  // XNN_ENABLE_KLEIDIAI
