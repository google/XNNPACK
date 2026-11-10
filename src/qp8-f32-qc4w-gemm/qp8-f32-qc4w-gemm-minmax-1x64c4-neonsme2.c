// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
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
#include <string.h>

#include "src/xnnpack/math.h"

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return 1;
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return 4 * svcntsw();
}

static const int8_t xnn_qc4w_lut[64] = {
    0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0, 0, 3,  0, 0, 0,
    4,  0, 0, 0, 5,  0, 0, 0, 6,  0, 0, 0, 7,  0, 0, 0,
    -8, 0, 0, 0, -7, 0, 0, 0, -6, 0, 0, 0, -5, 0, 0, 0,
    -4, 0, 0, 0, -3, 0, 0, 0, -2, 0, 0, 0, -1, 0, 0, 0};

static XNN_INLINE svint8x4_t xnn_luti4_load_s8_x4(
    const uint8_t* w, int64_t vnum) __arm_streaming __arm_in("zt0")
    __arm_preserves("za") {
  const svuint8x2_t vr = svld1_vnum_u8_x2(svptrue_c8(), w, vnum);
  const svint8x2_t vlo = svluti4_lane_zt_s8_x2(0, svget2_u8(vr, 0), 0);
  const svint8x2_t vhi = svluti4_lane_zt_s8_x2(0, svget2_u8(vr, 1), 0);
  return svcreate4_s8(svget2_s8(vlo, 0), svget2_s8(vlo, 1), svget2_s8(vhi, 0),
                      svget2_s8(vhi, 1));
}

static XNN_INLINE void xnn_qp8_f32_qc4w_epilogue_4vl(
    svint32x4_t vacc, const uint8_t* rhs_meta, svint32_t vlhs_neg_zp,
    svfloat32_t vlhs_scale, svfloat32_t vmin, svfloat32_t vmax, float* dst_row,
    size_t n_idx, size_t n) __arm_streaming __arm_preserves("za", "zt0") {
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc32 = svptrue_c32();
  const svint32x4_t vsums =
      svld1_vnum_s32_x4(pc32, (const int32_t*)rhs_meta, 0);
  const svfloat32x4_t vscales =
      svld1_vnum_f32_x4(pc32, (const float*)rhs_meta, 4);
  const svfloat32x4_t vbiases =
      svld1_vnum_f32_x4(pc32, (const float*)rhs_meta, 8);

  const svint32_t vacc0 = svmla_s32_x(pg32, svget4_s32(vacc, 0),
                                      svget4_s32(vsums, 0), vlhs_neg_zp);
  const svint32_t vacc1 = svmla_s32_x(pg32, svget4_s32(vacc, 1),
                                      svget4_s32(vsums, 1), vlhs_neg_zp);
  const svint32_t vacc2 = svmla_s32_x(pg32, svget4_s32(vacc, 2),
                                      svget4_s32(vsums, 2), vlhs_neg_zp);
  const svint32_t vacc3 = svmla_s32_x(pg32, svget4_s32(vacc, 3),
                                      svget4_s32(vsums, 3), vlhs_neg_zp);

  const svfloat32x4_t vf =
      svcvt_f32_s32_x4(svcreate4_s32(vacc0, vacc1, vacc2, vacc3));
  const svfloat32_t vf0 = svmla_f32_x(
      pg32, svget4_f32(vbiases, 0), svget4_f32(vf, 0),
      svmul_f32_x(pg32, vlhs_scale, svget4_f32(vscales, 0)));
  const svfloat32_t vf1 = svmla_f32_x(
      pg32, svget4_f32(vbiases, 1), svget4_f32(vf, 1),
      svmul_f32_x(pg32, vlhs_scale, svget4_f32(vscales, 1)));
  const svfloat32_t vf2 = svmla_f32_x(
      pg32, svget4_f32(vbiases, 2), svget4_f32(vf, 2),
      svmul_f32_x(pg32, vlhs_scale, svget4_f32(vscales, 2)));
  const svfloat32_t vf3 = svmla_f32_x(
      pg32, svget4_f32(vbiases, 3), svget4_f32(vf, 3),
      svmul_f32_x(pg32, vlhs_scale, svget4_f32(vscales, 3)));

  const svfloat32x4_t vout =
      svclamp_single_f32_x4(svcreate4_f32(vf0, vf1, vf2, vf3), vmin, vmax);
  svst1_f32_x4(svwhilelt_c32_u64(n_idx, n, 4), dst_row + n_idx, vout);
}

__arm_locally_streaming __arm_new("za", "zt0") static XNN_NOINLINE void
xnn_qp8_f32_qc4w_gemm_1x4vl_sme2_dot_impl(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row, float clamp_min,
    float clamp_max) {
  const size_t vl = svcntw();
  const size_t nr = 4 * vl;
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t lhs_row_stride =
      k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float);
  const size_t rhs_meta_offset = nr * (k_internal / 2);
  const size_t rhs_row_stride =
      rhs_meta_offset + nr * (sizeof(int32_t) + sizeof(float) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);
  svldr_zt(0, xnn_qc4w_lut);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const int8_t* lhs_row =
        (const int8_t*)((uintptr_t)lhs_packed + m_idx * lhs_row_stride);
    int32_t lhs_neg_zp;
    float lhs_scale;
    __builtin_memcpy(&lhs_neg_zp, lhs_row + k_internal, sizeof(lhs_neg_zp));
    __builtin_memcpy(&lhs_scale, lhs_row + k_internal + sizeof(int32_t),
                     sizeof(lhs_scale));
    const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scale);
    const svint32_t vlhs_neg_zp = svdup_n_s32(lhs_neg_zp);
    float* dst_row = (float*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 2 * nr < n; n_idx += 4 * nr) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const uint8_t* p1 = p0 + rhs_row_stride;
      const uint8_t* p2 = p1 + rhs_row_stride;
      const uint8_t* p3 = (n_idx + 3 * nr < n) ? (p2 + rhs_row_stride) : p2;
      const uint8_t* w0 = p0;
      const uint8_t* w1 = p1;
      const uint8_t* w2 = p2;
      const uint8_t* w3 = p3;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(2, xnn_luti4_load_s8_x4(w2, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(3, xnn_luti4_load_s8_x4(w3, 0), va, 0);

        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(2, xnn_luti4_load_s8_x4(w2, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(3, xnn_luti4_load_s8_x4(w3, 2), va, 1);

        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(2, xnn_luti4_load_s8_x4(w2, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(3, xnn_luti4_load_s8_x4(w3, 4), va, 2);

        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(2, xnn_luti4_load_s8_x4(w2, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(3, xnn_luti4_load_s8_x4(w3, 6), va, 3);

        w0 += 8 * nr;
        w1 += 8 * nr;
        w2 += 8 * nr;
        w3 += 8 * nr;
      }

      xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(0),
                                    p0 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 0 * nr, n);
      xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(1),
                                    p1 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 1 * nr, n);
      xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(2),
                                    p2 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 2 * nr, n);
      if (n_idx + 3 * nr < n) {
        xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(3),
                                      p3 + rhs_meta_offset, vlhs_neg_zp,
                                      vlhs_scale, vmin, vmax, dst_row,
                                      n_idx + 3 * nr, n);
      }
    }

    if (n_idx + nr < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const uint8_t* p1 = p0 + rhs_row_stride;
      const uint8_t* w0 = p0;
      const uint8_t* w1 = p1;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 6), va, 3);
        svdot_lane_za32_s8_vg1x4(1, xnn_luti4_load_s8_x4(w1, 6), va, 3);
        w0 += 8 * nr;
        w1 += 8 * nr;
      }

      xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(0),
                                    p0 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 0 * nr, n);
      xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(1),
                                    p1 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 1 * nr, n);
      n_idx += 2 * nr;
    }

    if (n_idx < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const uint8_t* w0 = p0;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 2), va, 1);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 4), va, 2);
        svdot_lane_za32_s8_vg1x4(0, xnn_luti4_load_s8_x4(w0, 6), va, 3);
        w0 += 8 * nr;
      }

      xnn_qp8_f32_qc4w_epilogue_4vl(svread_za32_s32_vg1x4(0),
                                    p0 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row, n_idx, n);
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
