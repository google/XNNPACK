// Copyright 2026 Google LLC
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
#include "kai/ukernels/matmul/matmul_clamp_f16_qai8dxp_qsi8cxp/kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot.h"

size_t xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot();
}

size_t xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot();
}

// Wraps the `kai_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* minmax_params) {
  const struct xnn_f16_minmax_params* f16_minmax_params =
      (const struct xnn_f16_minmax_params*)minmax_params;
  kai_run_matmul_clamp_f16_qai8dxp1x4_qsi8cxp4vlx4_1x4vl_sme2_dot(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row, dst_stride_col,
      xnn_float16_to_float(f16_minmax_params->scalar.min),
      xnn_float16_to_float(f16_minmax_params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <string.h>

size_t xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return 1;
}

size_t xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return 4 * svcntsw();
}

static XNN_INLINE void xnn_qp8_f16_epilogue_4vl(
    svint32x4_t vacc, const uint8_t* rhs_meta, svint32_t vlhs_neg_zp,
    svfloat32_t vlhs_scale, svfloat32_t vmin, svfloat32_t vmax,
    float16_t* dst_row, size_t n_idx, size_t n) __arm_streaming
    __arm_preserves("za") {
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

  const svfloat32x4_t vclamped =
      svclamp_single_f32_x4(svcreate4_f32(vf0, vf1, vf2, vf3), vmin, vmax);
  const svfloat16_t vh01 = svcvt_f16_f32_x2(
      svcreate2_f32(svget4_f32(vclamped, 0), svget4_f32(vclamped, 1)));
  const svfloat16_t vh23 = svcvt_f16_f32_x2(
      svcreate2_f32(svget4_f32(vclamped, 2), svget4_f32(vclamped, 3)));
  svst1_f16_x2(svwhilelt_c16_u64(n_idx, n, 2), dst_row + n_idx,
               svcreate2_f16(vh01, vh23));
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_qp8_f16_qc8w_gemm_1x4vl_sme2_dot_impl(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row, float clamp_min,
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
  const svcount_t pc8 = svptrue_c8();
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
    float16_t* dst_row = (float16_t*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 2 * nr < n; n_idx += 4 * nr) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const uint8_t* p1 = p0 + rhs_row_stride;
      const uint8_t* p2 = p1 + rhs_row_stride;
      const uint8_t* p3 = (n_idx + 3 * nr < n) ? (p2 + rhs_row_stride) : p2;
      const int8_t* w0 = (const int8_t*)p0;
      const int8_t* w1 = (const int8_t*)p1;
      const int8_t* w2 = (const int8_t*)p2;
      const int8_t* w3 = (const int8_t*)p3;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(2, svld1_vnum_s8_x4(pc8, w2, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(3, svld1_vnum_s8_x4(pc8, w3, 0), va, 0);

        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 4), va, 1);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 4), va, 1);
        svdot_lane_za32_s8_vg1x4(2, svld1_vnum_s8_x4(pc8, w2, 4), va, 1);
        svdot_lane_za32_s8_vg1x4(3, svld1_vnum_s8_x4(pc8, w3, 4), va, 1);

        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 8), va, 2);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 8), va, 2);
        svdot_lane_za32_s8_vg1x4(2, svld1_vnum_s8_x4(pc8, w2, 8), va, 2);
        svdot_lane_za32_s8_vg1x4(3, svld1_vnum_s8_x4(pc8, w3, 8), va, 2);

        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 12), va, 3);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 12), va, 3);
        svdot_lane_za32_s8_vg1x4(2, svld1_vnum_s8_x4(pc8, w2, 12), va, 3);
        svdot_lane_za32_s8_vg1x4(3, svld1_vnum_s8_x4(pc8, w3, 12), va, 3);

        w0 += 16 * nr;
        w1 += 16 * nr;
        w2 += 16 * nr;
        w3 += 16 * nr;
      }

      xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(0), p0 + rhs_meta_offset,
                               vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                               n_idx + 0 * nr, n);
      xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(1), p1 + rhs_meta_offset,
                               vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                               n_idx + 1 * nr, n);
      xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(2), p2 + rhs_meta_offset,
                               vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                               n_idx + 2 * nr, n);
      if (n_idx + 3 * nr < n) {
        xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(3), p3 + rhs_meta_offset,
                                 vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                                 n_idx + 3 * nr, n);
      }
    }

    if (n_idx + nr < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const uint8_t* p1 = p0 + rhs_row_stride;
      const int8_t* w0 = (const int8_t*)p0;
      const int8_t* w1 = (const int8_t*)p1;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 4), va, 1);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 4), va, 1);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 8), va, 2);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 8), va, 2);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 12), va, 3);
        svdot_lane_za32_s8_vg1x4(1, svld1_vnum_s8_x4(pc8, w1, 12), va, 3);
        w0 += 16 * nr;
        w1 += 16 * nr;
      }

      xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(0), p0 + rhs_meta_offset,
                               vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                               n_idx + 0 * nr, n);
      xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(1), p1 + rhs_meta_offset,
                               vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                               n_idx + 1 * nr, n);
      n_idx += 2 * nr;
    }

    if (n_idx < n) {
      const uint8_t* p0 =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_row_stride;
      const int8_t* w0 = (const int8_t*)p0;
      svzero_za();

      for (size_t k_idx = 0; k_idx < k_internal; k_idx += 16) {
        const svint8_t va = svld1rq_s8(pg8, lhs_row + k_idx);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 0), va, 0);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 4), va, 1);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 8), va, 2);
        svdot_lane_za32_s8_vg1x4(0, svld1_vnum_s8_x4(pc8, w0, 12), va, 3);
        w0 += 16 * nr;
      }

      xnn_qp8_f16_epilogue_4vl(svread_za32_s32_vg1x4(0), p0 + rhs_meta_offset,
                               vlhs_neg_zp, vlhs_scale, vmin, vmax, dst_row,
                               n_idx, n);
    }
  }
}

void xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* minmax_params) {
  (void)dst_stride_col;
  const struct xnn_f16_minmax_params* f16_minmax_params =
      (const struct xnn_f16_minmax_params*)minmax_params;
  xnn_qp8_f16_qc8w_gemm_1x4vl_sme2_dot_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      xnn_float16_to_float(f16_minmax_params->scalar.min),
      xnn_float16_to_float(f16_minmax_params->scalar.max));
}
#else
size_t xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr() {
  return 0;
}
size_t xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr() {
  return 0;
}
void xnn_qp8_f16_qc8w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
