// Copyright 2026 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsu2cxp/kai_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot.h"

// XNNPACK QC2 weights use signed two's-complement crumbs. This fast path is
// selected only when every per-channel weight zero point is zero.
static const int32_t xnn_qc2w_signed_lut[16] = {0, 1, -2, -1};

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr(void) {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();
}

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr(void) {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot();
}

void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
  assert(m == 1);
  kai_run_matmul_clamp_f32_qai8dxp1x4_qsu2cxp4vlx4_1x4vl_sme2_dot(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max, xnn_qc2w_signed_lut);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

#include "src/xnnpack/math.h"

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr(void) {
  return 1;
}

size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr(void) {
  return 4 * svcntsw();
}

// 64-byte ZT0 lookup table for signed 2-bit two's-complement crumbs {0, 1, -2, -1}.
static const int32_t xnn_qc2w_lut[16] = {0, 1, -2, -1};

static XNN_INLINE void xnn_qp8_f32_qc2w_epilogue_4vl(
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
xnn_qp8_f32_qc2w_gemm_1x4vl_sme2_dot_impl(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row, float clamp_min,
    float clamp_max) {
  const size_t vl = svcntw();
  const size_t nr = 4 * vl;
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t k_bytes = k_internal / 4;
  const size_t lhs_row_stride =
      k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float);
  const size_t rhs_meta_offset = nr * k_bytes;
  const size_t rhs_row_stride =
      rhs_meta_offset + nr * (sizeof(int32_t) + sizeof(float) + sizeof(float));
  const svbool_t pg8 = svptrue_b8();
  const svcount_t pc8 = svptrue_c8();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);
  svldr_zt(0, xnn_qc2w_lut);

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
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        const svuint8x4_t vr1 = svld1_u8_x4(pc8, w1);
        const svuint8x4_t vr2 = svld1_u8_x4(pc8, w2);
        const svuint8x4_t vr3 = svld1_u8_x4(pc8, w3);

        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x4(
            2, svluti2_lane_zt_s8_x4(0, svget4_u8(vr2, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x4(
            3, svluti2_lane_zt_s8_x4(0, svget4_u8(vr3, 0), 0), va, 0);

        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x4(
            2, svluti2_lane_zt_s8_x4(0, svget4_u8(vr2, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x4(
            3, svluti2_lane_zt_s8_x4(0, svget4_u8(vr3, 1), 0), va, 1);

        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x4(
            2, svluti2_lane_zt_s8_x4(0, svget4_u8(vr2, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x4(
            3, svluti2_lane_zt_s8_x4(0, svget4_u8(vr3, 2), 0), va, 2);

        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 3), 0), va, 3);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 3), 0), va, 3);
        svdot_lane_za32_s8_vg1x4(
            2, svluti2_lane_zt_s8_x4(0, svget4_u8(vr2, 3), 0), va, 3);
        svdot_lane_za32_s8_vg1x4(
            3, svluti2_lane_zt_s8_x4(0, svget4_u8(vr3, 3), 0), va, 3);

        w0 += 4 * nr;
        w1 += 4 * nr;
        w2 += 4 * nr;
        w3 += 4 * nr;
      }

      xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(0),
                                    p0 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 0 * nr, n);
      xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(1),
                                    p1 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 1 * nr, n);
      xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(2),
                                    p2 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 2 * nr, n);
      if (n_idx + 3 * nr < n) {
        xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(3),
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
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        const svuint8x4_t vr1 = svld1_u8_x4(pc8, w1);

        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 3), 0), va, 3);
        svdot_lane_za32_s8_vg1x4(
            1, svluti2_lane_zt_s8_x4(0, svget4_u8(vr1, 3), 0), va, 3);
        w0 += 4 * nr;
        w1 += 4 * nr;
      }

      xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(0),
                                    p0 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row,
                                    n_idx + 0 * nr, n);
      xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(1),
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
        const svuint8x4_t vr0 = svld1_u8_x4(pc8, w0);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 0), 0), va, 0);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 1), 0), va, 1);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 2), 0), va, 2);
        svdot_lane_za32_s8_vg1x4(
            0, svluti2_lane_zt_s8_x4(0, svget4_u8(vr0, 3), 0), va, 3);
        w0 += 4 * nr;
      }

      xnn_qp8_f32_qc2w_epilogue_4vl(svread_za32_s32_vg1x4(0),
                                    p0 + rhs_meta_offset, vlhs_neg_zp,
                                    vlhs_scale, vmin, vmax, dst_row, n_idx, n);
    }
  }
}

void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_qp8_f32_qc2w_gemm_1x4vl_sme2_dot_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_mr(void) {
  return 0;
}
size_t xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2_get_nr(void) {
  return 0;
}
void xnn_qp8_f32_qc2w_gemm_minmax_ukernel_1x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
