// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_qai8dxp_qsi4cxp/kai_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa.h"

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa();
}

void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  const struct xnn_f32_qc4w_minmax_params* minmax_params = params;
  kai_run_matmul_clamp_f32_qai8dxp1vlx8_qsi4cxp4vlx8_1vlx4vl_sme2_mopa(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      /*dst_stride_col=*/sizeof(float), minmax_params->scalar.min,
      minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

#include "src/xnnpack/math.h"

#if defined(__GNUC__) || defined(__clang__)
#define XNN_SME_PREFETCH(ptr) __builtin_prefetch((ptr), 0, 2)
#else
#define XNN_SME_PREFETCH(ptr) ((void)(ptr))
#endif

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr() {
  return svcntsw();
}

size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr() {
  return 4 * svcntsw();
}

static const int8_t xnn_qc4w_lut[64] = {
    0,  0, 0, 0, 1,  0, 0, 0, 2,  0, 0,  0, 3,  0, 0,  0, 4,  0, 0,  0, 5, 0,
    0,  0, 6, 0, 0,  0, 7, 0, 0,  0, -8, 0, 0,  0, -7, 0, 0,  0, -6, 0, 0, 0,
    -5, 0, 0, 0, -4, 0, 0, 0, -3, 0, 0,  0, -2, 0, 0,  0, -1, 0, 0,  0};

__arm_locally_streaming __arm_new("za", "zt0") static XNN_NOINLINE void
xnn_qp8_f32_qc4w_gemm_1vlx4vl_sme2_mopa_impl(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row, float clamp_min,
    float clamp_max) {
  const size_t vl = svcntw();
  const size_t mr = vl;
  const size_t nr = 4 * vl;
  const size_t k_internal = (k + 31) & ~(size_t)31;
  const size_t lhs_block_stride =
      mr * (k_internal * sizeof(int8_t) + sizeof(int32_t) + sizeof(float));
  const size_t rhs_block_stride =
      nr * ((k_internal / 2) + sizeof(int32_t) + sizeof(float) + sizeof(float));
  const size_t total_w_bytes = ((n + nr - 1) / nr) * rhs_block_stride;
#if defined(__APPLE__)
  const size_t prfm_min_w_size = 16 * 1024 * 1024;
#else
  const size_t prfm_min_w_size = 1024 * 1024;
#endif
  const int need_prfm = total_w_bytes > prfm_min_w_size;
  const svbool_t pg8 = svptrue_b8();
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc8 = svptrue_c8();
  const svcount_t pc32 = svptrue_c32();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);
  svldr_zt(0, xnn_qc4w_lut);

  for (size_t m_idx = 0; m_idx < m; m_idx += mr) {
    const size_t m_rem = m - m_idx < mr ? m - m_idx : mr;
    const uint8_t* lhs_block =
        (const uint8_t*)lhs_packed + (m_idx / mr) * lhs_block_stride;
    const int32_t* lhs_neg_zps = (const int32_t*)(lhs_block + mr * k_internal);
    const float* lhs_scales = (const float*)(lhs_neg_zps + mr);

    for (size_t n_idx = 0; n_idx < n; n_idx += nr) {
      const uint8_t* rhs_block =
          (const uint8_t*)rhs_packed + (n_idx / nr) * rhs_block_stride;
      const uint8_t* pf_ptr = NULL;
      if (need_prfm) {
        if (n_idx + nr < n) {
          pf_ptr = rhs_block + rhs_block_stride;
        } else if (m_idx + mr < m) {
          pf_ptr = (const uint8_t*)rhs_packed;
        }
      }
      svzero_za();

      const int8_t* a_ptr = (const int8_t*)lhs_block;
      const uint8_t* w_ptr = rhs_block;
      if (k_internal > 0) {
        svint8x4_t va = svld1_s8_x4(pc8, a_ptr);
        svuint8x4_t vr0 = svld1_vnum_u8_x4(pc8, w_ptr, 0);
        svint8x2_t vb0_01 = svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 0), 0);
        svint8x2_t vb0_23 = svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 1), 0);

        for (size_t k_idx = 0; k_idx + 16 < k_internal; k_idx += 16) {
          const svuint8x4_t vr1 = svld1_vnum_u8_x4(pc8, w_ptr, 4);
          const svint8x2_t vb1_01 =
              svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 2), 0);
          const svint8x2_t vb1_23 =
              svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 3), 0);
          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_01, 0));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_01, 1));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_23, 0));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_23, 1));

          const svint8x2_t vb2_01 =
              svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 0), 0);
          const svint8x2_t vb2_23 =
              svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 1), 0);
          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_01, 0));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_01, 1));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_23, 0));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_23, 1));

          const svint8x2_t vb3_01 =
              svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 2), 0);
          const svint8x2_t vb3_23 =
              svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 3), 0);
          svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_01, 0));
          svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_01, 1));
          svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_23, 0));
          svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_23, 1));

          a_ptr += 16 * mr;
          w_ptr += 8 * nr;
          const svint8_t va3 = svget4_s8(va, 3);
          va = svld1_s8_x4(pc8, a_ptr);
          vr0 = svld1_vnum_u8_x4(pc8, w_ptr, 0);
          vb0_01 = svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 0), 0);
          vb0_23 = svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 1), 0);
          svmopa_za32_s8_m(0, pg8, pg8, va3, svget2_s8(vb3_01, 0));
          svmopa_za32_s8_m(1, pg8, pg8, va3, svget2_s8(vb3_01, 1));
          svmopa_za32_s8_m(2, pg8, pg8, va3, svget2_s8(vb3_23, 0));
          svmopa_za32_s8_m(3, pg8, pg8, va3, svget2_s8(vb3_23, 1));

          if (pf_ptr != NULL) {
            XNN_SME_PREFETCH(pf_ptr + 0);
            XNN_SME_PREFETCH(pf_ptr + 64);
            XNN_SME_PREFETCH(pf_ptr + 128);
            XNN_SME_PREFETCH(pf_ptr + 192);
            XNN_SME_PREFETCH(pf_ptr + 256);
            XNN_SME_PREFETCH(pf_ptr + 320);
            XNN_SME_PREFETCH(pf_ptr + 384);
            XNN_SME_PREFETCH(pf_ptr + 448);
            pf_ptr += 8 * nr;
          }
        }

        const svuint8x4_t vr1 = svld1_vnum_u8_x4(pc8, w_ptr, 4);
        const svint8x2_t vb1_01 =
            svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 2), 0);
        const svint8x2_t vb1_23 =
            svluti4_lane_zt_s8_x2(0, svget4_u8(vr0, 3), 0);
        svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_01, 0));
        svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_01, 1));
        svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_23, 0));
        svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 0), svget2_s8(vb0_23, 1));

        const svint8x2_t vb2_01 =
            svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 0), 0);
        const svint8x2_t vb2_23 =
            svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 1), 0);
        svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_01, 0));
        svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_01, 1));
        svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_23, 0));
        svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 1), svget2_s8(vb1_23, 1));

        const svint8x2_t vb3_01 =
            svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 2), 0);
        const svint8x2_t vb3_23 =
            svluti4_lane_zt_s8_x2(0, svget4_u8(vr1, 3), 0);
        svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_01, 0));
        svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_01, 1));
        svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_23, 0));
        svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 2), svget2_s8(vb2_23, 1));

        svmopa_za32_s8_m(0, pg8, pg8, svget4_s8(va, 3), svget2_s8(vb3_01, 0));
        svmopa_za32_s8_m(1, pg8, pg8, svget4_s8(va, 3), svget2_s8(vb3_01, 1));
        svmopa_za32_s8_m(2, pg8, pg8, svget4_s8(va, 3), svget2_s8(vb3_23, 0));
        svmopa_za32_s8_m(3, pg8, pg8, svget4_s8(va, 3), svget2_s8(vb3_23, 1));

        if (pf_ptr != NULL) {
          XNN_SME_PREFETCH(pf_ptr + 0);
          XNN_SME_PREFETCH(pf_ptr + 128);
          XNN_SME_PREFETCH(pf_ptr + 256);
          XNN_SME_PREFETCH(pf_ptr + 384);
          pf_ptr += 8 * nr;
          XNN_SME_PREFETCH(pf_ptr + 0);
          XNN_SME_PREFETCH(pf_ptr + 256);
          XNN_SME_PREFETCH(pf_ptr + 512);
        }
      }

      const uint8_t* rhs_meta = rhs_block + nr * (k_internal / 2);
      const svint32x4_t vrsum =
          svld1_vnum_s32_x4(pc32, (const int32_t*)rhs_meta, 0);
      const svfloat32x4_t vrscale =
          svld1_vnum_f32_x4(pc32, (const float*)rhs_meta, 4);
      const svfloat32x4_t vrbias =
          svld1_vnum_f32_x4(pc32, (const float*)rhs_meta, 8);
      const svcount_t p_out = svwhilelt_c32_u64(n_idx, n, 4);

      uint8_t* dst_row =
          (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float);
      for (size_t row = 0; row < m_rem; ++row) {
        const svint32_t vlhs_zp = svdup_n_s32(lhs_neg_zps[row]);
        const svfloat32_t vlhs_scale = svdup_n_f32(lhs_scales[row]);
        const svint32x4_t vza = svreinterpret_s32_u8_x4(
            svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));

        const svint32_t vacc0 = svmla_s32_x(pg32, svget4_s32(vza, 0),
                                            svget4_s32(vrsum, 0), vlhs_zp);
        const svint32_t vacc1 = svmla_s32_x(pg32, svget4_s32(vza, 1),
                                            svget4_s32(vrsum, 1), vlhs_zp);
        const svint32_t vacc2 = svmla_s32_x(pg32, svget4_s32(vza, 2),
                                            svget4_s32(vrsum, 2), vlhs_zp);
        const svint32_t vacc3 = svmla_s32_x(pg32, svget4_s32(vza, 3),
                                            svget4_s32(vrsum, 3), vlhs_zp);

        const svfloat32x4_t vf =
            svcvt_f32_s32_x4(svcreate4_s32(vacc0, vacc1, vacc2, vacc3));
        const svfloat32_t vf0 =
            svmla_f32_x(pg32, svget4_f32(vrbias, 0), svget4_f32(vf, 0),
                        svmul_f32_x(pg32, vlhs_scale, svget4_f32(vrscale, 0)));
        const svfloat32_t vf1 =
            svmla_f32_x(pg32, svget4_f32(vrbias, 1), svget4_f32(vf, 1),
                        svmul_f32_x(pg32, vlhs_scale, svget4_f32(vrscale, 1)));
        const svfloat32_t vf2 =
            svmla_f32_x(pg32, svget4_f32(vrbias, 2), svget4_f32(vf, 2),
                        svmul_f32_x(pg32, vlhs_scale, svget4_f32(vrscale, 2)));
        const svfloat32_t vf3 =
            svmla_f32_x(pg32, svget4_f32(vrbias, 3), svget4_f32(vf, 3),
                        svmul_f32_x(pg32, vlhs_scale, svget4_f32(vrscale, 3)));

        const svfloat32x4_t vout = svclamp_single_f32_x4(
            svcreate4_f32(vf0, vf1, vf2, vf3), vmin, vmax);
        svst1_f32_x4(p_out, (float*)dst_row, vout);
        dst_row += dst_stride_row;
      }
    }
  }
}

void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  (void)dst_stride_col;
  const struct xnn_f32_qc4w_minmax_params* minmax_params = params;
  xnn_qp8_f32_qc4w_gemm_1vlx4vl_sme2_mopa_impl(
      m, n, k, lhs_packed, rhs_packed, dst, dst_stride_row,
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_mr() {
  return 0;
}
size_t xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2_get_nr() {
  return 0;
}
void xnn_qp8_f32_qc4w_gemm_minmax_ukernel_16x64c4__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const void* params) {
  assert(0);
}
#endif  // XNN_ENABLE_KLEIDIAI
