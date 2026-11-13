// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_32x32__sme_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_32x32__sme_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za")
static void pf32_qc4w_gemm_32x32_sme_lut_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max,
    int32_t kernel_zero_point)
{
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  const size_t w_block_size = 8 * vl + ((kc + 1) / 2) * 2 * vl + 8 * vl;

#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_min_w_size = 16 * 1024 * 1024;
#else
  const size_t prfm_min_w_size = 1024 * 1024;
#endif
  const size_t total_w_size = ((nc + nr - 1) / nr) * w_block_size;
  const bool need_prfm = (total_w_size > prfm_min_w_size);
#endif

  const svbool_t pg_32_all = svptrue_b32();
  const svfloat32_t vlut = svcvt_f32_s32_x(
      pg_32_all, svsub_n_s32_x(pg_32_all, svindex_s32(0, 1), kernel_zero_point));
  const svfloat32_t f32_one = svdup_n_f32(1.0f);
  const svfloat32_t vmin = svdup_n_f32(min);
  const svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  const uint8_t* w_base = (const uint8_t*) w;

  for (size_t m_start = 0; m_start < mr; m_start += 2 * vl) {
    const size_t m_chunk = (mr - m_start < 2 * vl) ? (mr - m_start) : (2 * vl);
    const float* a_panel = a_base + m_start * kc;
    float* c0 = (float*) ((uintptr_t) c + m_start * cm_stride);
    const uint8_t* w_cur = w_base;
    size_t nc_rem = nc;

    if (m_chunk <= vl) {
      while (nc_rem >= 2 * nr) {
        const uint8_t* w0 = w_cur;
        const uint8_t* w1 = w0 + w_block_size;
        w_cur += 2 * w_block_size;

#ifdef PREFETCH_W_MATRIX
        const uint8_t* pf0 = NULL;
        const uint8_t* pf1 = NULL;
        if (need_prfm) {
          if (nc_rem > 2 * nr) {
            pf0 = w_cur;
            pf1 = (nc_rem > 3 * nr) ? (w_cur + w_block_size) : w_cur;
          } else if (m_start + 2 * vl < mr) {
            pf0 = w_base;
            pf1 = (nc > nr) ? (w_base + w_block_size) : w_base;
          }
          if (pf0 != NULL) {
            __builtin_prefetch(pf0, 0, 2);
            __builtin_prefetch(pf1, 0, 2);
#if !defined(__APPLE__)
            __builtin_prefetch(pf0 + 64, 0, 2);
            __builtin_prefetch(pf1 + 64, 0, 2);
#endif
            pf0 += 8 * vl;
            pf1 += 8 * vl;
          }
        }
#endif

        const svfloat32_t vbias_0_lo = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
        const svfloat32_t vbias_0_hi = svld1_vnum_f32(pg_32_all, (const float*) w0, 1);
        const svfloat32_t vbias_1_lo = svld1_vnum_f32(pg_32_all, (const float*) w1, 0);
        const svfloat32_t vbias_1_hi = svld1_vnum_f32(pg_32_all, (const float*) w1, 1);
        w0 += 8 * vl;
        w1 += 8 * vl;

        svzero_za();
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, vbias_0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, vbias_0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, vbias_1_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, f32_one, vbias_1_hi);

        const float* a0 = a_panel;
        size_t k = kc;
        size_t k_chunks = k >> 2;
        if (k_chunks != 0) {
          svuint32_t vw0_01_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
          svuint32_t vw0_01_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
          svuint32_t vw1_01_lo = svld1ub_vnum_u32(pg_32_all, w1, 0);
          svuint32_t vw1_01_hi = svld1ub_vnum_u32(pg_32_all, w1, 1);
          svfloat32_t va0 = svld1_vnum_f32(pg_32_all, a0, 0);
          svfloat32_t va1 = svld1_vnum_f32(pg_32_all, a0, 2);

          for (--k_chunks; k_chunks != 0; --k_chunks) {
            const svuint32_t vw0_23_lo = svld1ub_vnum_u32(pg_32_all, w0, 2);
            const svuint32_t vw0_23_hi = svld1ub_vnum_u32(pg_32_all, w0, 3);
            const svuint32_t vw1_23_lo = svld1ub_vnum_u32(pg_32_all, w1, 2);
            const svuint32_t vw1_23_hi = svld1ub_vnum_u32(pg_32_all, w1, 3);
            const svfloat32_t va2 = svld1_vnum_f32(pg_32_all, a0, 4);
            const svfloat32_t va3 = svld1_vnum_f32(pg_32_all, a0, 6);
            w0 += 4 * vl;
            w1 += 4 * vl;
            a0 += 8 * vl;

            svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_01_lo, 0xF)));
            svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_01_hi, 0xF)));
            svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_01_lo, 0xF)));
            svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_01_hi, 0xF)));
            svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_01_lo, 4)));
            svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_01_hi, 4)));
            svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_01_lo, 4)));
            svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_01_hi, 4)));

            vw0_01_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
            vw0_01_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
            vw1_01_lo = svld1ub_vnum_u32(pg_32_all, w1, 0);
            vw1_01_hi = svld1ub_vnum_u32(pg_32_all, w1, 1);
            va0 = svld1_vnum_f32(pg_32_all, a0, 0);
            va1 = svld1_vnum_f32(pg_32_all, a0, 2);

            svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_23_lo, 0xF)));
            svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_23_hi, 0xF)));
            svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_23_lo, 0xF)));
            svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_23_hi, 0xF)));
            svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_23_lo, 4)));
            svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_23_hi, 4)));
            svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_23_lo, 4)));
            svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_23_hi, 4)));

#ifdef PREFETCH_W_MATRIX
            if (pf0 != NULL) {
              __builtin_prefetch(pf0, 0, 2);
              __builtin_prefetch(pf1, 0, 2);
              pf0 += 4 * vl;
              pf1 += 4 * vl;
            }
#endif
          }

          const svuint32_t vw0_23_lo = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vw0_23_hi = svld1ub_vnum_u32(pg_32_all, w0, 3);
          const svuint32_t vw1_23_lo = svld1ub_vnum_u32(pg_32_all, w1, 2);
          const svuint32_t vw1_23_hi = svld1ub_vnum_u32(pg_32_all, w1, 3);
          const svfloat32_t va2 = svld1_vnum_f32(pg_32_all, a0, 4);
          const svfloat32_t va3 = svld1_vnum_f32(pg_32_all, a0, 6);
          w0 += 4 * vl;
          w1 += 4 * vl;
          a0 += 8 * vl;

          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_01_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_01_hi, 0xF)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_01_lo, 0xF)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_01_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_01_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_01_hi, 4)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_01_lo, 4)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_01_hi, 4)));

          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_23_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_23_hi, 0xF)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_23_lo, 0xF)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_23_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_23_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_23_hi, 4)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_23_lo, 4)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_23_hi, 4)));
          k &= 3;
        }

        if (k >= 2) {
          const svuint32_t vw0_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
          const svuint32_t vw0_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
          const svuint32_t vw1_lo = svld1ub_vnum_u32(pg_32_all, w1, 0);
          const svuint32_t vw1_hi = svld1ub_vnum_u32(pg_32_all, w1, 1);
          w0 += 2 * vl;
          w1 += 2 * vl;
          const svfloat32_t va0 = svld1_vnum_f32(pg_32_all, a0, 0);
          const svfloat32_t va1 = svld1_vnum_f32(pg_32_all, a0, 2);
          a0 += 4 * vl;

          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_hi, 0xF)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_lo, 0xF)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_hi, 4)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_lo, 4)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_hi, 4)));
          k -= 2;
        }

        if (k != 0) {
          const svuint32_t vw0_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
          const svuint32_t vw0_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
          const svuint32_t vw1_lo = svld1ub_vnum_u32(pg_32_all, w1, 0);
          const svuint32_t vw1_hi = svld1ub_vnum_u32(pg_32_all, w1, 1);
          w0 += 2 * vl;
          w1 += 2 * vl;
          const svfloat32_t va0 = svld1_f32(pg_32_all, a0);
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_hi, 0xF)));
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_lo, 0xF)));
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_hi, 0xF)));
        }

        const svfloat32_t vs0 = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
        const svfloat32_t vs1 = svld1_vnum_f32(pg_32_all, (const float*) w0, 1);
        const svfloat32_t vs2 = svld1_vnum_f32(pg_32_all, (const float*) w1, 0);
        const svfloat32_t vs3 = svld1_vnum_f32(pg_32_all, (const float*) w1, 1);

        uint8_t* dst_row = (uint8_t*) c0;
        size_t row = 0;
        for (; row + 2 <= m_chunk; row += 2) {
          svfloat32_t r0_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 0);
          svfloat32_t r0_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 0);
          svfloat32_t r0_2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 0);
          svfloat32_t r0_3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row + 0);
          svfloat32_t r1_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 1);
          svfloat32_t r1_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 1);
          svfloat32_t r1_2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 1);
          svfloat32_t r1_3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row + 1);

          r0_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_0, vs0), vmin), vmax);
          r0_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_1, vs1), vmin), vmax);
          r0_2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_2, vs2), vmin), vmax);
          r0_3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_3, vs3), vmin), vmax);
          r1_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_0, vs0), vmin), vmax);
          r1_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_1, vs1), vmin), vmax);
          r1_2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_2, vs2), vmin), vmax);
          r1_3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_3, vs3), vmin), vmax);

          svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r0_0);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r0_1);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 2, r0_2);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 3, r0_3);
          dst_row += cm_stride;
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r1_0);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r1_1);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 2, r1_2);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 3, r1_3);
          dst_row += cm_stride;
        }
        for (; row < m_chunk; ++row) {
          svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row);
          svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row);
          svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row);
          svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row);
          r0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0, vs0), vmin), vmax);
          r1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1, vs1), vmin), vmax);
          r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vs2), vmin), vmax);
          r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3, vs3), vmin), vmax);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r0);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r1);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 2, r2);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 3, r3);
          dst_row += cm_stride;
        }

        c0 += 2 * nr;
        nc_rem -= 2 * nr;
      }

      if (nc_rem > nr) {
        const svbool_t pg1_lo = svwhilelt_b32_u64(nr, nc_rem);
        const svbool_t pg1_hi = svwhilelt_b32_u64(nr + vl, nc_rem);
        const uint8_t* w0 = w_cur;
        const uint8_t* w1 = w0 + w_block_size;

        const svfloat32_t vbias_0_lo = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
        const svfloat32_t vbias_0_hi = svld1_vnum_f32(pg_32_all, (const float*) w0, 1);
        const svfloat32_t vbias_1_lo = svld1_vnum_f32(pg1_lo, (const float*) w1, 0);
        const svfloat32_t vbias_1_hi = svld1_vnum_f32(pg1_hi, (const float*) w1, 1);
        w0 += 8 * vl;
        w1 += 8 * vl;

        svzero_za();
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, vbias_0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, vbias_0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg1_lo, f32_one, vbias_1_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg1_hi, f32_one, vbias_1_hi);

        const float* a0 = a_panel;
        size_t k = kc;
        for (; k >= 2; k -= 2) {
          const svuint32_t vw0_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
          const svuint32_t vw0_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
          const svuint32_t vw1_lo = svld1ub_vnum_u32(pg1_lo, w1, 0);
          const svuint32_t vw1_hi = svld1ub_vnum_u32(pg1_hi, w1, 1);
          w0 += 2 * vl;
          w1 += 2 * vl;
          const svfloat32_t va0 = svld1_vnum_f32(pg_32_all, a0, 0);
          const svfloat32_t va1 = svld1_vnum_f32(pg_32_all, a0, 2);
          a0 += 4 * vl;
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_hi, 0xF)));
          svmopa_za32_f32_m(2, pg_32_all, pg1_lo, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_lo, 0xF)));
          svmopa_za32_f32_m(3, pg_32_all, pg1_hi, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw0_hi, 4)));
          svmopa_za32_f32_m(2, pg_32_all, pg1_lo, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_lo, 4)));
          svmopa_za32_f32_m(3, pg_32_all, pg1_hi, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw1_hi, 4)));
        }
        if (k != 0) {
          const svuint32_t vw0_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
          const svuint32_t vw0_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
          const svuint32_t vw1_lo = svld1ub_vnum_u32(pg1_lo, w1, 0);
          const svuint32_t vw1_hi = svld1ub_vnum_u32(pg1_hi, w1, 1);
          w0 += 2 * vl;
          w1 += 2 * vl;
          const svfloat32_t va0 = svld1_f32(pg_32_all, a0);
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw0_hi, 0xF)));
          svmopa_za32_f32_m(2, pg_32_all, pg1_lo, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_lo, 0xF)));
          svmopa_za32_f32_m(3, pg_32_all, pg1_hi, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw1_hi, 0xF)));
        }

        const svfloat32_t vs0 = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
        const svfloat32_t vs1 = svld1_vnum_f32(pg_32_all, (const float*) w0, 1);
        const svfloat32_t vs2 = svld1_vnum_f32(pg1_lo, (const float*) w1, 0);
        const svfloat32_t vs3 = svld1_vnum_f32(pg1_hi, (const float*) w1, 1);

        uint8_t* dst_row = (uint8_t*) c0;
        for (size_t row = 0; row < m_chunk; ++row) {
          svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row);
          svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row);
          svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row);
          svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row);
          r0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0, vs0), vmin), vmax);
          r1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1, vs1), vmin), vmax);
          r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vs2), vmin), vmax);
          r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3, vs3), vmin), vmax);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r0);
          svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r1);
          svst1_vnum_f32(pg1_lo, (float*) dst_row, 2, r2);
          svst1_vnum_f32(pg1_hi, (float*) dst_row, 3, r3);
          dst_row += cm_stride;
        }
        nc_rem = 0;
      }

      if (nc_rem != 0) {
        const svbool_t pg0_lo = svwhilelt_b32_u64(0, nc_rem);
        const svbool_t pg0_hi = svwhilelt_b32_u64(vl, nc_rem);
        const uint8_t* w0 = w_cur;

        const svfloat32_t vbias_0_lo = svld1_vnum_f32(pg0_lo, (const float*) w0, 0);
        const svfloat32_t vbias_0_hi = svld1_vnum_f32(pg0_hi, (const float*) w0, 1);
        w0 += 8 * vl;

        svzero_za();
        svmopa_za32_f32_m(0, pg_32_all, pg0_lo, f32_one, vbias_0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, f32_one, vbias_0_hi);

        const float* a0 = a_panel;
        size_t k = kc;
        for (; k >= 4; k -= 4) {
          const svuint32_t vw_01_lo = svld1ub_vnum_u32(pg0_lo, w0, 0);
          const svuint32_t vw_01_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
          const svuint32_t vw_23_lo = svld1ub_vnum_u32(pg0_lo, w0, 2);
          const svuint32_t vw_23_hi = svld1ub_vnum_u32(pg0_hi, w0, 3);
          w0 += 4 * vl;
          const svfloat32_t va0 = svld1_vnum_f32(pg_32_all, a0, 0);
          const svfloat32_t va1 = svld1_vnum_f32(pg_32_all, a0, 2);
          const svfloat32_t va2 = svld1_vnum_f32(pg_32_all, a0, 4);
          const svfloat32_t va3 = svld1_vnum_f32(pg_32_all, a0, 6);
          a0 += 8 * vl;
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01_hi, 4)));
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va2, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va3, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23_hi, 4)));
        }
        if (k >= 2) {
          const svuint32_t vw_lo = svld1ub_vnum_u32(pg0_lo, w0, 0);
          const svuint32_t vw_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
          w0 += 2 * vl;
          const svfloat32_t va0 = svld1_vnum_f32(pg_32_all, a0, 0);
          const svfloat32_t va1 = svld1_vnum_f32(pg_32_all, a0, 2);
          a0 += 4 * vl;
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_hi, 0xF)));
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_lo, 4)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va1, svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_hi, 4)));
          k -= 2;
        }
        if (k != 0) {
          const svuint32_t vw_lo = svld1ub_vnum_u32(pg0_lo, w0, 0);
          const svuint32_t vw_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
          w0 += 2 * vl;
          const svfloat32_t va0 = svld1_f32(pg_32_all, a0);
          svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_lo, 0xF)));
          svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0, svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_hi, 0xF)));
        }

        const svfloat32_t vs0 = svld1_vnum_f32(pg0_lo, (const float*) w0, 0);
        const svfloat32_t vs1 = svld1_vnum_f32(pg0_hi, (const float*) w0, 1);
        uint8_t* dst_row = (uint8_t*) c0;
        size_t row = 0;
        for (; row + 4 <= m_chunk; row += 4) {
          svfloat32_t r0_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 0);
          svfloat32_t r0_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 0);
          svfloat32_t r1_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 1);
          svfloat32_t r1_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 1);
          svfloat32_t r2_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 2);
          svfloat32_t r2_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 2);
          svfloat32_t r3_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 3);
          svfloat32_t r3_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 3);

          r0_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_0, vs0), vmin), vmax);
          r0_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_1, vs1), vmin), vmax);
          r1_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_0, vs0), vmin), vmax);
          r1_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_1, vs1), vmin), vmax);
          r2_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_0, vs0), vmin), vmax);
          r2_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_1, vs1), vmin), vmax);
          r3_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_0, vs0), vmin), vmax);
          r3_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_1, vs1), vmin), vmax);

          svst1_vnum_f32(pg0_lo, (float*) dst_row, 0, r0_0);
          svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r0_1);
          dst_row += cm_stride;
          svst1_vnum_f32(pg0_lo, (float*) dst_row, 0, r1_0);
          svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r1_1);
          dst_row += cm_stride;
          svst1_vnum_f32(pg0_lo, (float*) dst_row, 0, r2_0);
          svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r2_1);
          dst_row += cm_stride;
          svst1_vnum_f32(pg0_lo, (float*) dst_row, 0, r3_0);
          svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r3_1);
          dst_row += cm_stride;
        }
        for (; row < m_chunk; ++row) {
          svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row);
          svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row);
          r0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0, vs0), vmin), vmax);
          r1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1, vs1), vmin), vmax);
          svst1_vnum_f32(pg0_lo, (float*) dst_row, 0, r0);
          svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r1);
          dst_row += cm_stride;
        }
      }
      continue;
    }

    while (nc_rem >= nr) {
      const uint8_t* w0 = w_cur;
      w_cur += w_block_size;

#ifdef PREFETCH_W_MATRIX
      const uint8_t* pf0 = NULL;
      if (need_prfm) {
        if (nc_rem > nr) {
          pf0 = w_cur;
        } else if (m_start + 2 * vl < mr) {
          pf0 = w_base;
        }
        if (pf0 != NULL) {
          __builtin_prefetch(pf0, 0, 2);
#if !defined(__APPLE__)
          __builtin_prefetch(pf0 + 64, 0, 2);
#endif
          pf0 += 8 * vl;
        }
      }
#endif

      const svfloat32_t vbias_lo = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
      const svfloat32_t vbias_hi = svld1_vnum_f32(pg_32_all, (const float*) w0, 1);
      w0 += 8 * vl;

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, vbias_lo);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, vbias_hi);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, vbias_lo);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, f32_one, vbias_hi);

      const float* a0 = a_panel;
      size_t k = kc;
      size_t k_chunks = k >> 2;
      if (k_chunks != 0) {
        svuint32_t vw_01_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
        svuint32_t vw_01_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
        svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        svfloat32_t va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
        svfloat32_t va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);

        for (--k_chunks; k_chunks != 0; --k_chunks) {
          const svuint32_t vw_23_lo = svld1ub_vnum_u32(pg_32_all, w0, 2);
          const svuint32_t vw_23_hi = svld1ub_vnum_u32(pg_32_all, w0, 3);
          const svfloat32_t va2_0 = svld1_vnum_f32(pg_32_all, a0, 4);
          const svfloat32_t va2_1 = svld1_vnum_f32(pg_32_all, a0, 5);
          const svfloat32_t va3_0 = svld1_vnum_f32(pg_32_all, a0, 6);
          const svfloat32_t va3_1 = svld1_vnum_f32(pg_32_all, a0, 7);
          w0 += 4 * vl;
          a0 += 8 * vl;

          const svfloat32_t vb0_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01_lo, 0xF));
          const svfloat32_t vb0_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01_hi, 0xF));
          const svfloat32_t vb1_lo = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01_lo, 4));
          const svfloat32_t vb1_hi = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01_hi, 4));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0_0, vb0_lo);
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0_0, vb0_hi);
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0_1, vb0_lo);
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0_1, vb0_hi);
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1_0, vb1_lo);
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1_0, vb1_hi);
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1_1, vb1_lo);
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1_1, vb1_hi);

          vw_01_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
          vw_01_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
          va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
          va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
          va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
          va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);

          const svfloat32_t vb2_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23_lo, 0xF));
          const svfloat32_t vb2_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23_hi, 0xF));
          const svfloat32_t vb3_lo = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23_lo, 4));
          const svfloat32_t vb3_hi = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23_hi, 4));
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va2_0, vb2_lo);
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va2_0, vb2_hi);
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va2_1, vb2_lo);
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va2_1, vb2_hi);
          svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va3_0, vb3_lo);
          svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va3_0, vb3_hi);
          svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va3_1, vb3_lo);
          svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va3_1, vb3_hi);

#ifdef PREFETCH_W_MATRIX
          if (pf0 != NULL) {
            __builtin_prefetch(pf0, 0, 2);
            pf0 += 4 * vl;
          }
#endif
        }

        const svuint32_t vw_23_lo = svld1ub_vnum_u32(pg_32_all, w0, 2);
        const svuint32_t vw_23_hi = svld1ub_vnum_u32(pg_32_all, w0, 3);
        const svfloat32_t va2_0 = svld1_vnum_f32(pg_32_all, a0, 4);
        const svfloat32_t va2_1 = svld1_vnum_f32(pg_32_all, a0, 5);
        const svfloat32_t va3_0 = svld1_vnum_f32(pg_32_all, a0, 6);
        const svfloat32_t va3_1 = svld1_vnum_f32(pg_32_all, a0, 7);
        w0 += 4 * vl;
        a0 += 8 * vl;

        const svfloat32_t vb0_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01_lo, 0xF));
        const svfloat32_t vb0_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01_hi, 0xF));
        const svfloat32_t vb1_lo = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01_lo, 4));
        const svfloat32_t vb1_hi = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01_hi, 4));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0_1, vb0_hi);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1_0, vb1_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1_0, vb1_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1_1, vb1_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1_1, vb1_hi);

        const svfloat32_t vb2_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23_lo, 0xF));
        const svfloat32_t vb2_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23_hi, 0xF));
        const svfloat32_t vb3_lo = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23_lo, 4));
        const svfloat32_t vb3_hi = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23_hi, 4));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va2_0, vb2_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va2_0, vb2_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va2_1, vb2_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va2_1, vb2_hi);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va3_0, vb3_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va3_0, vb3_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va3_1, vb3_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va3_1, vb3_hi);
        k &= 3;
      }

      if (k >= 2) {
        const svuint32_t vw_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
        w0 += 2 * vl;
        const svfloat32_t vb0_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_lo, 0xF));
        const svfloat32_t vb0_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_hi, 0xF));
        const svfloat32_t vb1_lo = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_lo, 4));
        const svfloat32_t vb1_hi = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_hi, 4));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        const svfloat32_t va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
        const svfloat32_t va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);
        a0 += 4 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0_1, vb0_hi);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1_0, vb1_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va1_0, vb1_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1_1, vb1_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1_1, vb1_hi);
        k -= 2;
      }

      if (k != 0) {
        const svuint32_t vw_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw_hi = svld1ub_vnum_u32(pg_32_all, w0, 1);
        w0 += 2 * vl;
        const svfloat32_t vb0_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_lo, 0xF));
        const svfloat32_t vb0_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_hi, 0xF));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va0_1, vb0_hi);
      }

      const svfloat32_t vscale_0 = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
      const svfloat32_t vscale_1 = svld1_vnum_f32(pg_32_all, (const float*) w0, 1);

      uint8_t* dst_row = (uint8_t*) c0;
      for (size_t row = 0; row < vl; row += 4) {
        svfloat32_t r0_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 0);
        svfloat32_t r0_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 0);
        svfloat32_t r1_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 1);
        svfloat32_t r1_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 1);
        svfloat32_t r2_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 2);
        svfloat32_t r2_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 2);
        svfloat32_t r3_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 3);
        svfloat32_t r3_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 3);

        r0_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_0, vscale_0), vmin), vmax);
        r0_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_1, vscale_1), vmin), vmax);
        r1_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_0, vscale_0), vmin), vmax);
        r1_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_1, vscale_1), vmin), vmax);
        r2_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_0, vscale_0), vmin), vmax);
        r2_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_1, vscale_1), vmin), vmax);
        r3_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_0, vscale_0), vmin), vmax);
        r3_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_1, vscale_1), vmin), vmax);

        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r0_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r0_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r1_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r1_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r2_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r2_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r3_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r3_1);
        dst_row += cm_stride;
      }

      const size_t rows1 = m_chunk - vl;
      size_t row = 0;
      for (; row + 4 <= rows1; row += 4) {
        svfloat32_t r0_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 0);
        svfloat32_t r0_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row + 0);
        svfloat32_t r1_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 1);
        svfloat32_t r1_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row + 1);
        svfloat32_t r2_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 2);
        svfloat32_t r2_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row + 2);
        svfloat32_t r3_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 3);
        svfloat32_t r3_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row + 3);

        r0_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_0, vscale_0), vmin), vmax);
        r0_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_1, vscale_1), vmin), vmax);
        r1_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_0, vscale_0), vmin), vmax);
        r1_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_1, vscale_1), vmin), vmax);
        r2_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_0, vscale_0), vmin), vmax);
        r2_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_1, vscale_1), vmin), vmax);
        r3_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_0, vscale_0), vmin), vmax);
        r3_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_1, vscale_1), vmin), vmax);

        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r0_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r0_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r1_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r1_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r2_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r2_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r3_0);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r3_1);
        dst_row += cm_stride;
      }
      for (; row < rows1; ++row) {
        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row);
        svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row);
        r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vscale_0), vmin), vmax);
        r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3, vscale_1), vmin), vmax);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r2);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 1, r3);
        dst_row += cm_stride;
      }

      c0 += nr;
      nc_rem -= nr;
    }

    if (nc_rem > vl) {
      const svbool_t pg0_hi = svwhilelt_b32_u64(vl, nc_rem);
      const uint8_t* w0 = w_cur;

      const svfloat32_t vbias_lo = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
      const svfloat32_t vbias_hi = svld1_vnum_f32(pg0_hi, (const float*) w0, 1);
      w0 += 8 * vl;

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, vbias_lo);
      svmopa_za32_f32_m(1, pg_32_all, pg0_hi, f32_one, vbias_hi);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, vbias_lo);
      svmopa_za32_f32_m(3, pg_32_all, pg0_hi, f32_one, vbias_hi);

      const float* a0 = a_panel;
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const svuint32_t vw_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
        w0 += 2 * vl;
        const svfloat32_t vb0_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_lo, 0xF));
        const svfloat32_t vb0_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_hi, 0xF));
        const svfloat32_t vb1_lo = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_lo, 4));
        const svfloat32_t vb1_hi = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_hi, 4));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        const svfloat32_t va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
        const svfloat32_t va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);
        a0 += 4 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg0_hi, va0_1, vb0_hi);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va1_0, vb1_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va1_0, vb1_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1_1, vb1_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg0_hi, va1_1, vb1_hi);
      }
      if (k != 0) {
        const svuint32_t vw_lo = svld1ub_vnum_u32(pg_32_all, w0, 0);
        const svuint32_t vw_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
        w0 += 2 * vl;
        const svfloat32_t vb0_lo = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_lo, 0xF));
        const svfloat32_t vb0_hi = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_hi, 0xF));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg0_hi, va0_1, vb0_hi);
      }

      const svfloat32_t vscale_0 = svld1_vnum_f32(pg_32_all, (const float*) w0, 0);
      const svfloat32_t vscale_1 = svld1_vnum_f32(pg0_hi, (const float*) w0, 1);

      uint8_t* dst_row = (uint8_t*) c0;
      for (size_t row = 0; row < vl; row += 4) {
        svfloat32_t r0_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 0);
        svfloat32_t r0_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 0);
        svfloat32_t r1_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 1);
        svfloat32_t r1_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 1);
        svfloat32_t r2_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 2);
        svfloat32_t r2_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 2);
        svfloat32_t r3_0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 3);
        svfloat32_t r3_1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, row + 3);

        r0_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_0, vscale_0), vmin), vmax);
        r0_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0_1, vscale_1), vmin), vmax);
        r1_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_0, vscale_0), vmin), vmax);
        r1_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1_1, vscale_1), vmin), vmax);
        r2_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_0, vscale_0), vmin), vmax);
        r2_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2_1, vscale_1), vmin), vmax);
        r3_0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_0, vscale_0), vmin), vmax);
        r3_1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3_1, vscale_1), vmin), vmax);

        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r0_0);
        svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r0_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r1_0);
        svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r1_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r2_0);
        svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r2_1);
        dst_row += cm_stride;
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r3_0);
        svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r3_1);
        dst_row += cm_stride;
      }

      const size_t rows1 = m_chunk - vl;
      for (size_t row = 0; row < rows1; ++row) {
        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row);
        svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, row);
        r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vscale_0), vmin), vmax);
        r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3, vscale_1), vmin), vmax);
        svst1_vnum_f32(pg_32_all, (float*) dst_row, 0, r2);
        svst1_vnum_f32(pg0_hi, (float*) dst_row, 1, r3);
        dst_row += cm_stride;
      }
      nc_rem = 0;
    }

    if (nc_rem != 0) {
      const svbool_t pg_n = svwhilelt_b32_u64(0, nc_rem);
      const uint8_t* w0 = w_cur;

      const svfloat32_t vbias = svld1_f32(pg_n, (const float*) w0);
      w0 += 8 * vl;
      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_n, f32_one, vbias);
      svmopa_za32_f32_m(2, pg_32_all, pg_n, f32_one, vbias);

      const float* a0 = a_panel;
      size_t k = kc;
      size_t k_chunks = k >> 2;
      if (k_chunks != 0) {
        svuint32_t vw_01 = svld1ub_vnum_u32(pg_n, w0, 0);
        svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        svfloat32_t va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
        svfloat32_t va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);

        for (--k_chunks; k_chunks != 0; --k_chunks) {
          const svuint32_t vw_23 = svld1ub_vnum_u32(pg_n, w0, 2);
          const svfloat32_t va2_0 = svld1_vnum_f32(pg_32_all, a0, 4);
          const svfloat32_t va2_1 = svld1_vnum_f32(pg_32_all, a0, 5);
          const svfloat32_t va3_0 = svld1_vnum_f32(pg_32_all, a0, 6);
          const svfloat32_t va3_1 = svld1_vnum_f32(pg_32_all, a0, 7);
          w0 += 4 * vl;
          a0 += 8 * vl;

          const svfloat32_t vb0 = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01, 0xF));
          const svfloat32_t vb1 = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01, 4));
          svmopa_za32_f32_m(0, pg_32_all, pg_n, va0_0, vb0);
          svmopa_za32_f32_m(2, pg_32_all, pg_n, va0_1, vb0);
          svmopa_za32_f32_m(0, pg_32_all, pg_n, va1_0, vb1);
          svmopa_za32_f32_m(2, pg_32_all, pg_n, va1_1, vb1);

          vw_01 = svld1ub_vnum_u32(pg_n, w0, 0);
          va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
          va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
          va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
          va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);

          const svfloat32_t vb2 = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23, 0xF));
          const svfloat32_t vb3 = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23, 4));
          svmopa_za32_f32_m(0, pg_32_all, pg_n, va2_0, vb2);
          svmopa_za32_f32_m(2, pg_32_all, pg_n, va2_1, vb2);
          svmopa_za32_f32_m(0, pg_32_all, pg_n, va3_0, vb3);
          svmopa_za32_f32_m(2, pg_32_all, pg_n, va3_1, vb3);
        }

        const svuint32_t vw_23 = svld1ub_vnum_u32(pg_n, w0, 2);
        const svfloat32_t va2_0 = svld1_vnum_f32(pg_32_all, a0, 4);
        const svfloat32_t va2_1 = svld1_vnum_f32(pg_32_all, a0, 5);
        const svfloat32_t va3_0 = svld1_vnum_f32(pg_32_all, a0, 6);
        const svfloat32_t va3_1 = svld1_vnum_f32(pg_32_all, a0, 7);
        w0 += 4 * vl;
        a0 += 8 * vl;

        const svfloat32_t vb0 = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_01, 0xF));
        const svfloat32_t vb1 = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_01, 4));
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va0_0, vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va0_1, vb0);
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va1_0, vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va1_1, vb1);

        const svfloat32_t vb2 = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw_23, 0xF));
        const svfloat32_t vb3 = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw_23, 4));
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va2_0, vb2);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va2_1, vb2);
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va3_0, vb3);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va3_1, vb3);
        k &= 3;
      }

      if (k >= 2) {
        const svuint32_t vw = svld1ub_vnum_u32(pg_n, w0, 0);
        w0 += 2 * vl;
        const svfloat32_t vb0 = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw, 0xF));
        const svfloat32_t vb1 = svtbl_f32(vlut, svlsr_n_u32_x(pg_32_all, vw, 4));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        const svfloat32_t va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
        const svfloat32_t va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);
        a0 += 4 * vl;
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va0_0, vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va0_1, vb0);
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va1_0, vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va1_1, vb1);
        k -= 2;
      }

      if (k != 0) {
        const svuint32_t vw = svld1ub_vnum_u32(pg_n, w0, 0);
        w0 += 2 * vl;
        const svfloat32_t vb0 = svtbl_f32(vlut, svand_n_u32_x(pg_32_all, vw, 0xF));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        svmopa_za32_f32_m(0, pg_32_all, pg_n, va0_0, vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_n, va0_1, vb0);
      }

      const svfloat32_t vscale = svld1_f32(pg_n, (const float*) w0);
      uint8_t* dst_row = (uint8_t*) c0;
      for (size_t row = 0; row < vl; row += 4) {
        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 0);
        svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 1);
        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 2);
        svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, row + 3);
        r0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0, vscale), vmin), vmax);
        r1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1, vscale), vmin), vmax);
        r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vscale), vmin), vmax);
        r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3, vscale), vmin), vmax);
        svst1_f32(pg_n, (float*) dst_row, r0);
        dst_row += cm_stride;
        svst1_f32(pg_n, (float*) dst_row, r1);
        dst_row += cm_stride;
        svst1_f32(pg_n, (float*) dst_row, r2);
        dst_row += cm_stride;
        svst1_f32(pg_n, (float*) dst_row, r3);
        dst_row += cm_stride;
      }

      const size_t rows1 = m_chunk - vl;
      size_t row = 0;
      for (; row + 4 <= rows1; row += 4) {
        svfloat32_t r0 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 0);
        svfloat32_t r1 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 1);
        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 2);
        svfloat32_t r3 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row + 3);
        r0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r0, vscale), vmin), vmax);
        r1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r1, vscale), vmin), vmax);
        r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vscale), vmin), vmax);
        r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r3, vscale), vmin), vmax);
        svst1_f32(pg_n, (float*) dst_row, r0);
        dst_row += cm_stride;
        svst1_f32(pg_n, (float*) dst_row, r1);
        dst_row += cm_stride;
        svst1_f32(pg_n, (float*) dst_row, r2);
        dst_row += cm_stride;
        svst1_f32(pg_n, (float*) dst_row, r3);
        dst_row += cm_stride;
      }
      for (; row < rows1; ++row) {
        svfloat32_t r2 = svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, row);
        r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svmul_f32_x(pg_32_all, r2, vscale), vmin), vmax);
        svst1_f32(pg_n, (float*) dst_row, r2);
        dst_row += cm_stride;
      }
    }
  }
}

// Generic SVL < 512-bit (vl < 16) fallback using arithmetic nibble subtract.
__arm_locally_streaming __arm_new("za")
static void pf32_qc4w_gemm_32x32_sme_generic(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max,
    int32_t kernel_zero_point)
{
  const size_t vl = svcntw();
  const size_t nr = 2 * vl;
  const size_t w_block_size = 8 * vl + ((kc + 1) / 2) * 2 * vl + 8 * vl;
  const svbool_t pg_32_all = svptrue_b32();
  const svfloat32_t f32_one = svdup_n_f32(1.0f);
  const svfloat32_t vmin = svdup_n_f32(min);
  const svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  const uint8_t* w_base = (const uint8_t*) w;

  for (size_t m_start = 0; m_start < mr; m_start += 2 * vl) {
    const size_t m_chunk = (mr - m_start >= 2 * vl) ? (2 * vl) : (mr - m_start);
    float* c0 = (float*) ((uintptr_t) c + m_start * cm_stride);
    const uint8_t* w_cur = w_base;
    size_t nc_rem = nc;

    while (nc_rem != 0) {
      const svbool_t pg0_lo = svwhilelt_b32_u64(0, nc_rem);
      const svbool_t pg0_hi = svwhilelt_b32_u64(vl, nc_rem);
      const uint8_t* w0 = w_cur;
      w_cur += w_block_size;

      const svfloat32_t vbias_0 = svld1_vnum_f32(pg0_lo, (const float*) w0, 0);
      const svfloat32_t vbias_1 = svld1_vnum_f32(pg0_hi, (const float*) w0, 1);
      w0 += 8 * vl;

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg0_lo, f32_one, vbias_0);
      svmopa_za32_f32_m(1, pg_32_all, pg0_hi, f32_one, vbias_1);
      svmopa_za32_f32_m(2, pg_32_all, pg0_lo, f32_one, vbias_0);
      svmopa_za32_f32_m(3, pg_32_all, pg0_hi, f32_one, vbias_1);

      const float* a0 = a_base + m_start * kc;
      size_t k = kc;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (; k >= 2; k -= 2) {
        const svuint32_t vw_lo = svld1ub_vnum_u32(pg0_lo, w0, 0);
        const svuint32_t vw_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
        w0 += 2 * vl;
        const svfloat32_t vb0_lo = svcvt_f32_s32_x(
            pg_32_all, svsub_n_s32_x(pg_32_all, svreinterpret_s32_u32(svand_n_u32_x(pg_32_all, vw_lo, 0xF)), kernel_zero_point));
        const svfloat32_t vb0_hi = svcvt_f32_s32_x(
            pg_32_all, svsub_n_s32_x(pg_32_all, svreinterpret_s32_u32(svand_n_u32_x(pg_32_all, vw_hi, 0xF)), kernel_zero_point));
        const svfloat32_t vb1_lo = svcvt_f32_s32_x(
            pg_32_all, svsub_n_s32_x(pg_32_all, svreinterpret_s32_u32(svlsr_n_u32_x(pg_32_all, vw_lo, 4)), kernel_zero_point));
        const svfloat32_t vb1_hi = svcvt_f32_s32_x(
            pg_32_all, svsub_n_s32_x(pg_32_all, svreinterpret_s32_u32(svlsr_n_u32_x(pg_32_all, vw_hi, 4)), kernel_zero_point));

        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        const svfloat32_t va1_0 = svld1_vnum_f32(pg_32_all, a0, 2);
        const svfloat32_t va1_1 = svld1_vnum_f32(pg_32_all, a0, 3);
        a0 += 4 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg0_lo, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg0_hi, va0_1, vb0_hi);
        svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va1_0, vb1_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va1_0, vb1_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg0_lo, va1_1, vb1_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg0_hi, va1_1, vb1_hi);
      }

      if (k != 0) {
        const svuint32_t vw_lo = svld1ub_vnum_u32(pg0_lo, w0, 0);
        const svuint32_t vw_hi = svld1ub_vnum_u32(pg0_hi, w0, 1);
        w0 += 2 * vl;
        const svfloat32_t vb0_lo = svcvt_f32_s32_x(
            pg_32_all, svsub_n_s32_x(pg_32_all, svreinterpret_s32_u32(svand_n_u32_x(pg_32_all, vw_lo, 0xF)), kernel_zero_point));
        const svfloat32_t vb0_hi = svcvt_f32_s32_x(
            pg_32_all, svsub_n_s32_x(pg_32_all, svreinterpret_s32_u32(svand_n_u32_x(pg_32_all, vw_hi, 0xF)), kernel_zero_point));
        const svfloat32_t va0_0 = svld1_vnum_f32(pg_32_all, a0, 0);
        const svfloat32_t va0_1 = svld1_vnum_f32(pg_32_all, a0, 1);
        svmopa_za32_f32_m(0, pg_32_all, pg0_lo, va0_0, vb0_lo);
        svmopa_za32_f32_m(1, pg_32_all, pg0_hi, va0_0, vb0_hi);
        svmopa_za32_f32_m(2, pg_32_all, pg0_lo, va0_1, vb0_lo);
        svmopa_za32_f32_m(3, pg_32_all, pg0_hi, va0_1, vb0_hi);
      }

      const svfloat32_t vscale_0 = svld1_vnum_f32(pg0_lo, (const float*) w0, 0);
      const svfloat32_t vscale_1 = svld1_vnum_f32(pg0_hi, (const float*) w0, 1);

      const size_t m_rows_0 = (m_chunk < vl) ? m_chunk : vl;
      float* c_inc = c0;
      XNN_PRAGMA_CLANG("clang loop unroll(disable)")
      for (size_t i = 0; i < m_rows_0; ++i) {
        svfloat32_t r0 = svmul_f32_x(pg_32_all, svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i), vscale_0);
        svfloat32_t r1 = svmul_f32_x(pg_32_all, svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i), vscale_1);
        r0 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, r0, vmin), vmax);
        r1 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, r1, vmin), vmax);
        svst1_f32(pg0_lo, c_inc, r0);
        svst1_f32(pg0_hi, c_inc + vl, r1);
        c_inc = (float*) ((uintptr_t) c_inc + cm_stride);
      }

      if (m_chunk > vl) {
        const size_t m_rows_1 = m_chunk - vl;
        c_inc = (float*) ((uintptr_t) c0 + vl * cm_stride);
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows_1; ++i) {
          svfloat32_t r2 = svmul_f32_x(pg_32_all, svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i), vscale_0);
          svfloat32_t r3 = svmul_f32_x(pg_32_all, svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i), vscale_1);
          r2 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, r2, vmin), vmax);
          r3 = svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, r3, vmin), vmax);
          svst1_f32(pg0_lo, c_inc, r2);
          svst1_f32(pg0_hi, c_inc + vl, r3);
          c_inc = (float*) ((uintptr_t) c_inc + cm_stride);
        }
      }

      if (nc_rem > nr) {
        c0 += nr;
        nc_rem -= nr;
      } else {
        nc_rem = 0;
      }
    }
  }
}

void xnn_pf32_qc4w_gemm_minmax_ukernel_32x32__sme(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* restrict params)
{
  if XNN_UNLIKELY(mr == 0 || nc == 0) {
    return;
  }
  assert(cn_stride == sizeof(float));
  (void) cn_stride;
  const size_t kc_elems = kc / sizeof(float);
  const size_t vl = svcntsw();
  if XNN_UNLIKELY(vl < 16) {
    pf32_qc4w_gemm_32x32_sme_generic(
        mr, nc, kc_elems, a, w, c, cm_stride,
        params->scalar.min, params->scalar.max, params->scalar.kernel_zero_point);
    return;
  }
  pf32_qc4w_gemm_32x32_sme_lut_impl(
      mr, nc, kc_elems, a, w, c, cm_stride,
      params->scalar.min, params->scalar.max, params->scalar.kernel_zero_point);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
