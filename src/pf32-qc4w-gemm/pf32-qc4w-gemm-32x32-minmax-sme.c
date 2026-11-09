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
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_line_size = 128;
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
  const size_t prfm_dist = 2048;
#else
  const size_t prfm_line_size = 64;
  const size_t prfm_min_w_size = (1024 * 1024);
  const size_t prfm_dist = 2048;
#endif
#endif
  size_t vl = svcntw();

  svbool_t pg_32_all = svptrue_b32();
  const svfloat32_t vlut = svcvt_f32_s32_x(pg_32_all, svsub_n_s32_x(pg_32_all, svindex_s32(0, 1), kernel_zero_point));

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  const uint8_t* w_base = (const uint8_t*) w;

#ifdef PREFETCH_W_MATRIX
  const size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
  const size_t w_block_size = 8 * vl + ((kc + 1) / 2) * 2 * vl + 8 * vl;
  const size_t total_w_size = (nc_rup_nr / (2 * vl)) * w_block_size;
  const size_t num_m_tiles = (mr + 2 * vl - 1) / (2 * vl);
  const bool need_prfm = (total_w_size > prfm_min_w_size);
  const size_t prfm_end = need_prfm ? (num_m_tiles * total_w_size) : 0;
  const size_t prfm_init = (prfm_end < prfm_dist) ? prfm_end : prfm_dist;
  size_t prfm_pos = 0;
  size_t prfm_off = 0;
  size_t prfm_tile_base = 0;
  while (prfm_pos < prfm_init) {
    __builtin_prefetch(w_base + prfm_off, 0, 2);
    if (prfm_off + prfm_line_size < total_w_size) {
      prfm_pos += prfm_line_size;
      prfm_off += prfm_line_size;
    } else {
      prfm_pos += total_w_size - prfm_off;
      prfm_off = 0;
    }
  }
#endif

  for (size_t m_start = 0; m_start < mr; m_start += 2 * vl) {
    size_t m_chunk = (mr - m_start >= 2 * vl) ? 2 * vl : (mr - m_start);

    float* c_base = (float*) ((uintptr_t) c + m_start * cm_stride);
    const uint8_t* w_ptr = w_base;
    size_t nc_rem = nc;
    float* c0 = c_base;

    while (nc_rem > vl) {
      svfloat32x2_t vb_bias_x2 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 1));
      w_ptr += 2 * vl * sizeof(float);

      svfloat32_t f32_bias_0 = svget2_f32(vb_bias_x2, 0);
      svfloat32_t f32_bias_1 = svget2_f32(vb_bias_x2, 1);
      svfloat32_t f32_one = svdup_n_f32(1.0f);

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, f32_bias_0);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, f32_bias_1);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, f32_bias_0);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, f32_one, f32_bias_1);

      size_t k = kc;
      const float* a0 = a_base + m_start * kc;
      const uint8_t* qw_ptr = w_ptr;

      size_t k_chunks = k / 4;
      while (k_chunks > 0) {
        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 1);
        const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 2);
        const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 3);
        qw_ptr += 4 * vl;
        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_0_hi = svlsr_n_u32_x(pg_32_all, vw01_0, 4);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svuint32_t vi01_1_hi = svlsr_n_u32_x(pg_32_all, vw01_1, 4);
        const svuint32_t vi23_0_lo = svand_n_u32_x(pg_32_all, vw23_0, 0xF);
        const svuint32_t vi23_0_hi = svlsr_n_u32_x(pg_32_all, vw23_0, 4);
        const svuint32_t vi23_1_lo = svand_n_u32_x(pg_32_all, vw23_1, 0xF);
        const svuint32_t vi23_1_hi = svlsr_n_u32_x(pg_32_all, vw23_1, 4);
        const svfloat32x4_t vb_x4_0 = svcreate4_f32(
            svtbl_f32(vlut, vi01_0_lo),
            svtbl_f32(vlut, vi01_1_lo),
            svtbl_f32(vlut, vi01_0_hi),
            svtbl_f32(vlut, vi01_1_hi));
        const svfloat32x4_t vb_x4_1 = svcreate4_f32(
            svtbl_f32(vlut, vi23_0_lo),
            svtbl_f32(vlut, vi23_1_lo),
            svtbl_f32(vlut, vi23_0_hi),
            svtbl_f32(vlut, vi23_1_hi));

#ifdef PREFETCH_W_MATRIX
        {
          size_t prfm_tgt = prfm_tile_base + (size_t) (qw_ptr - w_base) + prfm_dist;
          if (prfm_tgt > prfm_end) {
            prfm_tgt = prfm_end;
          }
          while (prfm_pos < prfm_tgt) {
            __builtin_prefetch(w_base + prfm_off, 0, 2);
            if (prfm_off + prfm_line_size < total_w_size) {
              prfm_pos += prfm_line_size;
              prfm_off += prfm_line_size;
            } else {
              prfm_pos += total_w_size - prfm_off;
              prfm_off = 0;
            }
          }
        }
#endif

        const svfloat32x4_t va_x4_0 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        const svfloat32x4_t va_x4_1 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 0),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 1),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 2),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 3));
        a0 += 8 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 3));

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 3));

        k_chunks--;
      }
      k %= 4;

      if (k >= 2) {
        const svfloat32x4_t va_x4 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        a0 += 4 * vl;

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 1);
        qw_ptr += 2 * vl;

        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_0_hi = svlsr_n_u32_x(pg_32_all, vw01_0, 4);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svuint32_t vi01_1_hi = svlsr_n_u32_x(pg_32_all, vw01_1, 4);
        const svfloat32x4_t vb_x4 = svcreate4_f32(
            svtbl_f32(vlut, vi01_0_lo),
            svtbl_f32(vlut, vi01_1_lo),
            svtbl_f32(vlut, vi01_0_hi),
            svtbl_f32(vlut, vi01_1_hi));

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget4_f32(vb_x4, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget4_f32(vb_x4, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget4_f32(vb_x4, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget4_f32(vb_x4, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget4_f32(vb_x4, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget4_f32(vb_x4, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget4_f32(vb_x4, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget4_f32(vb_x4, 3));

        k -= 2;
      }
      if (k != 0) {
        svfloat32x2_t va_x2_0 = svcreate2_f32(svld1_vnum_f32(pg_32_all, a0, 0), svld1_vnum_f32(pg_32_all, a0, 1));
        a0 += 2 * vl;
        svfloat32_t va0 = svget2_f32(va_x2_0, 0);
        svfloat32_t va1 = svget2_f32(va_x2_0, 1);

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 1);
        qw_ptr += 2 * vl;

        const svuint32_t vi01_0_lo = svand_n_u32_x(pg_32_all, vw01_0, 0xF);
        const svuint32_t vi01_1_lo = svand_n_u32_x(pg_32_all, vw01_1, 0xF);
        const svfloat32_t vb0 = svtbl_f32(vlut, vi01_0_lo);
        const svfloat32_t vb1 = svtbl_f32(vlut, vi01_1_lo);

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, vb0);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, vb0);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, vb1);
      }

      w_ptr = qw_ptr;

      svfloat32x2_t vb_scale_x2 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 1));
      w_ptr += 2 * vl * sizeof(float);

#ifdef PREFETCH_W_MATRIX
      {
        size_t prfm_tgt = prfm_tile_base + (size_t) (w_ptr - w_base) + prfm_dist;
        if (prfm_tgt > prfm_end) {
          prfm_tgt = prfm_end;
        }
        while (prfm_pos < prfm_tgt) {
          __builtin_prefetch(w_base + prfm_off, 0, 2);
          if (prfm_off + prfm_line_size < total_w_size) {
            prfm_pos += prfm_line_size;
            prfm_off += prfm_line_size;
          } else {
            prfm_pos += total_w_size - prfm_off;
            prfm_off = 0;
          }
        }
      }
#endif

      svfloat32_t vscale_0 = svget2_f32(vb_scale_x2, 0);
      svfloat32_t vscale_1 = svget2_f32(vb_scale_x2, 1);

      const svbool_t pg_n_0 = svwhilelt_b32_u64(0, nc_rem);
      const svbool_t pg_n_1 = svwhilelt_b32_u64(vl, nc_rem);

      {
        float* c_rows = c0;
        const size_t m_rows = (m_chunk < vl) ? m_chunk : vl;
        svfloat32x4_t z0_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 3));
        svfloat32x4_t z1_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z0_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 7));
          const svfloat32x4_t z1_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 7));
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z0_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 11));
            z1_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 11));
          }
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
        }
      }

      if (m_chunk > vl) {
        float* c_rows = (float*) ((uintptr_t) c0 + vl * cm_stride);
        const size_t m_rows = m_chunk - vl;
        svfloat32x4_t z2_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 3));
        svfloat32x4_t z3_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z2_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 7));
          const svfloat32x4_t z3_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 7));
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z2_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 11));
            z3_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 11));
          }
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
        }
      }

      if (nc_rem > 2 * vl) {
        c0 += 2 * vl;
        nc_rem -= 2 * vl;
      } else {
        nc_rem = 0;
      }
    }

    if (nc_rem != 0) {
      const svfloat32_t f32_bias = svld1_f32(pg_32_all, (const float*) w_ptr);
      w_ptr += 2 * vl * sizeof(float);
      const svfloat32_t f32_one = svdup_n_f32(1.0f);

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, f32_bias);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, f32_bias);

      const float* a0 = a_base + m_start * kc;
      const uint8_t* qw_ptr = w_ptr;
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw23 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 2);
        qw_ptr += 4 * vl;
        const svuint32_t vi01_lo = svand_n_u32_x(pg_32_all, vw01, 0xF);
        const svuint32_t vi01_hi = svlsr_n_u32_x(pg_32_all, vw01, 4);
        const svuint32_t vi23_lo = svand_n_u32_x(pg_32_all, vw23, 0xF);
        const svuint32_t vi23_hi = svlsr_n_u32_x(pg_32_all, vw23, 4);
        const svfloat32_t vb0 = svtbl_f32(vlut, vi01_lo);
        const svfloat32_t vb1 = svtbl_f32(vlut, vi01_hi);
        const svfloat32_t vb2 = svtbl_f32(vlut, vi23_lo);
        const svfloat32_t vb3 = svtbl_f32(vlut, vi23_hi);

#ifdef PREFETCH_W_MATRIX
        {
          size_t prfm_tgt = prfm_tile_base + (size_t) (qw_ptr - w_base) + prfm_dist;
          if (prfm_tgt > prfm_end) {
            prfm_tgt = prfm_end;
          }
          while (prfm_pos < prfm_tgt) {
            __builtin_prefetch(w_base + prfm_off, 0, 2);
            if (prfm_off + prfm_line_size < total_w_size) {
              prfm_pos += prfm_line_size;
              prfm_off += prfm_line_size;
            } else {
              prfm_pos += total_w_size - prfm_off;
              prfm_off = 0;
            }
          }
        }
#endif

        const svfloat32x4_t va_x4_0 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        const svfloat32x4_t va_x4_1 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 0),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 1),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 2),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 3));
        a0 += 8 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), vb0);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), vb1);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), vb2);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), vb2);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), vb3);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), vb3);
      }
      if (k >= 2) {
        const svfloat32x4_t va_x4 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        a0 += 4 * vl;
        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        qw_ptr += 2 * vl;
        const svuint32_t vi01_lo = svand_n_u32_x(pg_32_all, vw01, 0xF);
        const svuint32_t vi01_hi = svlsr_n_u32_x(pg_32_all, vw01, 4);
        const svfloat32_t vb0 = svtbl_f32(vlut, vi01_lo);
        const svfloat32_t vb1 = svtbl_f32(vlut, vi01_hi);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), vb0);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), vb1);
        k -= 2;
      }
      if (k != 0) {
        const svfloat32x2_t va_x2 = svcreate2_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1));
        a0 += 2 * vl;
        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        qw_ptr += 2 * vl;
        const svuint32_t vi01_lo = svand_n_u32_x(pg_32_all, vw01, 0xF);
        const svfloat32_t vb = svtbl_f32(vlut, vi01_lo);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget2_f32(va_x2, 0), vb);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget2_f32(va_x2, 1), vb);
      }

      w_ptr = qw_ptr;
      const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) w_ptr);
      w_ptr += 2 * vl * sizeof(float);

#ifdef PREFETCH_W_MATRIX
      {
        size_t prfm_tgt = prfm_tile_base + (size_t) (w_ptr - w_base) + prfm_dist;
        if (prfm_tgt > prfm_end) {
          prfm_tgt = prfm_end;
        }
        while (prfm_pos < prfm_tgt) {
          __builtin_prefetch(w_base + prfm_off, 0, 2);
          if (prfm_off + prfm_line_size < total_w_size) {
            prfm_pos += prfm_line_size;
            prfm_off += prfm_line_size;
          } else {
            prfm_pos += total_w_size - prfm_off;
            prfm_off = 0;
          }
        }
      }
#endif

      const svbool_t pg_n = svwhilelt_b32_u64(0, nc_rem);
      {
        float* c_rows = c0;
        const size_t m_rows = (m_chunk < vl) ? m_chunk : vl;
        svfloat32x4_t z0_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z0_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 7));
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z0_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 11));
          }
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
        }
      }

      if (m_chunk > vl) {
        float* c_rows = (float*) ((uintptr_t) c0 + vl * cm_stride);
        const size_t m_rows = m_chunk - vl;
        svfloat32x4_t z2_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z2_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 7));
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z2_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 11));
          }
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
        }
      }
    }
#ifdef PREFETCH_W_MATRIX
    prfm_tile_base += total_w_size;
#endif
  }
}

__arm_locally_streaming __arm_new("za")
static void pf32_qc4w_gemm_32x32_sme_fmagic_impl(
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
#ifdef PREFETCH_W_MATRIX
#if defined(__APPLE__)
  const size_t prfm_line_size = 128;
  const size_t prfm_min_w_size = (16 * 1024 * 1024);
  const size_t prfm_dist = 2048;
#else
  const size_t prfm_line_size = 64;
  const size_t prfm_min_w_size = (1024 * 1024);
  const size_t prfm_dist = 2048;
#endif
#endif
  size_t vl = svcntw();

  svbool_t pg_32_all = svptrue_b32();
  const svbool_t pg_16_all = svptrue_b16();
  const svuint32_t vmagic = svdup_n_u32(UINT32_C(0x64006400));
  const svuint32_t vmask = svdup_n_u32(UINT32_C(0x000F000F));
  const uint16_t magic_bias_bits = (uint16_t) (UINT16_C(0x6400) + kernel_zero_point);
  const svfloat16_t vmagic_bias = svreinterpret_f16_u16(svdup_n_u16(magic_bias_bits));

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  const uint8_t* w_base = (const uint8_t*) w;

#ifdef PREFETCH_W_MATRIX
  const size_t nc_rup_nr = ((nc + 2 * vl - 1) / (2 * vl)) * (2 * vl);
  const size_t w_block_size = 8 * vl + ((kc + 1) / 2) * 2 * vl + 8 * vl;
  const size_t total_w_size = (nc_rup_nr / (2 * vl)) * w_block_size;
  const size_t num_m_tiles = (mr + 2 * vl - 1) / (2 * vl);
  const bool need_prfm = (total_w_size > prfm_min_w_size);
  const size_t prfm_end = need_prfm ? (num_m_tiles * total_w_size) : 0;
  const size_t prfm_init = (prfm_end < prfm_dist) ? prfm_end : prfm_dist;
  size_t prfm_pos = 0;
  size_t prfm_off = 0;
  size_t prfm_tile_base = 0;
  while (prfm_pos < prfm_init) {
    __builtin_prefetch(w_base + prfm_off, 0, 2);
    if (prfm_off + prfm_line_size < total_w_size) {
      prfm_pos += prfm_line_size;
      prfm_off += prfm_line_size;
    } else {
      prfm_pos += total_w_size - prfm_off;
      prfm_off = 0;
    }
  }
#endif

  for (size_t m_start = 0; m_start < mr; m_start += 2 * vl) {
    size_t m_chunk = (mr - m_start >= 2 * vl) ? 2 * vl : (mr - m_start);

    float* c_base = (float*) ((uintptr_t) c + m_start * cm_stride);
    const uint8_t* w_ptr = w_base;
    size_t nc_rem = nc;
    float* c0 = c_base;

    while (nc_rem > vl) {
      svfloat32x2_t vb_bias_x2 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 1));
      w_ptr += 2 * vl * sizeof(float);

      svfloat32_t f32_bias_0 = svget2_f32(vb_bias_x2, 0);
      svfloat32_t f32_bias_1 = svget2_f32(vb_bias_x2, 1);
      svfloat32_t f32_one = svdup_n_f32(1.0f);

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, f32_bias_0);
      svmopa_za32_f32_m(1, pg_32_all, pg_32_all, f32_one, f32_bias_1);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, f32_bias_0);
      svmopa_za32_f32_m(3, pg_32_all, pg_32_all, f32_one, f32_bias_1);

      size_t k = kc;
      const float* a0 = a_base + m_start * kc;
      const uint8_t* qw_ptr = w_ptr;

      size_t k_chunks = k / 4;
      while (k_chunks > 0) {
        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 1);
        const svuint32_t vw23_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 2);
        const svuint32_t vw23_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 3);
        qw_ptr += 4 * vl;
        const svuint32_t vt01_0 = svbsl_u32(svsli_n_u32(vw01_0, vw01_0, 12), vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(svsli_n_u32(vw01_1, vw01_1, 12), vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        const svuint32_t vt23_0 = svbsl_u32(svsli_n_u32(vw23_0, vw23_0, 12), vmagic, vmask);
        const svfloat16_t vh23_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_0), vmagic_bias);
        const svuint32_t vt23_1 = svbsl_u32(svsli_n_u32(vw23_1, vw23_1, 12), vmagic, vmask);
        const svfloat16_t vh23_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23_1), vmagic_bias);
        const svfloat32x4_t vb_x4_0 = svcreate4_f32(
            svcvt_f32_f16_x(pg_32_all, vh01_0),
            svcvt_f32_f16_x(pg_32_all, vh01_1),
            svcvtlt_f32_f16_x(pg_32_all, vh01_0),
            svcvtlt_f32_f16_x(pg_32_all, vh01_1));
        const svfloat32x4_t vb_x4_1 = svcreate4_f32(
            svcvt_f32_f16_x(pg_32_all, vh23_0),
            svcvt_f32_f16_x(pg_32_all, vh23_1),
            svcvtlt_f32_f16_x(pg_32_all, vh23_0),
            svcvtlt_f32_f16_x(pg_32_all, vh23_1));

#ifdef PREFETCH_W_MATRIX
        {
          size_t prfm_tgt = prfm_tile_base + (size_t) (qw_ptr - w_base) + prfm_dist;
          if (prfm_tgt > prfm_end) {
            prfm_tgt = prfm_end;
          }
          while (prfm_pos < prfm_tgt) {
            __builtin_prefetch(w_base + prfm_off, 0, 2);
            if (prfm_off + prfm_line_size < total_w_size) {
              prfm_pos += prfm_line_size;
              prfm_off += prfm_line_size;
            } else {
              prfm_pos += total_w_size - prfm_off;
              prfm_off = 0;
            }
          }
        }
#endif

        const svfloat32x4_t va_x4_0 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        const svfloat32x4_t va_x4_1 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 0),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 1),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 2),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 3));
        a0 += 8 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget4_f32(vb_x4_0, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget4_f32(vb_x4_0, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget4_f32(vb_x4_0, 3));

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget4_f32(vb_x4_1, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget4_f32(vb_x4_1, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget4_f32(vb_x4_1, 3));

        k_chunks--;
      }
      k %= 4;

      if (k >= 2) {
        const svfloat32x4_t va_x4 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        a0 += 4 * vl;

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 1);
        qw_ptr += 2 * vl;

        const svuint32_t vt01_0 = svbsl_u32(svsli_n_u32(vw01_0, vw01_0, 12), vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(svsli_n_u32(vw01_1, vw01_1, 12), vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        const svfloat32x4_t vb_x4 = svcreate4_f32(
            svcvt_f32_f16_x(pg_32_all, vh01_0),
            svcvt_f32_f16_x(pg_32_all, vh01_1),
            svcvtlt_f32_f16_x(pg_32_all, vh01_0),
            svcvtlt_f32_f16_x(pg_32_all, vh01_1));

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget4_f32(vb_x4, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget4_f32(vb_x4, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget4_f32(vb_x4, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget4_f32(vb_x4, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget4_f32(vb_x4, 2));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget4_f32(vb_x4, 3));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget4_f32(vb_x4, 2));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget4_f32(vb_x4, 3));

        k -= 2;
      }
      if (k != 0) {
        svfloat32x2_t va_x2_0 = svcreate2_f32(svld1_vnum_f32(pg_32_all, a0, 0), svld1_vnum_f32(pg_32_all, a0, 1));
        a0 += 2 * vl;
        svfloat32_t va0 = svget2_f32(va_x2_0, 0);
        svfloat32_t va1 = svget2_f32(va_x2_0, 1);

        const svuint32_t vw01_0 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw01_1 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 1);
        qw_ptr += 2 * vl;

        const svuint32_t vt01_0 = svbsl_u32(vw01_0, vmagic, vmask);
        const svfloat16_t vh01_0 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_0), vmagic_bias);
        const svuint32_t vt01_1 = svbsl_u32(vw01_1, vmagic, vmask);
        const svfloat16_t vh01_1 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01_1), vmagic_bias);
        const svfloat32_t vb0 = svcvt_f32_f16_x(pg_32_all, vh01_0);
        const svfloat32_t vb1 = svcvt_f32_f16_x(pg_32_all, vh01_1);

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, va0, vb0);
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, va0, vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, va1, vb0);
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, va1, vb1);
      }

      w_ptr = qw_ptr;

      svfloat32x2_t vb_scale_x2 = svcreate2_f32(
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 0),
          svld1_vnum_f32(pg_32_all, (const float*) w_ptr, 1));
      w_ptr += 2 * vl * sizeof(float);

#ifdef PREFETCH_W_MATRIX
      {
        size_t prfm_tgt = prfm_tile_base + (size_t) (w_ptr - w_base) + prfm_dist;
        if (prfm_tgt > prfm_end) {
          prfm_tgt = prfm_end;
        }
        while (prfm_pos < prfm_tgt) {
          __builtin_prefetch(w_base + prfm_off, 0, 2);
          if (prfm_off + prfm_line_size < total_w_size) {
            prfm_pos += prfm_line_size;
            prfm_off += prfm_line_size;
          } else {
            prfm_pos += total_w_size - prfm_off;
            prfm_off = 0;
          }
        }
      }
#endif

      svfloat32_t vscale_0 = svget2_f32(vb_scale_x2, 0);
      svfloat32_t vscale_1 = svget2_f32(vb_scale_x2, 1);

      const svbool_t pg_n_0 = svwhilelt_b32_u64(0, nc_rem);
      const svbool_t pg_n_1 = svwhilelt_b32_u64(vl, nc_rem);

      {
        float* c_rows = c0;
        const size_t m_rows = (m_chunk < vl) ? m_chunk : vl;
        svfloat32x4_t z0_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 3));
        svfloat32x4_t z1_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z0_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 7));
          const svfloat32x4_t z1_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 7));
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_a, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z0_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 11));
            z1_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 1, i + 11));
          }
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z1_b, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
        }
      }

      if (m_chunk > vl) {
        float* c_rows = (float*) ((uintptr_t) c0 + vl * cm_stride);
        const size_t m_rows = m_chunk - vl;
        svfloat32x4_t z2_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 3));
        svfloat32x4_t z3_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z2_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 7));
          const svfloat32x4_t z3_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 7));
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_a, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z2_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 11));
            z3_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 3, i + 11));
          }
          {
            svfloat32x4_t scaled0 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 0), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 0), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 1), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 1), vscale_1));
            svfloat32x4_t scaled1 = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 2), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 2), vscale_1),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 3), vscale_0),
                svmul_f32_x(pg_32_all, svget4_f32(z3_b, 3), vscale_1));

            scaled0 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled0, 3), vmin), vmax));
            scaled1 = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(scaled1, 3), vmin), vmax));

            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n_0, c_row, svget4_f32(scaled0, 0));
            svst1_f32(pg_n_1, c_row + vl, svget4_f32(scaled0, 1));
            if (rows_left > 1) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(scaled0, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + cm_stride) + vl, svget4_f32(scaled0, 3));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(scaled1, 0));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 2 * cm_stride) + vl, svget4_f32(scaled1, 1));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n_0, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(scaled1, 2));
              svst1_f32(pg_n_1, (float*) ((uintptr_t) c_row + 3 * cm_stride) + vl, svget4_f32(scaled1, 3));
            }
          }
        }
      }

      if (nc_rem > 2 * vl) {
        c0 += 2 * vl;
        nc_rem -= 2 * vl;
      } else {
        nc_rem = 0;
      }
    }

    if (nc_rem != 0) {
      const svfloat32_t f32_bias = svld1_f32(pg_32_all, (const float*) w_ptr);
      w_ptr += 2 * vl * sizeof(float);
      const svfloat32_t f32_one = svdup_n_f32(1.0f);

      svzero_za();
      svmopa_za32_f32_m(0, pg_32_all, pg_32_all, f32_one, f32_bias);
      svmopa_za32_f32_m(2, pg_32_all, pg_32_all, f32_one, f32_bias);

      const float* a0 = a_base + m_start * kc;
      const uint8_t* qw_ptr = w_ptr;
      size_t k = kc;
      for (; k >= 4; k -= 4) {
        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        const svuint32_t vw23 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 2);
        qw_ptr += 4 * vl;
        const svuint32_t vt01 = svbsl_u32(svsli_n_u32(vw01, vw01, 12), vmagic, vmask);
        const svfloat16_t vh01 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01), vmagic_bias);
        const svuint32_t vt23 = svbsl_u32(svsli_n_u32(vw23, vw23, 12), vmagic, vmask);
        const svfloat16_t vh23 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt23), vmagic_bias);
        const svfloat32_t vb0 = svcvt_f32_f16_x(pg_32_all, vh01);
        const svfloat32_t vb1 = svcvtlt_f32_f16_x(pg_32_all, vh01);
        const svfloat32_t vb2 = svcvt_f32_f16_x(pg_32_all, vh23);
        const svfloat32_t vb3 = svcvtlt_f32_f16_x(pg_32_all, vh23);

#ifdef PREFETCH_W_MATRIX
        {
          size_t prfm_tgt = prfm_tile_base + (size_t) (qw_ptr - w_base) + prfm_dist;
          if (prfm_tgt > prfm_end) {
            prfm_tgt = prfm_end;
          }
          while (prfm_pos < prfm_tgt) {
            __builtin_prefetch(w_base + prfm_off, 0, 2);
            if (prfm_off + prfm_line_size < total_w_size) {
              prfm_pos += prfm_line_size;
              prfm_off += prfm_line_size;
            } else {
              prfm_pos += total_w_size - prfm_off;
              prfm_off = 0;
            }
          }
        }
#endif

        const svfloat32x4_t va_x4_0 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        const svfloat32x4_t va_x4_1 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 0),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 1),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 2),
            svld1_vnum_f32(pg_32_all, a0 + 4 * vl, 3));
        a0 += 8 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), vb0);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), vb1);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), vb2);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), vb2);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), vb3);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), vb3);
      }
      if (k >= 2) {
        const svfloat32x4_t va_x4 = svcreate4_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1),
            svld1_vnum_f32(pg_32_all, a0, 2),
            svld1_vnum_f32(pg_32_all, a0, 3));
        a0 += 4 * vl;
        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        qw_ptr += 2 * vl;
        const svuint32_t vt01 = svbsl_u32(svsli_n_u32(vw01, vw01, 12), vmagic, vmask);
        const svfloat16_t vh01 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01), vmagic_bias);
        const svfloat32_t vb0 = svcvt_f32_f16_x(pg_32_all, vh01);
        const svfloat32_t vb1 = svcvtlt_f32_f16_x(pg_32_all, vh01);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), vb0);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), vb0);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), vb1);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), vb1);
        k -= 2;
      }
      if (k != 0) {
        const svfloat32x2_t va_x2 = svcreate2_f32(
            svld1_vnum_f32(pg_32_all, a0, 0),
            svld1_vnum_f32(pg_32_all, a0, 1));
        a0 += 2 * vl;
        const svuint32_t vw01 = svld1ub_vnum_u32(pg_32_all, qw_ptr, 0);
        qw_ptr += 2 * vl;
        const svuint32_t vt01 = svbsl_u32(vw01, vmagic, vmask);
        const svfloat16_t vh01 = svsub_f16_x(pg_16_all, svreinterpret_f16_u32(vt01), vmagic_bias);
        const svfloat32_t vb = svcvt_f32_f16_x(pg_32_all, vh01);
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget2_f32(va_x2, 0), vb);
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget2_f32(va_x2, 1), vb);
      }

      w_ptr = qw_ptr;
      const svfloat32_t vscale = svld1_f32(pg_32_all, (const float*) w_ptr);
      w_ptr += 2 * vl * sizeof(float);

#ifdef PREFETCH_W_MATRIX
      {
        size_t prfm_tgt = prfm_tile_base + (size_t) (w_ptr - w_base) + prfm_dist;
        if (prfm_tgt > prfm_end) {
          prfm_tgt = prfm_end;
        }
        while (prfm_pos < prfm_tgt) {
          __builtin_prefetch(w_base + prfm_off, 0, 2);
          if (prfm_off + prfm_line_size < total_w_size) {
            prfm_pos += prfm_line_size;
            prfm_off += prfm_line_size;
          } else {
            prfm_pos += total_w_size - prfm_off;
            prfm_off = 0;
          }
        }
      }
#endif

      const svbool_t pg_n = svwhilelt_b32_u64(0, nc_rem);
      {
        float* c_rows = c0;
        const size_t m_rows = (m_chunk < vl) ? m_chunk : vl;
        svfloat32x4_t z0_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z0_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 7));
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z0_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 0, i + 11));
          }
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
        }
      }

      if (m_chunk > vl) {
        float* c_rows = (float*) ((uintptr_t) c0 + vl * cm_stride);
        const size_t m_rows = m_chunk - vl;
        svfloat32x4_t z2_a = svcreate4_f32(
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 0),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 1),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 2),
            svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, 3));
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z2_b = svcreate4_f32(
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 4),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 5),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 6),
              svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 7));
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z2_a = svcreate4_f32(
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 8),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 9),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 10),
                svread_hor_za32_f32_m(svundef_f32(), pg_32_all, 2, i + 11));
          }
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 3), vscale));
            vout = svcreate4_f32(
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 0), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 1), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 2), vmin), vmax),
                svminnm_f32_x(pg_32_all, svmaxnm_f32_x(pg_32_all, svget4_f32(vout, 3), vmin), vmax));
            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32(pg_n, c_row, svget4_f32(vout, 0));
            if (rows_left > 1) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + cm_stride), svget4_f32(vout, 1));
            }
            if (rows_left > 2) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 2 * cm_stride), svget4_f32(vout, 2));
            }
            if (rows_left > 3) {
              svst1_f32(pg_n, (float*) ((uintptr_t) c_row + 3 * cm_stride), svget4_f32(vout, 3));
            }
          }
        }
      }
    }
#ifdef PREFETCH_W_MATRIX
    prfm_tile_base += total_w_size;
#endif
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
    // With SVL >= 512 one Z register holds the whole 16-entry dequantization table (svtbl); smaller SVLs use
    // the fp16 magic-number unpack.
    if (svcntsw() >= 16) {
      pf32_qc4w_gemm_32x32_sme_lut_impl(
        mr, nc, kc / sizeof(float), a, w, c, cm_stride,
        params->scalar.min, params->scalar.max, params->scalar.kernel_zero_point);
    } else {
      pf32_qc4w_gemm_32x32_sme_fmagic_impl(
        mr, nc, kc / sizeof(float), a, w, c, cm_stride,
        params->scalar.min, params->scalar.max, params->scalar.kernel_zero_point);
    }
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME && XNN_ENABLE_ARM_SME_ACLE
