// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/common.h"

#if XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
#include <arm_sme.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_32x32__sme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf32_qc4w_gemm_minmax_ukernel_32x32__sme2_get_nr() {
  return 2 * svcntsw();
}

#define PREFETCH_W_MATRIX

__arm_locally_streaming __arm_new("za", "zt0")
static void pf32_qc4w_gemm_32x32_sme2_impl(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* a,
    const void* w,
    float* c,
    size_t cm_stride,
    float min,
    float max,
    const float* lut)
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

  svbool_t pg_8_all = svptrue_b8();
  svbool_t pg_32_all = svptrue_b32();
  svcount_t pcnt_32_all = svptrue_c32();

  svfloat32_t vmin = svdup_n_f32(min);
  svfloat32_t vmax = svdup_n_f32(max);

  const float* a_base = (const float*) a;
  const uint8_t* w_base = (const uint8_t*) w;

  svldr_zt(0, lut);

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
      svfloat32x2_t vb_bias_x2 = svld1_f32_x2(pcnt_32_all, (const float*) w_ptr);
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
        const svuint8_t vw = svld1_u8(pg_8_all, qw_ptr);
        qw_ptr += 4 * vl;
        const svfloat32x4_t vt_0 = svluti4_lane_zt_f32_x4(0, vw, 0);
        const svfloat32x4_t vt_1 = svluti4_lane_zt_f32_x4(0, vw, 1);
        const svfloat32x2_t vb_lo_0 = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt_0, 0), svget4_f32(vt_0, 1)));
        const svfloat32x2_t vb_hi_0 = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt_0, 2), svget4_f32(vt_0, 3)));
        const svfloat32x2_t vb_lo_1 = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt_1, 0), svget4_f32(vt_1, 1)));
        const svfloat32x2_t vb_hi_1 = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt_1, 2), svget4_f32(vt_1, 3)));

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

        const svfloat32x4_t va_x4_0 = svld1_f32_x4(pcnt_32_all, a0);
        const svfloat32x4_t va_x4_1 = svld1_f32_x4(pcnt_32_all, a0 + 4 * vl);
        a0 += 8 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget2_f32(vb_lo_0, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget2_f32(vb_hi_0, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget2_f32(vb_lo_0, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget2_f32(vb_hi_0, 0));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget2_f32(vb_lo_0, 1));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget2_f32(vb_hi_0, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget2_f32(vb_lo_0, 1));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget2_f32(vb_hi_0, 1));

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget2_f32(vb_lo_1, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget2_f32(vb_hi_1, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget2_f32(vb_lo_1, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget2_f32(vb_hi_1, 0));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget2_f32(vb_lo_1, 1));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget2_f32(vb_hi_1, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget2_f32(vb_lo_1, 1));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget2_f32(vb_hi_1, 1));

        k_chunks--;
      }
      k %= 4;

      if (k >= 2) {
        const svfloat32x4_t va_x4 = svld1_f32_x4(pcnt_32_all, a0);
        a0 += 4 * vl;
        const svuint8_t vw = svld1_u8(svwhilelt_b8_u64(0, 2 * vl), qw_ptr);
        qw_ptr += 2 * vl;
        const svfloat32x4_t vt = svluti4_lane_zt_f32_x4(0, vw, 0);
        const svfloat32x2_t vb_lo = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt, 0), svget4_f32(vt, 1)));
        const svfloat32x2_t vb_hi = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt, 2), svget4_f32(vt, 3)));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget2_f32(vb_lo, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget2_f32(vb_hi, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget2_f32(vb_lo, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget2_f32(vb_hi, 0));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget2_f32(vb_lo, 1));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget2_f32(vb_hi, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget2_f32(vb_lo, 1));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget2_f32(vb_hi, 1));
        k -= 2;
      }
      if (k != 0) {
        const svfloat32x2_t va_x2 = svld1_f32_x2(pcnt_32_all, a0);
        a0 += 2 * vl;
        const svuint8_t vw = svld1_u8(svwhilelt_b8_u64(0, 2 * vl), qw_ptr);
        qw_ptr += 2 * vl;
        const svfloat32x4_t vt = svluti4_lane_zt_f32_x4(0, vw, 0);
        const svfloat32x2_t vb_lo = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt, 0), svget4_f32(vt, 1)));
        const svfloat32x2_t vb_hi = svuzp_f32_x2(svcreate2_f32(svget4_f32(vt, 2), svget4_f32(vt, 3)));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget2_f32(va_x2, 0), svget2_f32(vb_lo, 0));
        svmopa_za32_f32_m(1, pg_32_all, pg_32_all, svget2_f32(va_x2, 0), svget2_f32(vb_hi, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget2_f32(va_x2, 1), svget2_f32(vb_lo, 0));
        svmopa_za32_f32_m(3, pg_32_all, pg_32_all, svget2_f32(va_x2, 1), svget2_f32(vb_hi, 0));
      }

      w_ptr = qw_ptr;

      svfloat32x2_t vb_scale_x2 = svld1_f32_x2(pcnt_32_all, (const float*) w_ptr);
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

      svcount_t pn = svwhilelt_c32_u64(0, nc_rem, 2);

      {
        float* c_rows = c0;
        const size_t m_rows = (m_chunk < vl) ? m_chunk : vl;
        svfloat32x4_t z0_a = svread_hor_za32_f32_vg4(0, 0);
        svfloat32x4_t z1_a = svread_hor_za32_f32_vg4(1, 0);
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z0_b = svread_hor_za32_f32_vg4(0, i + 4);
          const svfloat32x4_t z1_b = svread_hor_za32_f32_vg4(1, i + 4);
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

            scaled0 = svclamp_single_f32_x4(scaled0, vmin, vmax);
            scaled1 = svclamp_single_f32_x4(scaled1, vmin, vmax);

            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32_x2(pn, c_row, svcreate2_f32(svget4_f32(scaled0, 0), svget4_f32(scaled0, 1)));
            if (rows_left > 1) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + cm_stride),
                           svcreate2_f32(svget4_f32(scaled0, 2), svget4_f32(scaled0, 3)));
            }
            if (rows_left > 2) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 2 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 0), svget4_f32(scaled1, 1)));
            }
            if (rows_left > 3) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 3 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 2), svget4_f32(scaled1, 3)));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z0_a = svread_hor_za32_f32_vg4(0, i + 8);
            z1_a = svread_hor_za32_f32_vg4(1, i + 8);
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

            scaled0 = svclamp_single_f32_x4(scaled0, vmin, vmax);
            scaled1 = svclamp_single_f32_x4(scaled1, vmin, vmax);

            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32_x2(pn, c_row, svcreate2_f32(svget4_f32(scaled0, 0), svget4_f32(scaled0, 1)));
            if (rows_left > 1) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + cm_stride),
                           svcreate2_f32(svget4_f32(scaled0, 2), svget4_f32(scaled0, 3)));
            }
            if (rows_left > 2) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 2 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 0), svget4_f32(scaled1, 1)));
            }
            if (rows_left > 3) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 3 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 2), svget4_f32(scaled1, 3)));
            }
          }
        }
      }

      if (m_chunk > vl) {
        float* c_rows = (float*) ((uintptr_t) c0 + vl * cm_stride);
        const size_t m_rows = m_chunk - vl;
        svfloat32x4_t z2_a = svread_hor_za32_f32_vg4(2, 0);
        svfloat32x4_t z3_a = svread_hor_za32_f32_vg4(3, 0);
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z2_b = svread_hor_za32_f32_vg4(2, i + 4);
          const svfloat32x4_t z3_b = svread_hor_za32_f32_vg4(3, i + 4);
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

            scaled0 = svclamp_single_f32_x4(scaled0, vmin, vmax);
            scaled1 = svclamp_single_f32_x4(scaled1, vmin, vmax);

            float* c_row = (float*) ((uintptr_t) c_rows + i * cm_stride);
            const size_t rows_left = m_rows - i;
            svst1_f32_x2(pn, c_row, svcreate2_f32(svget4_f32(scaled0, 0), svget4_f32(scaled0, 1)));
            if (rows_left > 1) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + cm_stride),
                           svcreate2_f32(svget4_f32(scaled0, 2), svget4_f32(scaled0, 3)));
            }
            if (rows_left > 2) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 2 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 0), svget4_f32(scaled1, 1)));
            }
            if (rows_left > 3) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 3 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 2), svget4_f32(scaled1, 3)));
            }
          }
          if (i + 4 >= m_rows) {
            break;
          }
          if (i + 8 < m_rows) {
            z2_a = svread_hor_za32_f32_vg4(2, i + 8);
            z3_a = svread_hor_za32_f32_vg4(3, i + 8);
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

            scaled0 = svclamp_single_f32_x4(scaled0, vmin, vmax);
            scaled1 = svclamp_single_f32_x4(scaled1, vmin, vmax);

            float* c_row = (float*) ((uintptr_t) c_rows + (i + 4) * cm_stride);
            const size_t rows_left = m_rows - (i + 4);
            svst1_f32_x2(pn, c_row, svcreate2_f32(svget4_f32(scaled0, 0), svget4_f32(scaled0, 1)));
            if (rows_left > 1) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + cm_stride),
                           svcreate2_f32(svget4_f32(scaled0, 2), svget4_f32(scaled0, 3)));
            }
            if (rows_left > 2) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 2 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 0), svget4_f32(scaled1, 1)));
            }
            if (rows_left > 3) {
              svst1_f32_x2(pn, (float*) ((uintptr_t) c_row + 3 * cm_stride),
                           svcreate2_f32(svget4_f32(scaled1, 2), svget4_f32(scaled1, 3)));
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
        const svuint8_t vw = svld1_u8(pg_8_all, qw_ptr);
        qw_ptr += 4 * vl;
        const svfloat32x2_t vb01 = svuzp_f32_x2(svluti4_lane_zt_f32_x2(0, vw, 0));
        const svfloat32x2_t vb23 = svuzp_f32_x2(svluti4_lane_zt_f32_x2(0, vw, 2));

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

        const svfloat32x4_t va_x4_0 = svld1_f32_x4(pcnt_32_all, a0);
        const svfloat32x4_t va_x4_1 = svld1_f32_x4(pcnt_32_all, a0 + 4 * vl);
        a0 += 8 * vl;

        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 0), svget2_f32(vb01, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 1), svget2_f32(vb01, 0));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 2), svget2_f32(vb01, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_0, 3), svget2_f32(vb01, 1));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 0), svget2_f32(vb23, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 1), svget2_f32(vb23, 0));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 2), svget2_f32(vb23, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4_1, 3), svget2_f32(vb23, 1));
      }
      k %= 4;

      if (k >= 2) {
        const svfloat32x4_t va_x4 = svld1_f32_x4(pcnt_32_all, a0);
        a0 += 4 * vl;
        const svuint8_t vw = svld1_u8(svwhilelt_b8_u64(0, vl), qw_ptr);
        const svfloat32x2_t vb01 = svuzp_f32_x2(svluti4_lane_zt_f32_x2(0, vw, 0));
        qw_ptr += 2 * vl;
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 0), svget2_f32(vb01, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 1), svget2_f32(vb01, 0));
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget4_f32(va_x4, 2), svget2_f32(vb01, 1));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget4_f32(va_x4, 3), svget2_f32(vb01, 1));
        k -= 2;
      }
      if (k != 0) {
        const svfloat32x2_t va_x2 = svld1_f32_x2(pcnt_32_all, a0);
        a0 += 2 * vl;
        const svuint8_t vw = svld1_u8(svwhilelt_b8_u64(0, vl), qw_ptr);
        const svfloat32x2_t vb01 = svuzp_f32_x2(svluti4_lane_zt_f32_x2(0, vw, 0));
        qw_ptr += 2 * vl;
        svmopa_za32_f32_m(0, pg_32_all, pg_32_all, svget2_f32(va_x2, 0), svget2_f32(vb01, 0));
        svmopa_za32_f32_m(2, pg_32_all, pg_32_all, svget2_f32(va_x2, 1), svget2_f32(vb01, 0));
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
        svfloat32x4_t z0_a = svread_hor_za32_f32_vg4(0, 0);
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z0_b = svread_hor_za32_f32_vg4(0, i + 4);
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_a, 3), vscale));
            vout = svclamp_single_f32_x4(vout, vmin, vmax);

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
            z0_a = svread_hor_za32_f32_vg4(0, i + 8);
          }
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z0_b, 3), vscale));
            vout = svclamp_single_f32_x4(vout, vmin, vmax);

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
        svfloat32x4_t z2_a = svread_hor_za32_f32_vg4(2, 0);
        XNN_PRAGMA_CLANG("clang loop unroll(disable)")
        for (size_t i = 0; i < m_rows; i += 8) {
          const svfloat32x4_t z2_b = svread_hor_za32_f32_vg4(2, i + 4);
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_a, 3), vscale));
            vout = svclamp_single_f32_x4(vout, vmin, vmax);

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
            z2_a = svread_hor_za32_f32_vg4(2, i + 8);
          }
          {
            svfloat32x4_t vout = svcreate4_f32(
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 0), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 1), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 2), vscale),
                svmul_f32_x(pg_32_all, svget4_f32(z2_b, 3), vscale));
            vout = svclamp_single_f32_x4(vout, vmin, vmax);

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

void xnn_pf32_qc4w_gemm_minmax_ukernel_32x32__sme2(
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
    XNN_ALIGN(64) float lut[16];
    for (int32_t i = 0; i < 16; i++) {
      lut[i] = (float) (i - params->scalar.kernel_zero_point);
    }
    pf32_qc4w_gemm_32x32_sme2_impl(
      mr, nc, kc / sizeof(float), a, w, c, cm_stride,
      params->scalar.min, params->scalar.max, lut);
}

#undef PREFETCH_W_MATRIX

#endif  // XNN_ARCH_ARM64 && XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME_ACLE
