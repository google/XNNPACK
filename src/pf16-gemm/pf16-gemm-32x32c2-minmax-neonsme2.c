// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16p_f16p/kai_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa.h"

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa();
}

// Wraps the `kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f16_f16p2vlx2_f16p2vlx2_2vlx2vl_sme2_mopa(
      m, n, k / sizeof(xnn_float16), lhs_packed, rhs_packed, dst,
      dst_stride_row, /*dst_stride_col=*/sizeof(xnn_float16),
      xnn_float16_to_float(minmax_params->scalar.min),
      xnn_float16_to_float(minmax_params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pf16_gemm_2vlx2vl_sme2_mopa_impl(size_t m, size_t n, size_t k_elements,
                                     const float16_t* lhs_packed,
                                     const float16_t* rhs_packed, void* dst,
                                     size_t dst_stride_row, uint16_t min_bits,
                                     uint16_t max_bits) {
  const size_t vl = svcntw();
  const size_t m_step = 2 * vl;
  const size_t n_step = 2 * vl;
  const size_t k_rounded = (k_elements + 1) & ~(size_t)1;
  const size_t k_pairs = k_rounded / 2;
  const size_t lhs_panel_elems = m_step * k_rounded;
  const size_t rhs_panel_elems = n_step * (1 + k_rounded);
  const size_t total_w_bytes =
      ((n + n_step - 1) / n_step) * rhs_panel_elems * sizeof(float16_t);
#if defined(__APPLE__)
  const size_t prfm_min_w_size = 16 * 1024 * 1024;
#else
  const size_t prfm_min_w_size = 1024 * 1024;
#endif
  const int need_prfm = total_w_bytes > prfm_min_w_size;
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc16 = svptrue_c16();
  const svfloat16_t vones16 = svdup_n_f16((float16_t)1.0f);
  const svfloat16_t vmin = svreinterpret_f16_u16(svdup_n_u16(min_bits));
  const svfloat16_t vmax = svreinterpret_f16_u16(svdup_n_u16(max_bits));

  for (size_t m_idx = 0; m_idx < m; m_idx += m_step) {
    const size_t m_rem = m - m_idx < m_step ? m - m_idx : m_step;
    const float16_t* lhs_panel =
        lhs_packed + (m_idx / m_step) * lhs_panel_elems;

    for (size_t n_idx = 0; n_idx < n; n_idx += n_step) {
      if (m_rem <= vl && n_idx + n_step < n) {
        const float16_t* rhs_panel0 =
            rhs_packed + (n_idx / n_step) * rhs_panel_elems;
        const float16_t* rhs_panel1 = rhs_panel0 + rhs_panel_elems;
        const svfloat16_t vbias0 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)rhs_panel0));
        const svfloat16_t vbias1 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)(rhs_panel0 + vl)));
        const svfloat16_t vbias2 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)rhs_panel1));
        const svfloat16_t vbias3 = svreinterpret_f16_u32(
            svld1uh_u32(pg32, (const uint16_t*)(rhs_panel1 + vl)));

        svzero_za();
        svmopa_za32_f16_m(0, pg16, pg16, vones16, vbias0);
        svmopa_za32_f16_m(1, pg16, pg16, vones16, vbias1);
        svmopa_za32_f16_m(2, pg16, pg16, vones16, vbias2);
        svmopa_za32_f16_m(3, pg16, pg16, vones16, vbias3);

        const float16_t* a_ptr = lhs_panel;
        const float16_t* b0_ptr = rhs_panel0 + n_step;
        const float16_t* b1_ptr = rhs_panel1 + n_step;
        size_t kp = 0;
        for (; kp + 2 <= k_pairs; kp += 2) {
          const svfloat16x4_t va = svld1_f16_x4(pc16, a_ptr);
          const svfloat16x4_t vb0 = svld1_f16_x4(pc16, b0_ptr);
          const svfloat16x4_t vb1 = svld1_f16_x4(pc16, b1_ptr);

          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va, 0),
                            svget4_f16(vb0, 0));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va, 0),
                            svget4_f16(vb0, 1));
          svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va, 0),
                            svget4_f16(vb1, 0));
          svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va, 0),
                            svget4_f16(vb1, 1));

          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va, 2),
                            svget4_f16(vb0, 2));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va, 2),
                            svget4_f16(vb0, 3));
          svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va, 2),
                            svget4_f16(vb1, 2));
          svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va, 2),
                            svget4_f16(vb1, 3));

          a_ptr += 4 * m_step;
          b0_ptr += 4 * n_step;
          b1_ptr += 4 * n_step;
        }
        for (; kp < k_pairs; ++kp) {
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          const svfloat16x2_t vb0 = svld1_f16_x2(pc16, b0_ptr);
          const svfloat16x2_t vb1 = svld1_f16_x2(pc16, b1_ptr);
          a_ptr += 2 * m_step;
          b0_ptr += 2 * n_step;
          b1_ptr += 2 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, va0, svget2_f16(vb0, 0));
          svmopa_za32_f16_m(1, pg16, pg16, va0, svget2_f16(vb0, 1));
          svmopa_za32_f16_m(2, pg16, pg16, va0, svget2_f16(vb1, 0));
          svmopa_za32_f16_m(3, pg16, pg16, va0, svget2_f16(vb1, 1));
        }

        const svcount_t p_out2 = svwhilelt_c16_u64(n_idx, n, 2);
        uint8_t* dst_row =
            (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float16_t);
        for (size_t row = 0; row < m_rem; ++row) {
          const svfloat32x4_t vza = svreinterpret_f32_u8_x4(
              svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));
          const svfloat16_t vh01 = svcvt_f16_f32_x2(
              svcreate2_f32(svget4_f32(vza, 0), svget4_f32(vza, 1)));
          const svfloat16_t vh23 = svcvt_f16_f32_x2(
              svcreate2_f32(svget4_f32(vza, 2), svget4_f32(vza, 3)));
          const svfloat16x2_t vout =
              svclamp_single_f16_x2(svcreate2_f16(vh01, vh23), vmin, vmax);
          svst1_f16_x2(p_out2, (float16_t*)dst_row, vout);
          dst_row += dst_stride_row;
        }
        n_idx += n_step;
        continue;
      }

      const float16_t* rhs_panel =
          rhs_packed + (n_idx / n_step) * rhs_panel_elems;
      const float16_t* pf_ptr = NULL;
      if (need_prfm) {
        if (n_idx + n_step < n) {
          pf_ptr = rhs_panel + rhs_panel_elems;
        } else if (m_idx + m_step < m) {
          pf_ptr = rhs_packed;
        }
        if (pf_ptr != NULL) {
          __builtin_prefetch((const uint8_t*)pf_ptr + 0, 0, 2);
          pf_ptr += n_step;
        }
      }

      const svfloat16_t vbias0 =
          svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)rhs_panel));
      const svfloat16_t vbias1 = svreinterpret_f16_u32(
          svld1uh_u32(pg32, (const uint16_t*)(rhs_panel + vl)));

      svzero_za();
      svmopa_za32_f16_m(0, pg16, pg16, vones16, vbias0);
      svmopa_za32_f16_m(1, pg16, pg16, vones16, vbias1);

      const float16_t* a_ptr = lhs_panel;
      const float16_t* b_ptr = rhs_panel + n_step;
      size_t kp = 0;

      if (m_rem <= vl) {
        for (; kp + 2 <= k_pairs; kp += 2) {
          const svfloat16x4_t va = svld1_f16_x4(pc16, a_ptr);
          const svfloat16x4_t vb = svld1_f16_x4(pc16, b_ptr);
          a_ptr += 4 * m_step;
          b_ptr += 4 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va, 0),
                            svget4_f16(vb, 0));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va, 0),
                            svget4_f16(vb, 1));
          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va, 2),
                            svget4_f16(vb, 2));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va, 2),
                            svget4_f16(vb, 3));
        }
        for (; kp < k_pairs; ++kp) {
          const svfloat16_t va0 = svld1_vnum_f16(pg16, a_ptr, 0);
          const svfloat16x2_t vb = svld1_f16_x2(pc16, b_ptr);
          a_ptr += 2 * m_step;
          b_ptr += 2 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, va0, svget2_f16(vb, 0));
          svmopa_za32_f16_m(1, pg16, pg16, va0, svget2_f16(vb, 1));
        }
      } else {
        svmopa_za32_f16_m(2, pg16, pg16, vones16, vbias0);
        svmopa_za32_f16_m(3, pg16, pg16, vones16, vbias1);
        if (k_pairs >= 4) {
          svfloat16x4_t va_x4_0 = svld1_f16_x4(pc16, a_ptr);
          svfloat16x4_t vb_x4_0 = svld1_f16_x4(pc16, b_ptr);
          svfloat16x4_t va_x4_1 = svld1_f16_x4(pc16, a_ptr + 4 * m_step);
          svfloat16x4_t vb_x4_1 = svld1_f16_x4(pc16, b_ptr + 4 * n_step);
          a_ptr += 8 * m_step;
          b_ptr += 8 * n_step;

          for (; kp + 8 <= k_pairs; kp += 4) {
            svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_0, 0),
                              svget4_f16(vb_x4_0, 0));
            svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_0, 0),
                              svget4_f16(vb_x4_0, 1));
            svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_0, 1),
                              svget4_f16(vb_x4_0, 0));
            svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_0, 1),
                              svget4_f16(vb_x4_0, 1));
            svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_0, 2),
                              svget4_f16(vb_x4_0, 2));
            svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_0, 2),
                              svget4_f16(vb_x4_0, 3));
            svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_0, 3),
                              svget4_f16(vb_x4_0, 2));
            svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_0, 3),
                              svget4_f16(vb_x4_0, 3));

            va_x4_0 = svld1_f16_x4(pc16, a_ptr);
            vb_x4_0 = svld1_f16_x4(pc16, b_ptr);

            svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_1, 0),
                              svget4_f16(vb_x4_1, 0));
            svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_1, 0),
                              svget4_f16(vb_x4_1, 1));
            svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_1, 1),
                              svget4_f16(vb_x4_1, 0));
            svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_1, 1),
                              svget4_f16(vb_x4_1, 1));
            svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_1, 2),
                              svget4_f16(vb_x4_1, 2));
            svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_1, 2),
                              svget4_f16(vb_x4_1, 3));
            svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_1, 3),
                              svget4_f16(vb_x4_1, 2));
            svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_1, 3),
                              svget4_f16(vb_x4_1, 3));

            va_x4_1 = svld1_f16_x4(pc16, a_ptr + 4 * m_step);
            vb_x4_1 = svld1_f16_x4(pc16, b_ptr + 4 * n_step);
            a_ptr += 8 * m_step;
            b_ptr += 8 * n_step;

            if (pf_ptr != NULL) {
              __builtin_prefetch((const uint8_t*)pf_ptr + 0, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 64, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 128, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 192, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 256, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 320, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 384, 0, 2);
              __builtin_prefetch((const uint8_t*)pf_ptr + 448, 0, 2);
              pf_ptr += 8 * n_step;
            }
          }

          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_0, 0),
                            svget4_f16(vb_x4_0, 0));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_0, 0),
                            svget4_f16(vb_x4_0, 1));
          svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_0, 1),
                            svget4_f16(vb_x4_0, 0));
          svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_0, 1),
                            svget4_f16(vb_x4_0, 1));
          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_0, 2),
                            svget4_f16(vb_x4_0, 2));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_0, 2),
                            svget4_f16(vb_x4_0, 3));
          svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_0, 3),
                            svget4_f16(vb_x4_0, 2));
          svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_0, 3),
                            svget4_f16(vb_x4_0, 3));

          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_1, 0),
                            svget4_f16(vb_x4_1, 0));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_1, 0),
                            svget4_f16(vb_x4_1, 1));
          svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_1, 1),
                            svget4_f16(vb_x4_1, 0));
          svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_1, 1),
                            svget4_f16(vb_x4_1, 1));
          svmopa_za32_f16_m(0, pg16, pg16, svget4_f16(va_x4_1, 2),
                            svget4_f16(vb_x4_1, 2));
          svmopa_za32_f16_m(1, pg16, pg16, svget4_f16(va_x4_1, 2),
                            svget4_f16(vb_x4_1, 3));
          svmopa_za32_f16_m(2, pg16, pg16, svget4_f16(va_x4_1, 3),
                            svget4_f16(vb_x4_1, 2));
          svmopa_za32_f16_m(3, pg16, pg16, svget4_f16(va_x4_1, 3),
                            svget4_f16(vb_x4_1, 3));
          kp += 4;
        }
        for (; kp < k_pairs; ++kp) {
          const svfloat16x2_t va = svld1_f16_x2(pc16, a_ptr);
          const svfloat16x2_t vb = svld1_f16_x2(pc16, b_ptr);
          a_ptr += 2 * m_step;
          b_ptr += 2 * n_step;

          svmopa_za32_f16_m(0, pg16, pg16, svget2_f16(va, 0),
                            svget2_f16(vb, 0));
          svmopa_za32_f16_m(1, pg16, pg16, svget2_f16(va, 0),
                            svget2_f16(vb, 1));
          svmopa_za32_f16_m(2, pg16, pg16, svget2_f16(va, 1),
                            svget2_f16(vb, 0));
          svmopa_za32_f16_m(3, pg16, pg16, svget2_f16(va, 1),
                            svget2_f16(vb, 1));
        }
      }

      const svbool_t p_out = svwhilelt_b16_u64(0, n - n_idx);
      const size_t rows0 = m_rem < vl ? m_rem : vl;
      const size_t paired_rows = m_rem > vl ? (m_rem - vl) : 0;
      uint8_t* dst_top =
          (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float16_t);
      uint8_t* dst_bot = dst_top + vl * dst_stride_row;

      size_t row = 0;
      for (; row < paired_rows; ++row) {
        const svfloat32x4_t vza = svreinterpret_f32_u8_x4(
            svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));
        const svfloat16_t vh_top = svcvt_f16_f32_x2(
            svcreate2_f32(svget4_f32(vza, 0), svget4_f32(vza, 1)));
        const svfloat16_t vh_bot = svcvt_f16_f32_x2(
            svcreate2_f32(svget4_f32(vza, 2), svget4_f32(vza, 3)));
        const svfloat16x2_t vout =
            svclamp_single_f16_x2(svcreate2_f16(vh_top, vh_bot), vmin, vmax);
        svst1_f16(p_out, (float16_t*)dst_top, svget2_f16(vout, 0));
        svst1_f16(p_out, (float16_t*)dst_bot, svget2_f16(vout, 1));
        dst_top += dst_stride_row;
        dst_bot += dst_stride_row;
      }
      for (; row < rows0; ++row) {
        const svfloat32x2_t vza = svreinterpret_f32_u8_x2(
            svread_hor_za8_u8_vg2(0, (uint32_t)(row * 4)));
        const svfloat16_t vh = svcvt_f16_f32_x2(vza);
        const svfloat16_t vout =
            svmin_f16_x(pg16, svmax_f16_x(pg16, vh, vmin), vmax);
        svst1_f16(p_out, (float16_t*)dst_top, vout);
        dst_top += dst_stride_row;
      }
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  (void)dst_stride_col;
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &minmax_params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &minmax_params->scalar.max, sizeof(uint16_t));
  xnn_pf16_gemm_2vlx2vl_sme2_mopa_impl(
      m, n, k / sizeof(xnn_float16), (const float16_t*)lhs_packed,
      (const float16_t*)rhs_packed, dst, dst_stride_row, min_bits, max_bits);
}
#else
#include <assert.h>

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_mr() {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2_get_nr() {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pf16_gemm_minmax_ukernel_32x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
