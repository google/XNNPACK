// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.h"

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr() { return 1; }

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot();
}

// Wraps the `kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot(
      m, n, k / sizeof(xnn_float16), lhs_packed, /*unused lhs_stride=*/0,
      rhs_packed, dst, dst_stride_row, /*dst_stride_col=*/sizeof(xnn_float16),
      xnn_float16_to_float(minmax_params->scalar.min),
      xnn_float16_to_float(minmax_params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr() { return 1; }

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf16_gemm_1x16vl_sme2_mla_impl(size_t m, size_t n,
                                                         size_t k_elements,
                                                         const float16_t* lhs,
                                                         const float16_t*
                                                             rhs_packed,
                                                         void* dst,
                                                         size_t dst_stride_row,
                                                         uint16_t min_bits,
                                                         uint16_t max_bits) {
  const size_t vl = svcntw();
  const size_t n_step = 2 * vl;
  const size_t k_rounded = (k_elements + 1) & ~(size_t)1;
  const size_t panel_elems = n_step * (1 + k_rounded);
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  const svbool_t p_row0 = svwhilelt_b16_u64(0, 2);
  const svbool_t p_row1 = sveor_b_z(pg16, svwhilelt_b16_u64(0, 4), p_row0);
  const svbool_t p_row2 =
      sveor_b_z(pg16, svwhilelt_b16_u64(0, 6), svwhilelt_b16_u64(0, 4));
  const svbool_t p_row3 =
      sveor_b_z(pg16, svwhilelt_b16_u64(0, 8), svwhilelt_b16_u64(0, 6));
  const svfloat16_t vzero16 = svdup_n_f16((float16_t)0.0f);
  const svfloat16_t vones16 = svdup_n_f16((float16_t)1.0f);
  const svfloat16_t vmin = svreinterpret_f16_u16(svdup_n_u16(min_bits));
  const svfloat16_t vmax = svreinterpret_f16_u16(svdup_n_u16(max_bits));

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float16_t* lhs_row = lhs + m_idx * k_elements;
    float16_t* dst_row = (float16_t*)((uintptr_t)dst + m_idx * dst_stride_row);

    for (size_t n_idx = 0; n_idx < n; n_idx += 2 * n_step) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      const float16_t* p1 = (n_idx + n_step < n) ? p0 + panel_elems : p0;
      const svfloat16_t vbias_a = svld1_f16(pg16, p0);
      const svfloat16_t vbias_b = svld1_f16(pg16, p1);
      const svfloat16_t vbias0 = svzip1_f16(vbias_a, vzero16);
      const svfloat16_t vbias1 = svzip2_f16(vbias_a, vzero16);
      const svfloat16_t vbias2 = svzip1_f16(vbias_b, vzero16);
      const svfloat16_t vbias3 = svzip2_f16(vbias_b, vzero16);

      svzero_za();
      svmopa_za32_f16_m(0, p_row0, pg16, vones16, vbias0);
      svmopa_za32_f16_m(1, p_row0, pg16, vones16, vbias1);
      svmopa_za32_f16_m(2, p_row0, pg16, vones16, vbias2);
      svmopa_za32_f16_m(3, p_row0, pg16, vones16, vbias3);

      const float16_t* w0 = p0 + n_step;
      const float16_t* w1 = p1 + n_step;
      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);
#pragma clang loop unroll(full)
        for (int u = 0; u < 4; ++u) {
          const svbool_t p_row =
              u == 0 ? p_row0 : (u == 1 ? p_row1 : (u == 2 ? p_row2 : p_row3));
          const svfloat16_t wb0 = svld1_vnum_f16(pg16, w0, 2 * u + 0);
          const svfloat16_t wb1 = svld1_vnum_f16(pg16, w0, 2 * u + 1);
          const svfloat16_t wb2 = svld1_vnum_f16(pg16, w1, 2 * u + 0);
          const svfloat16_t wb3 = svld1_vnum_f16(pg16, w1, 2 * u + 1);
          svmopa_za32_f16_m(0, p_row, pg16, va, wb0);
          svmopa_za32_f16_m(1, p_row, pg16, va, wb1);
          svmopa_za32_f16_m(2, p_row, pg16, va, wb2);
          svmopa_za32_f16_m(3, p_row, pg16, va, wb3);
        }
        w0 += 8 * n_step;
        w1 += 8 * n_step;
      }

      for (; k_idx + 2 <= k_elements; k_idx += 2) {
        uint32_t a_pair;
        __builtin_memcpy(&a_pair, lhs_row + k_idx, sizeof(uint32_t));
        const svfloat16_t va = svreinterpret_f16_u32(svdup_n_u32(a_pair));
        svmopa_za32_f16_m(0, p_row0, pg16, va, svld1_vnum_f16(pg16, w0, 0));
        svmopa_za32_f16_m(1, p_row0, pg16, va, svld1_vnum_f16(pg16, w0, 1));
        svmopa_za32_f16_m(2, p_row0, pg16, va, svld1_vnum_f16(pg16, w1, 0));
        svmopa_za32_f16_m(3, p_row0, pg16, va, svld1_vnum_f16(pg16, w1, 1));
        w0 += 2 * n_step;
        w1 += 2 * n_step;
      }

      if (k_idx < k_elements) {
        uint16_t a_last;
        __builtin_memcpy(&a_last, lhs_row + k_idx, sizeof(uint16_t));
        const svfloat16_t va =
            svreinterpret_f16_u32(svdup_n_u32((uint32_t)a_last));
        svmopa_za32_f16_m(0, p_row0, pg16, va, svld1_vnum_f16(pg16, w0, 0));
        svmopa_za32_f16_m(1, p_row0, pg16, va, svld1_vnum_f16(pg16, w0, 1));
        svmopa_za32_f16_m(2, p_row0, pg16, va, svld1_vnum_f16(pg16, w1, 0));
        svmopa_za32_f16_m(3, p_row0, pg16, va, svld1_vnum_f16(pg16, w1, 1));
      }

      const svfloat32_t vf0 = svadd_f32_x(
          pg32,
          svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 0),
                      svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 1)),
          svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 2),
                      svread_hor_za32_f32_m(svundef_f32(), pg32, 0, 3)));
      const svfloat32_t vf1 = svadd_f32_x(
          pg32,
          svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 0),
                      svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 1)),
          svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 2),
                      svread_hor_za32_f32_m(svundef_f32(), pg32, 1, 3)));
      svfloat16_t vout0 =
          svuzp1_f16(svcvt_f16_f32_x(pg32, vf0), svcvt_f16_f32_x(pg32, vf1));
      vout0 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout0, vmin), vmax);
      const svbool_t p_out0 = svwhilelt_b16_u64(0, n - n_idx);
      svst1_f16(p_out0, dst_row + n_idx, vout0);

      if (n_idx + n_step < n) {
        const svfloat32_t vf2 = svadd_f32_x(
            pg32,
            svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 0),
                        svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 1)),
            svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 2),
                        svread_hor_za32_f32_m(svundef_f32(), pg32, 2, 3)));
        const svfloat32_t vf3 = svadd_f32_x(
            pg32,
            svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 0),
                        svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 1)),
            svadd_f32_x(pg32, svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 2),
                        svread_hor_za32_f32_m(svundef_f32(), pg32, 3, 3)));
        svfloat16_t vout1 =
            svuzp1_f16(svcvt_f16_f32_x(pg32, vf2), svcvt_f16_f32_x(pg32, vf3));
        vout1 = svmin_f16_m(pg16, svmax_f16_m(pg16, vout1, vmin), vmax);
        const svbool_t p_out1 = svwhilelt_b16_u64(n_step, n - n_idx);
        svst1_f16(p_out1, dst_row + n_idx + n_step, vout1);
      }
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  (void)dst_stride_col;
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &minmax_params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &minmax_params->scalar.max, sizeof(uint16_t));
  xnn_pf16_gemm_1x16vl_sme2_mla_impl(
      m, n, k / sizeof(xnn_float16), (const float16_t*)lhs_packed,
      (const float16_t*)rhs_packed, dst, dst_stride_row, min_bits, max_bits);
}
#else
#include <assert.h>

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr() { return 0; }
size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr() { return 0; }
void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
