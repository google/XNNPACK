// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla.h"

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme_get_mr() { return 1; }

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla();
}

// Wraps the `kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x8vl_sme_mla(
      m, n, k / sizeof(xnn_float16), lhs_packed, /*unused lhs_stride=*/0,
      rhs_packed, dst, dst_stride_row, /*dst_stride_col=*/sizeof(xnn_float16),
      xnn_float16_to_float(minmax_params->scalar.min),
      xnn_float16_to_float(minmax_params->scalar.max));
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>
#include <string.h>

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme_get_mr() { return 1; }

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme_get_nr() {
  return 2 * svcntsw();
}

static XNN_INLINE svfloat32_t xnn_pf16_sme_load_bias_half(
    const float16_t* p) __arm_streaming {
  const svbool_t pg32 = svptrue_b32();
  return svcvt_f32_f16_x(
      pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)p)));
}

static XNN_INLINE void xnn_pf16_sme_store_panel(
    svfloat32_t vf0, svfloat32_t vf1, svfloat16_t vmin, svfloat16_t vmax,
    float16_t* dst_row, size_t col_idx, size_t n) __arm_streaming {
  const svbool_t pg16 = svptrue_b16();
  const svbool_t pg32 = svptrue_b32();
  svfloat16_t vout =
      svuzp1_f16(svcvt_f16_f32_x(pg32, vf0), svcvt_f16_f32_x(pg32, vf1));
  vout = svmin_f16_m(pg16, svmax_f16_m(pg16, vout, vmin), vmax);
  svst1_f16(svwhilelt_b16_u64(col_idx, n), dst_row + col_idx, vout);
}

__arm_locally_streaming static XNN_NOINLINE void
xnn_pf16_gemm_1x8vl_sme_mla_impl(size_t m, size_t n, size_t k_elements,
                                 const float16_t* lhs,
                                 const float16_t* rhs_packed, void* dst,
                                 size_t dst_stride_row, uint16_t min_bits,
                                 uint16_t max_bits) {
  const size_t vl = svcntw();
  const size_t n_step = 2 * vl;
  const size_t k_rounded = (k_elements + 1) & ~(size_t)1;
  const size_t panel_elems = n_step * (1 + k_rounded);
  const svbool_t pg16 = svptrue_b16();
  const svfloat16_t vmin = svreinterpret_f16_u16(svdup_n_u16(min_bits));
  const svfloat16_t vmax = svreinterpret_f16_u16(svdup_n_u16(max_bits));

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float16_t* lhs_row = lhs + m_idx * k_elements;
    float16_t* dst_row = (float16_t*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 2 * n_step < n; n_idx += 4 * n_step) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      const float16_t* p1 = p0 + panel_elems;
      const float16_t* p2 = p1 + panel_elems;
      const float16_t* p3 = (n_idx + 3 * n_step < n) ? (p2 + panel_elems) : p0;

      svfloat32_t vacc00 = xnn_pf16_sme_load_bias_half(p0);
      svfloat32_t vacc01 = xnn_pf16_sme_load_bias_half(p0 + vl);
      svfloat32_t vacc10 = xnn_pf16_sme_load_bias_half(p1);
      svfloat32_t vacc11 = xnn_pf16_sme_load_bias_half(p1 + vl);
      svfloat32_t vacc20 = xnn_pf16_sme_load_bias_half(p2);
      svfloat32_t vacc21 = xnn_pf16_sme_load_bias_half(p2 + vl);
      svfloat32_t vacc30 = xnn_pf16_sme_load_bias_half(p3);
      svfloat32_t vacc31 = xnn_pf16_sme_load_bias_half(p3 + vl);

      const float16_t* w0 = p0 + n_step;
      const float16_t* w1 = p1 + n_step;
      const float16_t* w2 = p2 + n_step;
      const float16_t* w3 = p3 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);
#define XNN_PF16_SME_MLA_STEP_4P(U)                                   \
  do {                                                                \
    const svfloat16_t wb00 = svld1_vnum_f16(pg16, w0, 2 * (U) + 0);   \
    const svfloat16_t wb01 = svld1_vnum_f16(pg16, w0, 2 * (U) + 1);   \
    const svfloat16_t wb10 = svld1_vnum_f16(pg16, w1, 2 * (U) + 0);   \
    const svfloat16_t wb11 = svld1_vnum_f16(pg16, w1, 2 * (U) + 1);   \
    const svfloat16_t wb20 = svld1_vnum_f16(pg16, w2, 2 * (U) + 0);   \
    const svfloat16_t wb21 = svld1_vnum_f16(pg16, w2, 2 * (U) + 1);   \
    const svfloat16_t wb30 = svld1_vnum_f16(pg16, w3, 2 * (U) + 0);   \
    const svfloat16_t wb31 = svld1_vnum_f16(pg16, w3, 2 * (U) + 1);   \
    vacc00 = svmlalb_lane_f32(vacc00, wb00, va, 2 * (U) + 0);         \
    vacc01 = svmlalb_lane_f32(vacc01, wb01, va, 2 * (U) + 0);         \
    vacc10 = svmlalb_lane_f32(vacc10, wb10, va, 2 * (U) + 0);         \
    vacc11 = svmlalb_lane_f32(vacc11, wb11, va, 2 * (U) + 0);         \
    vacc20 = svmlalb_lane_f32(vacc20, wb20, va, 2 * (U) + 0);         \
    vacc21 = svmlalb_lane_f32(vacc21, wb21, va, 2 * (U) + 0);         \
    vacc30 = svmlalb_lane_f32(vacc30, wb30, va, 2 * (U) + 0);         \
    vacc31 = svmlalb_lane_f32(vacc31, wb31, va, 2 * (U) + 0);         \
    vacc00 = svmlalt_lane_f32(vacc00, wb00, va, 2 * (U) + 1);         \
    vacc01 = svmlalt_lane_f32(vacc01, wb01, va, 2 * (U) + 1);         \
    vacc10 = svmlalt_lane_f32(vacc10, wb10, va, 2 * (U) + 1);         \
    vacc11 = svmlalt_lane_f32(vacc11, wb11, va, 2 * (U) + 1);         \
    vacc20 = svmlalt_lane_f32(vacc20, wb20, va, 2 * (U) + 1);         \
    vacc21 = svmlalt_lane_f32(vacc21, wb21, va, 2 * (U) + 1);         \
    vacc30 = svmlalt_lane_f32(vacc30, wb30, va, 2 * (U) + 1);         \
    vacc31 = svmlalt_lane_f32(vacc31, wb31, va, 2 * (U) + 1);         \
  } while (0)
        XNN_PF16_SME_MLA_STEP_4P(0);
        XNN_PF16_SME_MLA_STEP_4P(1);
        XNN_PF16_SME_MLA_STEP_4P(2);
        XNN_PF16_SME_MLA_STEP_4P(3);
#undef XNN_PF16_SME_MLA_STEP_4P
        w0 += 8 * n_step;
        w1 += 8 * n_step;
        w2 += 8 * n_step;
        w3 += 8 * n_step;
      }

      for (; k_idx + 2 <= k_elements; k_idx += 2) {
        const svfloat16_t va = svld1rq_f16(
            svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        const svfloat16_t wb00 = svld1_vnum_f16(pg16, w0, 0);
        const svfloat16_t wb01 = svld1_vnum_f16(pg16, w0, 1);
        const svfloat16_t wb10 = svld1_vnum_f16(pg16, w1, 0);
        const svfloat16_t wb11 = svld1_vnum_f16(pg16, w1, 1);
        const svfloat16_t wb20 = svld1_vnum_f16(pg16, w2, 0);
        const svfloat16_t wb21 = svld1_vnum_f16(pg16, w2, 1);
        const svfloat16_t wb30 = svld1_vnum_f16(pg16, w3, 0);
        const svfloat16_t wb31 = svld1_vnum_f16(pg16, w3, 1);

        vacc00 = svmlalb_lane_f32(vacc00, wb00, va, 0);
        vacc01 = svmlalb_lane_f32(vacc01, wb01, va, 0);
        vacc10 = svmlalb_lane_f32(vacc10, wb10, va, 0);
        vacc11 = svmlalb_lane_f32(vacc11, wb11, va, 0);
        vacc20 = svmlalb_lane_f32(vacc20, wb20, va, 0);
        vacc21 = svmlalb_lane_f32(vacc21, wb21, va, 0);
        vacc30 = svmlalb_lane_f32(vacc30, wb30, va, 0);
        vacc31 = svmlalb_lane_f32(vacc31, wb31, va, 0);

        vacc00 = svmlalt_lane_f32(vacc00, wb00, va, 1);
        vacc01 = svmlalt_lane_f32(vacc01, wb01, va, 1);
        vacc10 = svmlalt_lane_f32(vacc10, wb10, va, 1);
        vacc11 = svmlalt_lane_f32(vacc11, wb11, va, 1);
        vacc20 = svmlalt_lane_f32(vacc20, wb20, va, 1);
        vacc21 = svmlalt_lane_f32(vacc21, wb21, va, 1);
        vacc30 = svmlalt_lane_f32(vacc30, wb30, va, 1);
        vacc31 = svmlalt_lane_f32(vacc31, wb31, va, 1);

        w0 += 2 * n_step;
        w1 += 2 * n_step;
        w2 += 2 * n_step;
        w3 += 2 * n_step;
      }

      if (k_idx < k_elements) {
        const svfloat16_t va = svld1rq_f16(
            svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        vacc00 = svmlalb_lane_f32(vacc00, svld1_vnum_f16(pg16, w0, 0), va, 0);
        vacc01 = svmlalb_lane_f32(vacc01, svld1_vnum_f16(pg16, w0, 1), va, 0);
        vacc10 = svmlalb_lane_f32(vacc10, svld1_vnum_f16(pg16, w1, 0), va, 0);
        vacc11 = svmlalb_lane_f32(vacc11, svld1_vnum_f16(pg16, w1, 1), va, 0);
        vacc20 = svmlalb_lane_f32(vacc20, svld1_vnum_f16(pg16, w2, 0), va, 0);
        vacc21 = svmlalb_lane_f32(vacc21, svld1_vnum_f16(pg16, w2, 1), va, 0);
        vacc30 = svmlalb_lane_f32(vacc30, svld1_vnum_f16(pg16, w3, 0), va, 0);
        vacc31 = svmlalb_lane_f32(vacc31, svld1_vnum_f16(pg16, w3, 1), va, 0);
      }

      xnn_pf16_sme_store_panel(vacc00, vacc01, vmin, vmax, dst_row,
                               n_idx + 0 * n_step, n);
      xnn_pf16_sme_store_panel(vacc10, vacc11, vmin, vmax, dst_row,
                               n_idx + 1 * n_step, n);
      xnn_pf16_sme_store_panel(vacc20, vacc21, vmin, vmax, dst_row,
                               n_idx + 2 * n_step, n);
      if (n_idx + 3 * n_step < n) {
        xnn_pf16_sme_store_panel(vacc30, vacc31, vmin, vmax, dst_row,
                                 n_idx + 3 * n_step, n);
      }
    }

    for (; n_idx + n_step < n; n_idx += 2 * n_step) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      const float16_t* p1 = p0 + panel_elems;

      svfloat32_t vacc00 = xnn_pf16_sme_load_bias_half(p0);
      svfloat32_t vacc01 = xnn_pf16_sme_load_bias_half(p0 + vl);
      svfloat32_t vacc10 = xnn_pf16_sme_load_bias_half(p1);
      svfloat32_t vacc11 = xnn_pf16_sme_load_bias_half(p1 + vl);

      const float16_t* w0 = p0 + n_step;
      const float16_t* w1 = p1 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);
#define XNN_PF16_SME_MLA_STEP_2P(U)                                   \
  do {                                                                \
    const svfloat16_t wb00 = svld1_vnum_f16(pg16, w0, 2 * (U) + 0);   \
    const svfloat16_t wb01 = svld1_vnum_f16(pg16, w0, 2 * (U) + 1);   \
    const svfloat16_t wb10 = svld1_vnum_f16(pg16, w1, 2 * (U) + 0);   \
    const svfloat16_t wb11 = svld1_vnum_f16(pg16, w1, 2 * (U) + 1);   \
    vacc00 = svmlalb_lane_f32(vacc00, wb00, va, 2 * (U) + 0);         \
    vacc01 = svmlalb_lane_f32(vacc01, wb01, va, 2 * (U) + 0);         \
    vacc10 = svmlalb_lane_f32(vacc10, wb10, va, 2 * (U) + 0);         \
    vacc11 = svmlalb_lane_f32(vacc11, wb11, va, 2 * (U) + 0);         \
    vacc00 = svmlalt_lane_f32(vacc00, wb00, va, 2 * (U) + 1);         \
    vacc01 = svmlalt_lane_f32(vacc01, wb01, va, 2 * (U) + 1);         \
    vacc10 = svmlalt_lane_f32(vacc10, wb10, va, 2 * (U) + 1);         \
    vacc11 = svmlalt_lane_f32(vacc11, wb11, va, 2 * (U) + 1);         \
  } while (0)
        XNN_PF16_SME_MLA_STEP_2P(0);
        XNN_PF16_SME_MLA_STEP_2P(1);
        XNN_PF16_SME_MLA_STEP_2P(2);
        XNN_PF16_SME_MLA_STEP_2P(3);
#undef XNN_PF16_SME_MLA_STEP_2P
        w0 += 8 * n_step;
        w1 += 8 * n_step;
      }

      for (; k_idx + 2 <= k_elements; k_idx += 2) {
        const svfloat16_t va = svld1rq_f16(
            svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        const svfloat16_t wb00 = svld1_vnum_f16(pg16, w0, 0);
        const svfloat16_t wb01 = svld1_vnum_f16(pg16, w0, 1);
        const svfloat16_t wb10 = svld1_vnum_f16(pg16, w1, 0);
        const svfloat16_t wb11 = svld1_vnum_f16(pg16, w1, 1);

        vacc00 = svmlalb_lane_f32(vacc00, wb00, va, 0);
        vacc01 = svmlalb_lane_f32(vacc01, wb01, va, 0);
        vacc10 = svmlalb_lane_f32(vacc10, wb10, va, 0);
        vacc11 = svmlalb_lane_f32(vacc11, wb11, va, 0);

        vacc00 = svmlalt_lane_f32(vacc00, wb00, va, 1);
        vacc01 = svmlalt_lane_f32(vacc01, wb01, va, 1);
        vacc10 = svmlalt_lane_f32(vacc10, wb10, va, 1);
        vacc11 = svmlalt_lane_f32(vacc11, wb11, va, 1);

        w0 += 2 * n_step;
        w1 += 2 * n_step;
      }

      if (k_idx < k_elements) {
        const svfloat16_t va = svld1rq_f16(
            svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        vacc00 = svmlalb_lane_f32(vacc00, svld1_vnum_f16(pg16, w0, 0), va, 0);
        vacc01 = svmlalb_lane_f32(vacc01, svld1_vnum_f16(pg16, w0, 1), va, 0);
        vacc10 = svmlalb_lane_f32(vacc10, svld1_vnum_f16(pg16, w1, 0), va, 0);
        vacc11 = svmlalb_lane_f32(vacc11, svld1_vnum_f16(pg16, w1, 1), va, 0);
      }

      xnn_pf16_sme_store_panel(vacc00, vacc01, vmin, vmax, dst_row, n_idx, n);
      xnn_pf16_sme_store_panel(vacc10, vacc11, vmin, vmax, dst_row,
                               n_idx + n_step, n);
    }

    if (n_idx < n) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      svfloat32_t vacc00 = xnn_pf16_sme_load_bias_half(p0);
      svfloat32_t vacc01 = xnn_pf16_sme_load_bias_half(p0 + vl);
      const float16_t* w0 = p0 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);
#define XNN_PF16_SME_MLA_STEP_1P(U)                                   \
  do {                                                                \
    const svfloat16_t wb00 = svld1_vnum_f16(pg16, w0, 2 * (U) + 0);   \
    const svfloat16_t wb01 = svld1_vnum_f16(pg16, w0, 2 * (U) + 1);   \
    vacc00 = svmlalb_lane_f32(vacc00, wb00, va, 2 * (U) + 0);         \
    vacc01 = svmlalb_lane_f32(vacc01, wb01, va, 2 * (U) + 0);         \
    vacc00 = svmlalt_lane_f32(vacc00, wb00, va, 2 * (U) + 1);         \
    vacc01 = svmlalt_lane_f32(vacc01, wb01, va, 2 * (U) + 1);         \
  } while (0)
        XNN_PF16_SME_MLA_STEP_1P(0);
        XNN_PF16_SME_MLA_STEP_1P(1);
        XNN_PF16_SME_MLA_STEP_1P(2);
        XNN_PF16_SME_MLA_STEP_1P(3);
#undef XNN_PF16_SME_MLA_STEP_1P
        w0 += 8 * n_step;
      }

      for (; k_idx + 2 <= k_elements; k_idx += 2) {
        const svfloat16_t va = svld1rq_f16(
            svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        const svfloat16_t wb00 = svld1_vnum_f16(pg16, w0, 0);
        const svfloat16_t wb01 = svld1_vnum_f16(pg16, w0, 1);
        vacc00 = svmlalb_lane_f32(vacc00, wb00, va, 0);
        vacc01 = svmlalb_lane_f32(vacc01, wb01, va, 0);
        vacc00 = svmlalt_lane_f32(vacc00, wb00, va, 1);
        vacc01 = svmlalt_lane_f32(vacc01, wb01, va, 1);
        w0 += 2 * n_step;
      }

      if (k_idx < k_elements) {
        const svfloat16_t va = svld1rq_f16(
            svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        vacc00 = svmlalb_lane_f32(vacc00, svld1_vnum_f16(pg16, w0, 0), va, 0);
        vacc01 = svmlalb_lane_f32(vacc01, svld1_vnum_f16(pg16, w0, 1), va, 0);
      }

      xnn_pf16_sme_store_panel(vacc00, vacc01, vmin, vmax, dst_row, n_idx, n);
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  (void)dst_stride_col;
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &minmax_params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &minmax_params->scalar.max, sizeof(uint16_t));
  xnn_pf16_gemm_1x8vl_sme_mla_impl(
      m, n, k / sizeof(xnn_float16), (const float16_t*)lhs_packed,
      (const float16_t*)rhs_packed, dst, dst_stride_row, min_bits, max_bits);
}
#else
#include <assert.h>

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme_get_mr() { return 0; }
size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme_get_nr() { return 0; }
void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
