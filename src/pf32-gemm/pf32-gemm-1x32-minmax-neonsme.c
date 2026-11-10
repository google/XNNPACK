// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla.h"

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() { return 1; }

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla();
}

// Wraps the `kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  kai_run_matmul_clamp_f32_f32_f32p2vlx1b_1x8vl_sme_mla(
      m, n, k / sizeof(float), lhs_packed, /*unused_lhs_stride=*/0, rhs_packed,
      dst, dst_stride_row, /*dst_stride_col=*/sizeof(float),
      minmax_params->scalar.min, minmax_params->scalar.max);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() { return 1; }

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() {
  return 2 * svcntsw();
}

static XNN_INLINE void xnn_pf32_sme_store_panel(
    svfloat32_t vacc0, svfloat32_t vacc1, svfloat32_t vmin, svfloat32_t vmax,
    float* dst_row, size_t col_idx, size_t vl, size_t n) __arm_streaming {
  const svbool_t pg32 = svptrue_b32();
  vacc0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vacc0, vmin), vmax);
  vacc1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vacc1, vmin), vmax);
  svst1_vnum_f32(svwhilelt_b32_u64(col_idx, n), dst_row + col_idx, 0, vacc0);
  svst1_vnum_f32(svwhilelt_b32_u64(col_idx + vl, n), dst_row + col_idx, 1,
                 vacc1);
}

__arm_locally_streaming static XNN_NOINLINE void
xnn_pf32_gemm_1x8vl_sme_mla_impl(size_t m, size_t n, size_t k_elements,
                                 const float* lhs, const float* rhs_packed,
                                 float* dst, size_t dst_stride_row,
                                 float clamp_min, float clamp_max) {
  const size_t vl = svcntw();
  const size_t n_step = 2 * vl;
  const size_t panel_floats = n_step * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float* lhs_row = lhs + m_idx * k_elements;
    float* dst_row = (float*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 2 * n_step < n; n_idx += 4 * n_step) {
      const float* p0 = rhs_packed + (n_idx / n_step) * panel_floats;
      const float* p1 = p0 + panel_floats;
      const float* p2 = p1 + panel_floats;
      const float* p3 = (n_idx + 3 * n_step < n) ? (p2 + panel_floats) : p0;

      svfloat32_t vacc00 = svld1_vnum_f32(pg32, p0, 0);
      svfloat32_t vacc01 = svld1_vnum_f32(pg32, p0, 1);
      svfloat32_t vacc10 = svld1_vnum_f32(pg32, p1, 0);
      svfloat32_t vacc11 = svld1_vnum_f32(pg32, p1, 1);
      svfloat32_t vacc20 = svld1_vnum_f32(pg32, p2, 0);
      svfloat32_t vacc21 = svld1_vnum_f32(pg32, p2, 1);
      svfloat32_t vacc30 = svld1_vnum_f32(pg32, p3, 0);
      svfloat32_t vacc31 = svld1_vnum_f32(pg32, p3, 1);

      const float* w0 = p0 + n_step;
      const float* w1 = p1 + n_step;
      const float* w2 = p2 + n_step;
      const float* w3 = p3 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 4 <= k_elements; k_idx += 4) {
        const svfloat32_t va = svld1rq_f32(pg32, lhs_row + k_idx);
#define XNN_PF32_SME_MLA_4P(U)                                         \
  do {                                                                 \
    vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 2 * (U)), \
                            va, (U));                                  \
    vacc01 = svmla_lane_f32(                                           \
        vacc01, svld1_vnum_f32(pg32, w0, 2 * (U) + 1), va, (U));       \
    vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 2 * (U)), \
                            va, (U));                                  \
    vacc11 = svmla_lane_f32(                                           \
        vacc11, svld1_vnum_f32(pg32, w1, 2 * (U) + 1), va, (U));       \
    vacc20 = svmla_lane_f32(vacc20, svld1_vnum_f32(pg32, w2, 2 * (U)), \
                            va, (U));                                  \
    vacc21 = svmla_lane_f32(                                           \
        vacc21, svld1_vnum_f32(pg32, w2, 2 * (U) + 1), va, (U));       \
    vacc30 = svmla_lane_f32(vacc30, svld1_vnum_f32(pg32, w3, 2 * (U)), \
                            va, (U));                                  \
    vacc31 = svmla_lane_f32(                                           \
        vacc31, svld1_vnum_f32(pg32, w3, 2 * (U) + 1), va, (U));       \
  } while (0)
        XNN_PF32_SME_MLA_4P(0);
        XNN_PF32_SME_MLA_4P(1);
        XNN_PF32_SME_MLA_4P(2);
        XNN_PF32_SME_MLA_4P(3);
#undef XNN_PF32_SME_MLA_4P
        w0 += 4 * n_step;
        w1 += 4 * n_step;
        w2 += 4 * n_step;
        w3 += 4 * n_step;
      }

      if (k_idx < k_elements) {
        const svfloat32_t va =
            svld1rq_f32(svwhilelt_b32_u64(k_idx, k_elements), lhs_row + k_idx);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 0), va, 0);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 1), va, 0);
        vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 0), va, 0);
        vacc11 = svmla_lane_f32(vacc11, svld1_vnum_f32(pg32, w1, 1), va, 0);
        vacc20 = svmla_lane_f32(vacc20, svld1_vnum_f32(pg32, w2, 0), va, 0);
        vacc21 = svmla_lane_f32(vacc21, svld1_vnum_f32(pg32, w2, 1), va, 0);
        vacc30 = svmla_lane_f32(vacc30, svld1_vnum_f32(pg32, w3, 0), va, 0);
        vacc31 = svmla_lane_f32(vacc31, svld1_vnum_f32(pg32, w3, 1), va, 0);
        if (k_idx + 1 < k_elements) {
          vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 2), va, 1);
          vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 3), va, 1);
          vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 2), va, 1);
          vacc11 = svmla_lane_f32(vacc11, svld1_vnum_f32(pg32, w1, 3), va, 1);
          vacc20 = svmla_lane_f32(vacc20, svld1_vnum_f32(pg32, w2, 2), va, 1);
          vacc21 = svmla_lane_f32(vacc21, svld1_vnum_f32(pg32, w2, 3), va, 1);
          vacc30 = svmla_lane_f32(vacc30, svld1_vnum_f32(pg32, w3, 2), va, 1);
          vacc31 = svmla_lane_f32(vacc31, svld1_vnum_f32(pg32, w3, 3), va, 1);
        }
        if (k_idx + 2 < k_elements) {
          vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 4), va, 2);
          vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 5), va, 2);
          vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 4), va, 2);
          vacc11 = svmla_lane_f32(vacc11, svld1_vnum_f32(pg32, w1, 5), va, 2);
          vacc20 = svmla_lane_f32(vacc20, svld1_vnum_f32(pg32, w2, 4), va, 2);
          vacc21 = svmla_lane_f32(vacc21, svld1_vnum_f32(pg32, w2, 5), va, 2);
          vacc30 = svmla_lane_f32(vacc30, svld1_vnum_f32(pg32, w3, 4), va, 2);
          vacc31 = svmla_lane_f32(vacc31, svld1_vnum_f32(pg32, w3, 5), va, 2);
        }
      }

      xnn_pf32_sme_store_panel(vacc00, vacc01, vmin, vmax, dst_row,
                               n_idx + 0 * n_step, vl, n);
      xnn_pf32_sme_store_panel(vacc10, vacc11, vmin, vmax, dst_row,
                               n_idx + 1 * n_step, vl, n);
      xnn_pf32_sme_store_panel(vacc20, vacc21, vmin, vmax, dst_row,
                               n_idx + 2 * n_step, vl, n);
      if (n_idx + 3 * n_step < n) {
        xnn_pf32_sme_store_panel(vacc30, vacc31, vmin, vmax, dst_row,
                                 n_idx + 3 * n_step, vl, n);
      }
    }

    for (; n_idx + n_step < n; n_idx += 2 * n_step) {
      const float* p0 = rhs_packed + (n_idx / n_step) * panel_floats;
      const float* p1 = p0 + panel_floats;

      svfloat32_t vacc00 = svld1_vnum_f32(pg32, p0, 0);
      svfloat32_t vacc01 = svld1_vnum_f32(pg32, p0, 1);
      svfloat32_t vacc10 = svld1_vnum_f32(pg32, p1, 0);
      svfloat32_t vacc11 = svld1_vnum_f32(pg32, p1, 1);

      const float* w0 = p0 + n_step;
      const float* w1 = p1 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 4 <= k_elements; k_idx += 4) {
        const svfloat32_t va = svld1rq_f32(pg32, lhs_row + k_idx);
#define XNN_PF32_SME_MLA_2P(U)                                         \
  do {                                                                 \
    vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 2 * (U)), \
                            va, (U));                                  \
    vacc01 = svmla_lane_f32(                                           \
        vacc01, svld1_vnum_f32(pg32, w0, 2 * (U) + 1), va, (U));       \
    vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 2 * (U)), \
                            va, (U));                                  \
    vacc11 = svmla_lane_f32(                                           \
        vacc11, svld1_vnum_f32(pg32, w1, 2 * (U) + 1), va, (U));       \
  } while (0)
        XNN_PF32_SME_MLA_2P(0);
        XNN_PF32_SME_MLA_2P(1);
        XNN_PF32_SME_MLA_2P(2);
        XNN_PF32_SME_MLA_2P(3);
#undef XNN_PF32_SME_MLA_2P
        w0 += 4 * n_step;
        w1 += 4 * n_step;
      }

      if (k_idx < k_elements) {
        const svfloat32_t va =
            svld1rq_f32(svwhilelt_b32_u64(k_idx, k_elements), lhs_row + k_idx);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 0), va, 0);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 1), va, 0);
        vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 0), va, 0);
        vacc11 = svmla_lane_f32(vacc11, svld1_vnum_f32(pg32, w1, 1), va, 0);
        if (k_idx + 1 < k_elements) {
          vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 2), va, 1);
          vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 3), va, 1);
          vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 2), va, 1);
          vacc11 = svmla_lane_f32(vacc11, svld1_vnum_f32(pg32, w1, 3), va, 1);
        }
        if (k_idx + 2 < k_elements) {
          vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 4), va, 2);
          vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 5), va, 2);
          vacc10 = svmla_lane_f32(vacc10, svld1_vnum_f32(pg32, w1, 4), va, 2);
          vacc11 = svmla_lane_f32(vacc11, svld1_vnum_f32(pg32, w1, 5), va, 2);
        }
      }

      xnn_pf32_sme_store_panel(vacc00, vacc01, vmin, vmax, dst_row, n_idx, vl,
                               n);
      xnn_pf32_sme_store_panel(vacc10, vacc11, vmin, vmax, dst_row,
                               n_idx + n_step, vl, n);
    }

    if (n_idx < n) {
      const float* p0 = rhs_packed + (n_idx / n_step) * panel_floats;
      svfloat32_t vacc00 = svld1_vnum_f32(pg32, p0, 0);
      svfloat32_t vacc01 = svld1_vnum_f32(pg32, p0, 1);
      const float* w0 = p0 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 4 <= k_elements; k_idx += 4) {
        const svfloat32_t va = svld1rq_f32(pg32, lhs_row + k_idx);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 0), va, 0);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 1), va, 0);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 2), va, 1);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 3), va, 1);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 4), va, 2);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 5), va, 2);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 6), va, 3);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 7), va, 3);
        w0 += 4 * n_step;
      }

      if (k_idx < k_elements) {
        const svfloat32_t va =
            svld1rq_f32(svwhilelt_b32_u64(k_idx, k_elements), lhs_row + k_idx);
        vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 0), va, 0);
        vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 1), va, 0);
        if (k_idx + 1 < k_elements) {
          vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 2), va, 1);
          vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 3), va, 1);
        }
        if (k_idx + 2 < k_elements) {
          vacc00 = svmla_lane_f32(vacc00, svld1_vnum_f32(pg32, w0, 4), va, 2);
          vacc01 = svmla_lane_f32(vacc01, svld1_vnum_f32(pg32, w0, 5), va, 2);
        }
      }

      xnn_pf32_sme_store_panel(vacc00, vacc01, vmin, vmax, dst_row, n_idx, vl,
                               n);
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  (void)dst_stride_col;
  xnn_pf32_gemm_1x8vl_sme_mla_impl(
      m, n, k / sizeof(float), (const float*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_mr() { return 0; }
size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme_get_nr() { return 0; }
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
