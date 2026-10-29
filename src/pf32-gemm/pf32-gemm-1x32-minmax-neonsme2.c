// Copyright 2024 Google LLC
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
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32_f32p/kai_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla.h"

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2_get_mr() { return 1; }

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32_f32p2vlx1b_1x16vl_sme2_mla();
}

// Wraps the `kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(m == 1);
  assert(k % sizeof(float) == 0);
  assert(dst_stride_col == sizeof(float));
  (void)dst_stride_col;

  const size_t k_elements = k / sizeof(float);
  const struct kai_matmul_uker_config config = {0};
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_f32_f32_f32p4vsx1bf32_1x32vs_sme2_mla();
  const struct kai_matmul_uker_lhs_dim_args lhs_shape = {m, k_elements};
  const struct kai_matmul_uker_rhs_dim_args rhs_shape = {n, k_elements};
  struct kai_matmul_uker_args args = {
      .flags = KAI_MATMUL_UKER_FLAGS_ARGS_CLAMP,
      .shape.m = m,
      .shape.n = n,
      .shape.k = k_elements,

      .operand.dst.ptr = dst,
      .operand.dst.stride.m = dst_stride_row,
      .operand.lhs.ptr = lhs_packed,
      .operand.lhs.stride = api.get_lhs_stride(&config, &lhs_shape),
      .operand.rhs.ptr = rhs_packed,
      .operand.rhs.stride = api.get_rhs_stride(&config, &rhs_shape),

      .activation.clamp.min_ptr = &minmax_params->scalar.min,
      .activation.clamp.max_ptr = &minmax_params->scalar.max,
  };
  api.run(&config, &args);
}
#elif XNN_ENABLE_ARM_SME_ACLE && XNN_ARCH_ARM64
#include <arm_sme.h>
#include <stdint.h>

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2_get_mr() { return 1; }

size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pf32_gemm_1x32vs_sme2_mla_impl(size_t m, size_t n, size_t k_elements,
                                   const float* lhs, const float* rhs_packed,
                                   float* dst, size_t dst_stride_row,
                                   float clamp_min, float clamp_max) {
  const size_t vl = svcntw();
  const size_t panel_floats = vl * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc32 = svptrue_c32();
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; ++m_idx) {
    const float* lhs_row = lhs + m_idx * k_elements;
    float* dst_row = (float*)((uintptr_t)dst + m_idx * dst_stride_row);

    size_t n_idx = 0;
    for (; n_idx + 4 * vl < n; n_idx += 8 * vl) {
      const float* p0 = rhs_packed + (n_idx / vl) * panel_floats;
      const float* p1 = p0 + panel_floats;
      const float* p2 = p1 + panel_floats;
      const float* p3 = p2 + panel_floats;
      const float* p4 = p3 + panel_floats;
      const float* p5 = (n_idx + 5 * vl < n) ? (p4 + panel_floats) : p0;
      const float* p6 = (n_idx + 6 * vl < n) ? (p4 + 2 * panel_floats) : p0;
      const float* p7 = (n_idx + 7 * vl < n) ? (p6 + panel_floats) : p0;

      svwrite_za32_f32_vg1x4(
          0, svcreate4_f32(svld1_f32(pg32, p0), svld1_f32(pg32, p1),
                           svld1_f32(pg32, p2), svld1_f32(pg32, p3)));
      svwrite_za32_f32_vg1x4(
          1, svcreate4_f32(svld1_f32(pg32, p4), svld1_f32(pg32, p5),
                           svld1_f32(pg32, p6), svld1_f32(pg32, p7)));

      const float* w0 = p0 + vl;
      const float* w1 = p1 + vl;
      const float* w2 = p2 + vl;
      const float* w3 = p3 + vl;
      const float* w4 = p4 + vl;
      const float* w5 = p5 + vl;
      const float* w6 = p6 + vl;
      const float* w7 = p7 + vl;

      size_t k_idx = 0;
      for (; k_idx + 4 <= k_elements; k_idx += 4) {
        const svfloat32_t va = svld1rq_f32(pg32, lhs_row + k_idx);

        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 0),
                                                svld1_vnum_f32(pg32, w1, 0),
                                                svld1_vnum_f32(pg32, w2, 0),
                                                svld1_vnum_f32(pg32, w3, 0)),
                                  va, 0);
        svmla_lane_za32_f32_vg1x4(1,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w4, 0),
                                                svld1_vnum_f32(pg32, w5, 0),
                                                svld1_vnum_f32(pg32, w6, 0),
                                                svld1_vnum_f32(pg32, w7, 0)),
                                  va, 0);

        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 1),
                                                svld1_vnum_f32(pg32, w1, 1),
                                                svld1_vnum_f32(pg32, w2, 1),
                                                svld1_vnum_f32(pg32, w3, 1)),
                                  va, 1);
        svmla_lane_za32_f32_vg1x4(1,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w4, 1),
                                                svld1_vnum_f32(pg32, w5, 1),
                                                svld1_vnum_f32(pg32, w6, 1),
                                                svld1_vnum_f32(pg32, w7, 1)),
                                  va, 1);

        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 2),
                                                svld1_vnum_f32(pg32, w1, 2),
                                                svld1_vnum_f32(pg32, w2, 2),
                                                svld1_vnum_f32(pg32, w3, 2)),
                                  va, 2);
        svmla_lane_za32_f32_vg1x4(1,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w4, 2),
                                                svld1_vnum_f32(pg32, w5, 2),
                                                svld1_vnum_f32(pg32, w6, 2),
                                                svld1_vnum_f32(pg32, w7, 2)),
                                  va, 2);

        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 3),
                                                svld1_vnum_f32(pg32, w1, 3),
                                                svld1_vnum_f32(pg32, w2, 3),
                                                svld1_vnum_f32(pg32, w3, 3)),
                                  va, 3);
        svmla_lane_za32_f32_vg1x4(1,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w4, 3),
                                                svld1_vnum_f32(pg32, w5, 3),
                                                svld1_vnum_f32(pg32, w6, 3),
                                                svld1_vnum_f32(pg32, w7, 3)),
                                  va, 3);

        w0 += 4 * vl;
        w1 += 4 * vl;
        w2 += 4 * vl;
        w3 += 4 * vl;
        w4 += 4 * vl;
        w5 += 4 * vl;
        w6 += 4 * vl;
        w7 += 4 * vl;
      }

      if (k_idx < k_elements) {
        const svfloat32_t va =
            svld1rq_f32(svwhilelt_b32_u64(k_idx, k_elements), lhs_row + k_idx);
        svmla_lane_za32_f32_vg1x4(
            0,
            svcreate4_f32(svld1_f32(pg32, w0), svld1_f32(pg32, w1),
                          svld1_f32(pg32, w2), svld1_f32(pg32, w3)),
            va, 0);
        svmla_lane_za32_f32_vg1x4(
            1,
            svcreate4_f32(svld1_f32(pg32, w4), svld1_f32(pg32, w5),
                          svld1_f32(pg32, w6), svld1_f32(pg32, w7)),
            va, 0);
        if (k_idx + 1 < k_elements) {
          svmla_lane_za32_f32_vg1x4(0,
                                    svcreate4_f32(svld1_vnum_f32(pg32, w0, 1),
                                                  svld1_vnum_f32(pg32, w1, 1),
                                                  svld1_vnum_f32(pg32, w2, 1),
                                                  svld1_vnum_f32(pg32, w3, 1)),
                                    va, 1);
          svmla_lane_za32_f32_vg1x4(1,
                                    svcreate4_f32(svld1_vnum_f32(pg32, w4, 1),
                                                  svld1_vnum_f32(pg32, w5, 1),
                                                  svld1_vnum_f32(pg32, w6, 1),
                                                  svld1_vnum_f32(pg32, w7, 1)),
                                    va, 1);
        }
        if (k_idx + 2 < k_elements) {
          svmla_lane_za32_f32_vg1x4(0,
                                    svcreate4_f32(svld1_vnum_f32(pg32, w0, 2),
                                                  svld1_vnum_f32(pg32, w1, 2),
                                                  svld1_vnum_f32(pg32, w2, 2),
                                                  svld1_vnum_f32(pg32, w3, 2)),
                                    va, 2);
          svmla_lane_za32_f32_vg1x4(1,
                                    svcreate4_f32(svld1_vnum_f32(pg32, w4, 2),
                                                  svld1_vnum_f32(pg32, w5, 2),
                                                  svld1_vnum_f32(pg32, w6, 2),
                                                  svld1_vnum_f32(pg32, w7, 2)),
                                    va, 2);
        }
      }

      const svfloat32x4_t vout0 =
          svclamp_single_f32_x4(svread_za32_f32_vg1x4(0), vmin, vmax);
      const svfloat32x4_t vout1 =
          svclamp_single_f32_x4(svread_za32_f32_vg1x4(1), vmin, vmax);
      svst1_f32_x4(pc32, dst_row + n_idx, vout0);
      svst1_f32_x4(svwhilelt_c32_u64(n_idx + 4 * vl, n, 4),
                   dst_row + n_idx + 4 * vl, vout1);
    }

    for (; n_idx < n; n_idx += 4 * vl) {
      const float* p0 = rhs_packed + (n_idx / vl) * panel_floats;
      const float* p1 = (n_idx + 1 * vl < n) ? (p0 + panel_floats) : p0;
      const float* p2 = (n_idx + 2 * vl < n) ? (p0 + 2 * panel_floats) : p0;
      const float* p3 = (n_idx + 3 * vl < n) ? (p2 + panel_floats) : p0;

      svwrite_za32_f32_vg1x4(
          0, svcreate4_f32(svld1_f32(pg32, p0), svld1_f32(pg32, p1),
                           svld1_f32(pg32, p2), svld1_f32(pg32, p3)));

      const float* w0 = p0 + vl;
      const float* w1 = p1 + vl;
      const float* w2 = p2 + vl;
      const float* w3 = p3 + vl;

      size_t k_idx = 0;
      for (; k_idx + 4 <= k_elements; k_idx += 4) {
        const svfloat32_t va = svld1rq_f32(pg32, lhs_row + k_idx);

        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 0),
                                                svld1_vnum_f32(pg32, w1, 0),
                                                svld1_vnum_f32(pg32, w2, 0),
                                                svld1_vnum_f32(pg32, w3, 0)),
                                  va, 0);
        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 1),
                                                svld1_vnum_f32(pg32, w1, 1),
                                                svld1_vnum_f32(pg32, w2, 1),
                                                svld1_vnum_f32(pg32, w3, 1)),
                                  va, 1);
        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 2),
                                                svld1_vnum_f32(pg32, w1, 2),
                                                svld1_vnum_f32(pg32, w2, 2),
                                                svld1_vnum_f32(pg32, w3, 2)),
                                  va, 2);
        svmla_lane_za32_f32_vg1x4(0,
                                  svcreate4_f32(svld1_vnum_f32(pg32, w0, 3),
                                                svld1_vnum_f32(pg32, w1, 3),
                                                svld1_vnum_f32(pg32, w2, 3),
                                                svld1_vnum_f32(pg32, w3, 3)),
                                  va, 3);

        w0 += 4 * vl;
        w1 += 4 * vl;
        w2 += 4 * vl;
        w3 += 4 * vl;
      }

      if (k_idx < k_elements) {
        const svfloat32_t va =
            svld1rq_f32(svwhilelt_b32_u64(k_idx, k_elements), lhs_row + k_idx);
        svmla_lane_za32_f32_vg1x4(
            0,
            svcreate4_f32(svld1_f32(pg32, w0), svld1_f32(pg32, w1),
                          svld1_f32(pg32, w2), svld1_f32(pg32, w3)),
            va, 0);
        if (k_idx + 1 < k_elements) {
          svmla_lane_za32_f32_vg1x4(0,
                                    svcreate4_f32(svld1_vnum_f32(pg32, w0, 1),
                                                  svld1_vnum_f32(pg32, w1, 1),
                                                  svld1_vnum_f32(pg32, w2, 1),
                                                  svld1_vnum_f32(pg32, w3, 1)),
                                    va, 1);
        }
        if (k_idx + 2 < k_elements) {
          svmla_lane_za32_f32_vg1x4(0,
                                    svcreate4_f32(svld1_vnum_f32(pg32, w0, 2),
                                                  svld1_vnum_f32(pg32, w1, 2),
                                                  svld1_vnum_f32(pg32, w2, 2),
                                                  svld1_vnum_f32(pg32, w3, 2)),
                                    va, 2);
        }
      }

      const svfloat32x4_t vout0 =
          svclamp_single_f32_x4(svread_za32_f32_vg1x4(0), vmin, vmax);
      svst1_f32_x4(svwhilelt_c32_u64(n_idx, n, 4), dst_row + n_idx, vout0);
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(m == 1);
  assert(k % sizeof(float) == 0);
  assert(dst_stride_col == sizeof(float));
  (void)dst_stride_col;
  xnn_pf32_gemm_1x32vs_sme2_mla_impl(
      m, n, k / sizeof(float), (const float*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2_get_mr() { return 0; }
size_t xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2_get_nr() { return 0; }
void xnn_pf32_gemm_minmax_ukernel_1x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
