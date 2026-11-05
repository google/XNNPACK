// Copyright 2024 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/gemm.h"
#include "src/xnnpack/microparams.h"

#if XNN_ENABLE_KLEIDIAI
#include "kai/ukernels/matmul/kai_matmul.h"
#include "kai/ukernels/matmul/kai_matmul_types.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_f32p_f32p/kai_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa.h"

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
  return kai_get_mr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f32_f32p2vlx1_f32p2vlx1biasf32_sme2_mopa();
}

// Wraps the `kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(k % sizeof(float) == 0);
  assert(dst_stride_col == sizeof(float));
  (void)dst_stride_col;
  const size_t k_elements = k / sizeof(float);

  const struct kai_matmul_uker_config config = {0};
  const struct kai_matmul_uker_api api =
      kai_matmul_clamp_f32_f32p4vsx1_f32p4vsx1bf32_8vsx8vs_sme2_mopa();
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
#include <assert.h>
#include <stdint.h>

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
  return 2 * svcntsw();
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
  return 2 * svcntsw();
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pf32_gemm_8vsx8vs_sme2_mopa_impl(size_t m, size_t n, size_t k_elements,
                                     const float* lhs_packed,
                                     const float* rhs_packed, float* dst,
                                     size_t dst_stride_row, float clamp_min,
                                     float clamp_max) {
  const size_t vl = svcntw();
  const size_t lhs_panel_floats = vl * k_elements;
  const size_t rhs_panel_floats = vl * (1 + k_elements);
  const size_t total_w_bytes =
      ((n + vl - 1) / vl) * rhs_panel_floats * sizeof(float);
#if defined(__APPLE__)
  const size_t prfm_min_w_size = 16 * 1024 * 1024;
#else
  const size_t prfm_min_w_size = 1024 * 1024;
#endif
  const int need_prfm = total_w_bytes > prfm_min_w_size;
  const svbool_t pg32 = svptrue_b32();
  const svcount_t pc32 = svptrue_c32();
  const svfloat32_t vones = svdup_n_f32(1.0f);
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; m_idx += 2 * vl) {
    const size_t m_rem = m - m_idx < 2 * vl ? m - m_idx : 2 * vl;
    const float* lhs_p0 = lhs_packed + (m_idx / vl) * lhs_panel_floats;
    const float* lhs_p1 = lhs_p0 + lhs_panel_floats;

    for (size_t n_idx = 0; n_idx < n; n_idx += 2 * vl) {
      if (m_rem <= vl && n_idx + 2 * vl < n) {
        const float* rhs_p0 = rhs_packed + (n_idx / vl) * rhs_panel_floats;
        const float* rhs_p1 = rhs_p0 + rhs_panel_floats;
        const float* rhs_p2 = rhs_p1 + rhs_panel_floats;
        const float* rhs_p3 =
            (n_idx + 3 * vl < n) ? (rhs_p2 + rhs_panel_floats) : rhs_p0;
        const float* b0_ptr = rhs_p0 + vl;
        const float* b1_ptr = rhs_p1 + vl;
        const float* b2_ptr = rhs_p2 + vl;
        const float* b3_ptr = rhs_p3 + vl;

        svzero_za();
        svmopa_za32_f32_m(0, pg32, pg32, vones, svld1_f32(pg32, rhs_p0));
        svmopa_za32_f32_m(1, pg32, pg32, vones, svld1_f32(pg32, rhs_p1));
        svmopa_za32_f32_m(2, pg32, pg32, vones, svld1_f32(pg32, rhs_p2));
        svmopa_za32_f32_m(3, pg32, pg32, vones, svld1_f32(pg32, rhs_p3));

        const float* a0_ptr = lhs_p0;
        size_t k_idx = 0;
        for (; k_idx + 4 <= k_elements; k_idx += 4) {
          const svfloat32x4_t va0 = svld1_f32_x4(pc32, a0_ptr);
          const svfloat32x4_t vb0 = svld1_f32_x4(pc32, b0_ptr);
          const svfloat32x4_t vb1 = svld1_f32_x4(pc32, b1_ptr);
          const svfloat32x4_t vb2 = svld1_f32_x4(pc32, b2_ptr);
          const svfloat32x4_t vb3 = svld1_f32_x4(pc32, b3_ptr);

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb0, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb1, 0));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb2, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb3, 0));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb0, 1));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb1, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb2, 1));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb3, 1));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb0, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb1, 2));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb2, 2));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb3, 2));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb0, 3));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb1, 3));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb2, 3));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb3, 3));

          a0_ptr += 4 * vl;
          b0_ptr += 4 * vl;
          b1_ptr += 4 * vl;
          b2_ptr += 4 * vl;
          b3_ptr += 4 * vl;
        }
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_f32(pg32, a0_ptr);
          const svfloat32_t vb0 = svld1_f32(pg32, b0_ptr);
          const svfloat32_t vb1 = svld1_f32(pg32, b1_ptr);
          const svfloat32_t vb2 = svld1_f32(pg32, b2_ptr);
          const svfloat32_t vb3 = svld1_f32(pg32, b3_ptr);
          a0_ptr += vl;
          b0_ptr += vl;
          b1_ptr += vl;
          b2_ptr += vl;
          b3_ptr += vl;

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
          svmopa_za32_f32_m(2, pg32, pg32, va0, vb2);
          svmopa_za32_f32_m(3, pg32, pg32, va0, vb3);
        }

        const svcount_t p_out4 = svwhilelt_c32_u64(n_idx, n, 4);
        uint8_t* dst_row =
            (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float);
        for (size_t row = 0; row < m_rem; ++row) {
          const svfloat32x4_t vza = svreinterpret_f32_u8_x4(
              svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));
          const svfloat32x4_t vout = svclamp_single_f32_x4(vza, vmin, vmax);
          svst1_f32_x4(p_out4, (float*)dst_row, vout);
          dst_row += dst_stride_row;
        }
        n_idx += 2 * vl;
        continue;
      }

      const float* rhs_p0 = rhs_packed + (n_idx / vl) * rhs_panel_floats;
      const float* rhs_p1 =
          (n_idx + vl < n) ? (rhs_p0 + rhs_panel_floats) : rhs_p0;
      const float* pf0 = NULL;
      const float* pf1 = NULL;
      if (need_prfm) {
        if (n_idx + 2 * vl < n) {
          pf0 = rhs_p0 + 2 * rhs_panel_floats;
          pf1 = (n_idx + 3 * vl < n) ? (pf0 + rhs_panel_floats) : pf0;
        } else if (m_idx + 2 * vl < m) {
          pf0 = rhs_packed;
          pf1 = (vl < n) ? (rhs_packed + rhs_panel_floats) : rhs_packed;
        }
        if (pf0 != NULL) {
          __builtin_prefetch(pf0, 0, 2);
          __builtin_prefetch(pf1, 0, 2);
          pf0 += vl;
          pf1 += vl;
        }
      }

      const float* a0_ptr = lhs_p0;
      const float* b0_ptr = rhs_p0 + vl;
      const float* b1_ptr = rhs_p1 + vl;
      const svfloat32_t vbias0 = svld1_f32(pg32, rhs_p0);
      const svfloat32_t vbias1 = svld1_f32(pg32, rhs_p1);
      svzero_za();
      svmopa_za32_f32_m(0, pg32, pg32, vones, vbias0);
      svmopa_za32_f32_m(1, pg32, pg32, vones, vbias1);

      size_t k_idx = 0;
      if (m_rem <= vl) {
        for (; k_idx + 4 <= k_elements; k_idx += 4) {
          const svfloat32x4_t va0 = svld1_f32_x4(pc32, a0_ptr);
          const svfloat32x4_t vb0 = svld1_f32_x4(pc32, b0_ptr);
          const svfloat32x4_t vb1 = svld1_f32_x4(pc32, b1_ptr);
          a0_ptr += 4 * vl;
          b0_ptr += 4 * vl;
          b1_ptr += 4 * vl;

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb0, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb1, 0));
          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb0, 1));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb1, 1));
          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb0, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb1, 2));
          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb0, 3));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb1, 3));
        }
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_f32(pg32, a0_ptr);
          const svfloat32_t vb0 = svld1_f32(pg32, b0_ptr);
          const svfloat32_t vb1 = svld1_f32(pg32, b1_ptr);
          a0_ptr += vl;
          b0_ptr += vl;
          b1_ptr += vl;
          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
        }
      } else {
        const float* a1_ptr = lhs_p1;
        svmopa_za32_f32_m(2, pg32, pg32, vones, vbias0);
        svmopa_za32_f32_m(3, pg32, pg32, vones, vbias1);
        if (k_elements >= 4) {
          svfloat32x4_t va0 = svld1_f32_x4(pc32, a0_ptr);
          svfloat32x4_t va1 = svld1_f32_x4(pc32, a1_ptr);
          svfloat32x4_t vb0 = svld1_f32_x4(pc32, b0_ptr);
          svfloat32x4_t vb1 = svld1_f32_x4(pc32, b1_ptr);
          a0_ptr += 4 * vl;
          a1_ptr += 4 * vl;
          b0_ptr += 4 * vl;
          b1_ptr += 4 * vl;

          for (; k_idx + 8 <= k_elements; k_idx += 4) {
            const svfloat32_t va0_2 = svget4_f32(va0, 2);
            const svfloat32_t va0_3 = svget4_f32(va0, 3);
            const svfloat32_t vb0_2 = svget4_f32(vb0, 2);
            const svfloat32_t vb0_3 = svget4_f32(vb0, 3);

            svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 0),
                              svget4_f32(vb0, 0));
            svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 0),
                              svget4_f32(vb1, 0));
            svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va1, 0),
                              svget4_f32(vb0, 0));
            svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va1, 0),
                              svget4_f32(vb1, 0));

            svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 1),
                              svget4_f32(vb0, 1));
            svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 1),
                              svget4_f32(vb1, 1));
            svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va1, 1),
                              svget4_f32(vb0, 1));
            svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va1, 1),
                              svget4_f32(vb1, 1));

            va0 = svld1_f32_x4(pc32, a0_ptr);
            vb0 = svld1_f32_x4(pc32, b0_ptr);
            a0_ptr += 4 * vl;
            b0_ptr += 4 * vl;

            const svfloat32_t va1_2 = svget4_f32(va1, 2);
            const svfloat32_t va1_3 = svget4_f32(va1, 3);
            const svfloat32_t vb1_2 = svget4_f32(vb1, 2);
            const svfloat32_t vb1_3 = svget4_f32(vb1, 3);

            svmopa_za32_f32_m(0, pg32, pg32, va0_2, vb0_2);
            svmopa_za32_f32_m(1, pg32, pg32, va0_2, vb1_2);
            svmopa_za32_f32_m(2, pg32, pg32, va1_2, vb0_2);
            svmopa_za32_f32_m(3, pg32, pg32, va1_2, vb1_2);

            svmopa_za32_f32_m(0, pg32, pg32, va0_3, vb0_3);
            svmopa_za32_f32_m(1, pg32, pg32, va0_3, vb1_3);
            svmopa_za32_f32_m(2, pg32, pg32, va1_3, vb0_3);
            svmopa_za32_f32_m(3, pg32, pg32, va1_3, vb1_3);

            va1 = svld1_f32_x4(pc32, a1_ptr);
            vb1 = svld1_f32_x4(pc32, b1_ptr);
            a1_ptr += 4 * vl;
            b1_ptr += 4 * vl;

            if (pf0 != NULL) {
              __builtin_prefetch((const uint8_t*)pf0 + 0, 0, 2);
              __builtin_prefetch((const uint8_t*)pf0 + 64, 0, 2);
              __builtin_prefetch((const uint8_t*)pf0 + 128, 0, 2);
              __builtin_prefetch((const uint8_t*)pf0 + 192, 0, 2);
              __builtin_prefetch((const uint8_t*)pf1 + 0, 0, 2);
              __builtin_prefetch((const uint8_t*)pf1 + 64, 0, 2);
              __builtin_prefetch((const uint8_t*)pf1 + 128, 0, 2);
              __builtin_prefetch((const uint8_t*)pf1 + 192, 0, 2);
              pf0 += 4 * vl;
              pf1 += 4 * vl;
            }
          }

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb0, 0));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 0),
                            svget4_f32(vb1, 0));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va1, 0),
                            svget4_f32(vb0, 0));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va1, 0),
                            svget4_f32(vb1, 0));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb0, 1));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 1),
                            svget4_f32(vb1, 1));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va1, 1),
                            svget4_f32(vb0, 1));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va1, 1),
                            svget4_f32(vb1, 1));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb0, 2));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 2),
                            svget4_f32(vb1, 2));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va1, 2),
                            svget4_f32(vb0, 2));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va1, 2),
                            svget4_f32(vb1, 2));

          svmopa_za32_f32_m(0, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb0, 3));
          svmopa_za32_f32_m(1, pg32, pg32, svget4_f32(va0, 3),
                            svget4_f32(vb1, 3));
          svmopa_za32_f32_m(2, pg32, pg32, svget4_f32(va1, 3),
                            svget4_f32(vb0, 3));
          svmopa_za32_f32_m(3, pg32, pg32, svget4_f32(va1, 3),
                            svget4_f32(vb1, 3));
          k_idx += 4;
        }
        for (; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_f32(pg32, a0_ptr);
          const svfloat32_t va1 = svld1_f32(pg32, a1_ptr);
          const svfloat32_t vb0 = svld1_f32(pg32, b0_ptr);
          const svfloat32_t vb1 = svld1_f32(pg32, b1_ptr);
          a0_ptr += vl;
          a1_ptr += vl;
          b0_ptr += vl;
          b1_ptr += vl;

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
          svmopa_za32_f32_m(2, pg32, pg32, va1, vb0);
          svmopa_za32_f32_m(3, pg32, pg32, va1, vb1);
        }
      }

      const svcount_t p_out2 = svwhilelt_c32_u64(n_idx, n, 2);
      const size_t rows0 = m_rem < vl ? m_rem : vl;
      const size_t paired_rows = m_rem > vl ? (m_rem - vl) : 0;
      uint8_t* dst_top =
          (uint8_t*)dst + m_idx * dst_stride_row + n_idx * sizeof(float);
      uint8_t* dst_bot = dst_top + vl * dst_stride_row;

      size_t row = 0;
      for (; row < paired_rows; ++row) {
        const svfloat32x4_t vza = svreinterpret_f32_u8_x4(
            svread_hor_za8_u8_vg4(0, (uint32_t)(row * 4)));
        const svfloat32x4_t vout = svclamp_single_f32_x4(vza, vmin, vmax);
        svst1_f32_x2(p_out2, (float*)dst_top,
                     svcreate2_f32(svget4_f32(vout, 0), svget4_f32(vout, 1)));
        svst1_f32_x2(p_out2, (float*)dst_bot,
                     svcreate2_f32(svget4_f32(vout, 2), svget4_f32(vout, 3)));
        dst_top += dst_stride_row;
        dst_bot += dst_stride_row;
      }
      for (; row < rows0; ++row) {
        const svfloat32x2_t vza = svreinterpret_f32_u8_x2(
            svread_hor_za8_u8_vg2(0, (uint32_t)(row * 4)));
        const svfloat32x2_t vout = svclamp_single_f32_x2(vza, vmin, vmax);
        svst1_f32_x2(p_out2, (float*)dst_top, vout);
        dst_top += dst_stride_row;
      }
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(k % sizeof(float) == 0);
  assert(dst_stride_col == sizeof(float));
  (void)dst_stride_col;
  xnn_pf32_gemm_8vsx8vs_sme2_mopa_impl(
      m, n, k / sizeof(float), (const float*)lhs_packed, rhs_packed, dst,
      dst_stride_row, minmax_params->scalar.min, minmax_params->scalar.max);
}
#else
#include <assert.h>

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_mr() {
  assert(
      "Calling KleidiAI kai_get_mr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

size_t xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2_get_nr() {
  assert(
      "Calling KleidiAI kai_get_nr wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
  return 0;
}

void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const float* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
