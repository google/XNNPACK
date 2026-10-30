// Copyright 2024 Google LLC
// Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stddef.h>

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
    const void* rhs_packed, float* dst, size_t dst_stride_row,
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

__arm_locally_streaming __arm_new(
    "za") static void xnn_pf32_gemm_8vsx8vs_sme_mopa_impl(size_t m, size_t n,
                                                          size_t k_elements,
                                                          const float*
                                                              lhs_packed,
                                                          const float*
                                                              rhs_packed,
                                                          float* dst,
                                                          size_t dst_stride_row,
                                                          float clamp_min,
                                                          float clamp_max) {
  const size_t vl = svcntw();
  const size_t lhs_panel_floats = vl * k_elements;
  const size_t rhs_panel_floats = vl * (1 + k_elements);
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vones = svdup_n_f32(1.0f);
  const svfloat32_t vmin = svdup_n_f32(clamp_min);
  const svfloat32_t vmax = svdup_n_f32(clamp_max);

  for (size_t m_idx = 0; m_idx < m; m_idx += 2 * vl) {
    const size_t m_rem = m - m_idx < 2 * vl ? m - m_idx : 2 * vl;
    const float* lhs_p0 = lhs_packed + (m_idx / vl) * lhs_panel_floats;
    const float* lhs_p1 = lhs_p0 + lhs_panel_floats;

    for (size_t n_idx = 0; n_idx < n; n_idx += 2 * vl) {
      if (m_rem <= vl && n_idx + 2 * vl < n) {
        const size_t n_rem4 = n - n_idx < 4 * vl ? n - n_idx : 4 * vl;
        const float* rhs_p0 = rhs_packed + (n_idx / vl) * rhs_panel_floats;
        const float* rhs_p1 = rhs_p0 + rhs_panel_floats;
        const float* rhs_p2 = rhs_p1 + rhs_panel_floats;
        const float* rhs_p3 = rhs_p2 + rhs_panel_floats;
        const float* rhs_w0 = rhs_p0 + vl;
        const float* rhs_w1 = rhs_p1 + vl;
        const float* rhs_w2 = rhs_p2 + vl;
        const float* rhs_w3 = rhs_p3 + vl;
        const svfloat32_t vbias0 = svld1_f32(pg32, rhs_p0);
        const svfloat32_t vbias1 = svld1_f32(pg32, rhs_p1);
        const svfloat32_t vbias2 = svld1_f32(pg32, rhs_p2);
        const svfloat32_t vbias3 =
            n_rem4 > 3 * vl ? svld1_f32(pg32, rhs_p3) : svdup_n_f32(0.0f);
        svzero_za();
        svmopa_za32_f32_m(0, pg32, pg32, vones, vbias0);
        svmopa_za32_f32_m(1, pg32, pg32, vones, vbias1);
        svmopa_za32_f32_m(2, pg32, pg32, vones, vbias2);
        svmopa_za32_f32_m(3, pg32, pg32, vones, vbias3);

        if (n_rem4 > 3 * vl) {
          const float* a0_ptr = lhs_p0;
          const float* b0_ptr = rhs_w0;
          const float* b1_ptr = rhs_w1;
          const float* b2_ptr = rhs_w2;
          const float* b3_ptr = rhs_w3;
          size_t k_idx = 0;
          for (; k_idx + 4 <= k_elements; k_idx += 4) {
#pragma clang loop unroll(full)
            for (int u = 0; u < 4; ++u) {
              const svfloat32_t va0 = svld1_vnum_f32(pg32, a0_ptr, u);
              const svfloat32_t vb0 = svld1_vnum_f32(pg32, b0_ptr, u);
              const svfloat32_t vb1 = svld1_vnum_f32(pg32, b1_ptr, u);
              const svfloat32_t vb2 = svld1_vnum_f32(pg32, b2_ptr, u);
              const svfloat32_t vb3 = svld1_vnum_f32(pg32, b3_ptr, u);

              svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
              svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
              svmopa_za32_f32_m(2, pg32, pg32, va0, vb2);
              svmopa_za32_f32_m(3, pg32, pg32, va0, vb3);
            }
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
        } else {
          for (size_t k_idx = 0; k_idx < k_elements; ++k_idx) {
            const svfloat32_t va0 = svld1_f32(pg32, lhs_p0 + k_idx * vl);
            const svfloat32_t vb0 = svld1_f32(pg32, rhs_w0 + k_idx * vl);
            const svfloat32_t vb1 = svld1_f32(pg32, rhs_w1 + k_idx * vl);
            const svfloat32_t vb2 = svld1_f32(pg32, rhs_w2 + k_idx * vl);

            svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
            svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
            svmopa_za32_f32_m(2, pg32, pg32, va0, vb2);
          }
        }

        const svbool_t p_out2 = svwhilelt_b32_u64(2 * vl, n_rem4);
        const svbool_t p_out3 = svwhilelt_b32_u64(3 * vl, n_rem4);

        for (size_t row = 0; row < m_rem; ++row) {
          svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
          svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
          svfloat32_t vf2 = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
          vf2 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf2, vmin), vmax);
          float* dst_ptr =
              (float*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
          svst1_vnum_f32(pg32, dst_ptr, 0, vf0);
          svst1_vnum_f32(pg32, dst_ptr, 1, vf1);
          svst1_vnum_f32(p_out2, dst_ptr, 2, vf2);
          if (n_rem4 > 3 * vl) {
            svfloat32_t vf3 =
                svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
            vf3 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf3, vmin), vmax);
            svst1_vnum_f32(p_out3, dst_ptr, 3, vf3);
          }
        }
        n_idx += 2 * vl;
        continue;
      }

      const size_t n_rem = n - n_idx < 2 * vl ? n - n_idx : 2 * vl;
      const float* rhs_p0 = rhs_packed + (n_idx / vl) * rhs_panel_floats;
      const float* rhs_p1 = rhs_p0 + rhs_panel_floats;
      const float* rhs_w0 = rhs_p0 + vl;
      const float* rhs_w1 = rhs_p1 + vl;
      const svfloat32_t vbias0 = svld1_f32(pg32, rhs_p0);
      const svfloat32_t vbias1 =
          n_rem > vl ? svld1_f32(pg32, rhs_p1) : svdup_n_f32(0.0f);
      svzero_za();
      svmopa_za32_f32_m(0, pg32, pg32, vones, vbias0);
      svmopa_za32_f32_m(1, pg32, pg32, vones, vbias1);
      svmopa_za32_f32_m(2, pg32, pg32, vones, vbias0);
      svmopa_za32_f32_m(3, pg32, pg32, vones, vbias1);

      if (m_rem > vl && n_rem > vl) {
        const float* a0_ptr = lhs_p0;
        const float* a1_ptr = lhs_p1;
        const float* b0_ptr = rhs_w0;
        const float* b1_ptr = rhs_w1;
        size_t k_idx = 0;
        for (; k_idx + 4 <= k_elements; k_idx += 4) {
#pragma clang loop unroll(full)
          for (int u = 0; u < 4; ++u) {
            const svfloat32_t va0 = svld1_vnum_f32(pg32, a0_ptr, u);
            const svfloat32_t va1 = svld1_vnum_f32(pg32, a1_ptr, u);
            const svfloat32_t vb0 = svld1_vnum_f32(pg32, b0_ptr, u);
            const svfloat32_t vb1 = svld1_vnum_f32(pg32, b1_ptr, u);

            svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
            svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
            svmopa_za32_f32_m(2, pg32, pg32, va1, vb0);
            svmopa_za32_f32_m(3, pg32, pg32, va1, vb1);
          }
          a0_ptr += 4 * vl;
          a1_ptr += 4 * vl;
          b0_ptr += 4 * vl;
          b1_ptr += 4 * vl;
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
      } else if (m_rem > vl) {
        for (size_t k_idx = 0; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_f32(pg32, lhs_p0 + k_idx * vl);
          const svfloat32_t va1 = svld1_f32(pg32, lhs_p1 + k_idx * vl);
          const svfloat32_t vb0 = svld1_f32(pg32, rhs_w0 + k_idx * vl);

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(2, pg32, pg32, va1, vb0);
        }
      } else if (n_rem > vl) {
        for (size_t k_idx = 0; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_f32(pg32, lhs_p0 + k_idx * vl);
          const svfloat32_t vb0 = svld1_f32(pg32, rhs_w0 + k_idx * vl);
          const svfloat32_t vb1 = svld1_f32(pg32, rhs_w1 + k_idx * vl);

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
          svmopa_za32_f32_m(1, pg32, pg32, va0, vb1);
        }
      } else {
        for (size_t k_idx = 0; k_idx < k_elements; ++k_idx) {
          const svfloat32_t va0 = svld1_f32(pg32, lhs_p0 + k_idx * vl);
          const svfloat32_t vb0 = svld1_f32(pg32, rhs_w0 + k_idx * vl);

          svmopa_za32_f32_m(0, pg32, pg32, va0, vb0);
        }
      }

      const svbool_t p_out0 = svwhilelt_b32_u64(0, n_rem);
      const svbool_t p_out1 = svwhilelt_b32_u64(vl, n_rem);

      const size_t rows0 = m_rem < vl ? m_rem : vl;
      for (size_t row = 0; row < rows0; ++row) {
        svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 0, row);
        svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 1, row);
        vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
        vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
        float* dst_ptr =
            (float*)((uintptr_t)dst + (m_idx + row) * dst_stride_row) + n_idx;
        svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
        svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
      }

      if (m_rem > vl) {
        const size_t rows1 = m_rem - vl;
        for (size_t row = 0; row < rows1; ++row) {
          svfloat32_t vf0 = svread_hor_za32_f32_m(svundef_f32(), pg32, 2, row);
          svfloat32_t vf1 = svread_hor_za32_f32_m(svundef_f32(), pg32, 3, row);
          vf0 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf0, vmin), vmax);
          vf1 = svmin_f32_m(pg32, svmax_f32_m(pg32, vf1, vmin), vmax);
          float* dst_ptr =
              (float*)((uintptr_t)dst + (m_idx + vl + row) * dst_stride_row) +
              n_idx;
          svst1_vnum_f32(p_out0, dst_ptr, 0, vf0);
          svst1_vnum_f32(p_out1, dst_ptr, 1, vf1);
        }
      }
    }
  }
}

void xnn_pf32_gemm_minmax_ukernel_32x32__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(k % sizeof(float) == 0);
  assert(dst_stride_col == sizeof(float));
  (void)dst_stride_col;
  xnn_pf32_gemm_8vsx8vs_sme_mopa_impl(
      m, n, k / sizeof(float), (const float*)lhs_packed,
      (const float*)rhs_packed, dst, dst_stride_row, minmax_params->scalar.min,
      minmax_params->scalar.max);
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
    const void* rhs_packed, float* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f32_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
