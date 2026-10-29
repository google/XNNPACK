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
#include "kai/ukernels/matmul/matmul_clamp_f16_f16_f16p/kai_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot.h"

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr() { return 1; }

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr() {
  return kai_get_nr_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot();
}

// Wraps the `kai_run_matmul_clamp_f16_f16_f16p2vlx2b_1x16vl_sme2_dot`
// GEMM microkernel with a name that is compatible with our tooling.
void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
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

static XNN_INLINE svfloat32x4_t
xnn_pf16_load_bias_2panels(const float16_t* pa, const float16_t* pb,
                           size_t vl) __arm_streaming __arm_preserves("za") {
  const svbool_t pg32 = svptrue_b32();
  const svfloat32_t vb0 = svcvt_f32_f16_x(
      pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)pa)));
  const svfloat32_t vb1 = svcvt_f32_f16_x(
      pg32,
      svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(pa + vl))));
  const svfloat32_t vb2 = svcvt_f32_f16_x(
      pg32, svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)pb)));
  const svfloat32_t vb3 = svcvt_f32_f16_x(
      pg32,
      svreinterpret_f16_u32(svld1uh_u32(pg32, (const uint16_t*)(pb + vl))));
  return svcreate4_f32(vb0, vb1, vb2, vb3);
}

static XNN_INLINE svfloat16x4_t
xnn_pf16_load_w_2panels(const float16_t* wa, const float16_t* wb,
                        int64_t vnum) __arm_streaming __arm_preserves("za") {
  const svcount_t pc16 = svptrue_c16();
  const svfloat16x2_t va = svld1_vnum_f16_x2(pc16, wa, vnum);
  const svfloat16x2_t vb = svld1_vnum_f16_x2(pc16, wb, vnum);
  return svcreate4_f16(svget2_f16(va, 0), svget2_f16(va, 1), svget2_f16(vb, 0),
                       svget2_f16(vb, 1));
}

static XNN_INLINE void xnn_pf16_store_2panels(
    svfloat32x4_t vacc, svfloat16_t vmin, svfloat16_t vmax, float16_t* dst_row,
    size_t col_idx, size_t n) __arm_streaming __arm_preserves("za") {
  const svfloat16_t vh01 =
      svcvt_f16_f32_x2(svcreate2_f32(svget4_f32(vacc, 0), svget4_f32(vacc, 1)));
  const svfloat16_t vh23 =
      svcvt_f16_f32_x2(svcreate2_f32(svget4_f32(vacc, 2), svget4_f32(vacc, 3)));
  const svfloat16x2_t vout =
      svclamp_single_f16_x2(svcreate2_f16(vh01, vh23), vmin, vmax);
  svst1_f16_x2(svwhilelt_c16_u64(col_idx, n, 2), dst_row + col_idx, vout);
}

__arm_locally_streaming __arm_new("za") static XNN_NOINLINE void
xnn_pf16_gemm_1x16vl_sme2_dot_impl(size_t m, size_t n, size_t k_elements,
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
    for (; n_idx + 6 * n_step < n; n_idx += 8 * n_step) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      const float16_t* p1 = p0 + panel_elems;
      const float16_t* p2 = p1 + panel_elems;
      const float16_t* p3 = p2 + panel_elems;
      const float16_t* p4 = p3 + panel_elems;
      const float16_t* p5 = p4 + panel_elems;
      const float16_t* p6 = p5 + panel_elems;
      const float16_t* p7 = (n_idx + 7 * n_step < n) ? (p6 + panel_elems) : p0;

      svwrite_za32_f32_vg1x4(0, xnn_pf16_load_bias_2panels(p0, p1, vl));
      svwrite_za32_f32_vg1x4(1, xnn_pf16_load_bias_2panels(p2, p3, vl));
      svwrite_za32_f32_vg1x4(2, xnn_pf16_load_bias_2panels(p4, p5, vl));
      svwrite_za32_f32_vg1x4(3, xnn_pf16_load_bias_2panels(p6, p7, vl));

      const float16_t* w0 = p0 + n_step;
      const float16_t* w1 = p1 + n_step;
      const float16_t* w2 = p2 + n_step;
      const float16_t* w3 = p3 + n_step;
      const float16_t* w4 = p4 + n_step;
      const float16_t* w5 = p5 + n_step;
      const float16_t* w6 = p6 + n_step;
      const float16_t* w7 = p7 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);

        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(2, xnn_pf16_load_w_2panels(w4, w5, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(3, xnn_pf16_load_w_2panels(w6, w7, 0), va, 0);

        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 2), va, 1);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 2), va, 1);
        svdot_lane_za32_f16_vg1x4(2, xnn_pf16_load_w_2panels(w4, w5, 2), va, 1);
        svdot_lane_za32_f16_vg1x4(3, xnn_pf16_load_w_2panels(w6, w7, 2), va, 1);

        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 4), va, 2);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 4), va, 2);
        svdot_lane_za32_f16_vg1x4(2, xnn_pf16_load_w_2panels(w4, w5, 4), va, 2);
        svdot_lane_za32_f16_vg1x4(3, xnn_pf16_load_w_2panels(w6, w7, 4), va, 2);

        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 6), va, 3);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 6), va, 3);
        svdot_lane_za32_f16_vg1x4(2, xnn_pf16_load_w_2panels(w4, w5, 6), va, 3);
        svdot_lane_za32_f16_vg1x4(3, xnn_pf16_load_w_2panels(w6, w7, 6), va, 3);

        w0 += 8 * n_step;
        w1 += 8 * n_step;
        w2 += 8 * n_step;
        w3 += 8 * n_step;
        w4 += 8 * n_step;
        w5 += 8 * n_step;
        w6 += 8 * n_step;
        w7 += 8 * n_step;
      }

      for (; k_idx < k_elements; k_idx += 2) {
        const svfloat16_t va =
            svld1rq_f16(svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(2, xnn_pf16_load_w_2panels(w4, w5, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(3, xnn_pf16_load_w_2panels(w6, w7, 0), va, 0);
        w0 += 2 * n_step;
        w1 += 2 * n_step;
        w2 += 2 * n_step;
        w3 += 2 * n_step;
        w4 += 2 * n_step;
        w5 += 2 * n_step;
        w6 += 2 * n_step;
        w7 += 2 * n_step;
      }

      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(0), vmin, vmax, dst_row,
                             n_idx + 0 * n_step, n);
      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(1), vmin, vmax, dst_row,
                             n_idx + 2 * n_step, n);
      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(2), vmin, vmax, dst_row,
                             n_idx + 4 * n_step, n);
      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(3), vmin, vmax, dst_row,
                             n_idx + 6 * n_step, n);
    }

    for (; n_idx + 2 * n_step < n; n_idx += 4 * n_step) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      const float16_t* p1 = p0 + panel_elems;
      const float16_t* p2 = p1 + panel_elems;
      const float16_t* p3 = (n_idx + 3 * n_step < n) ? (p2 + panel_elems) : p0;

      svwrite_za32_f32_vg1x4(0, xnn_pf16_load_bias_2panels(p0, p1, vl));
      svwrite_za32_f32_vg1x4(1, xnn_pf16_load_bias_2panels(p2, p3, vl));

      const float16_t* w0 = p0 + n_step;
      const float16_t* w1 = p1 + n_step;
      const float16_t* w2 = p2 + n_step;
      const float16_t* w3 = p3 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 2), va, 1);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 2), va, 1);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 4), va, 2);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 4), va, 2);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 6), va, 3);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 6), va, 3);
        w0 += 8 * n_step;
        w1 += 8 * n_step;
        w2 += 8 * n_step;
        w3 += 8 * n_step;
      }

      for (; k_idx < k_elements; k_idx += 2) {
        const svfloat16_t va =
            svld1rq_f16(svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(1, xnn_pf16_load_w_2panels(w2, w3, 0), va, 0);
        w0 += 2 * n_step;
        w1 += 2 * n_step;
        w2 += 2 * n_step;
        w3 += 2 * n_step;
      }

      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(0), vmin, vmax, dst_row,
                             n_idx + 0 * n_step, n);
      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(1), vmin, vmax, dst_row,
                             n_idx + 2 * n_step, n);
    }

    if (n_idx < n) {
      const float16_t* p0 = rhs_packed + (n_idx / n_step) * panel_elems;
      const float16_t* p1 = (n_idx + n_step < n) ? (p0 + panel_elems) : p0;

      svwrite_za32_f32_vg1x4(0, xnn_pf16_load_bias_2panels(p0, p1, vl));

      const float16_t* w0 = p0 + n_step;
      const float16_t* w1 = p1 + n_step;

      size_t k_idx = 0;
      for (; k_idx + 8 <= k_elements; k_idx += 8) {
        const svfloat16_t va = svld1rq_f16(pg16, lhs_row + k_idx);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 0), va, 0);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 2), va, 1);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 4), va, 2);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 6), va, 3);
        w0 += 8 * n_step;
        w1 += 8 * n_step;
      }

      for (; k_idx < k_elements; k_idx += 2) {
        const svfloat16_t va =
            svld1rq_f16(svwhilelt_b16_u64(k_idx, k_elements), lhs_row + k_idx);
        svdot_lane_za32_f16_vg1x4(0, xnn_pf16_load_w_2panels(w0, w1, 0), va, 0);
        w0 += 2 * n_step;
        w1 += 2 * n_step;
      }

      xnn_pf16_store_2panels(svread_za32_f32_vg1x4(0), vmin, vmax, dst_row,
                             n_idx, n);
    }
  }
}

void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  (void)dst_stride_col;
  uint16_t min_bits;
  uint16_t max_bits;
  memcpy(&min_bits, &minmax_params->scalar.min, sizeof(uint16_t));
  memcpy(&max_bits, &minmax_params->scalar.max, sizeof(uint16_t));
  xnn_pf16_gemm_1x16vl_sme2_dot_impl(
      m, n, k / sizeof(xnn_float16), (const float16_t*)lhs_packed,
      (const float16_t*)rhs_packed, dst, dst_stride_row, min_bits, max_bits);
}
#else
#include <assert.h>

size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_mr() { return 0; }
size_t xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2_get_nr() { return 0; }
void xnn_pf16_gemm_minmax_ukernel_1x32c2__neonsme2(
    size_t m, size_t n, size_t k, const void* lhs_packed,
    const void* rhs_packed, void* dst, size_t dst_stride_row,
    size_t dst_stride_col, const struct xnn_f16_minmax_params* minmax_params) {
  assert(
      "Calling KleidiAI microkernel wrapper, but XNNPACK was compiled without "
      "`XNN_ENABLE_KLEIDIAI`." &&
      0);
}
#endif  // XNN_ENABLE_KLEIDIAI
