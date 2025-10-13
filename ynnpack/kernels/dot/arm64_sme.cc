// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/arm64_sme.h"

#include <arm_sme.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/dot/arm64_sme_internal.h"
#include "ynnpack/kernels/dot/dot.h"

namespace ynn {

size_t sme_vl(float) {
  if (is_arch_supported(arch_flag::sme) || is_arch_supported(arch_flag::sme2)) {
    return svcnt(float{});
  } else {
    return 0;
  }
}

size_t sme_vl(int32_t) {
  if (is_arch_supported(arch_flag::sme) || is_arch_supported(arch_flag::sme2)) {
    return svcnt(int32_t{});
  } else {
    return 0;
  }
}

namespace {

template <typename TAB, typename TC>
__arm_new("za") __arm_locally_streaming void dot_impl(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  const size_t svl = svcnt(TC{});
  assert(M <= svl);

  // This is how many elements of the k dimension are multiplied and accumulated
  // at once.
  constexpr size_t dot_factor = sizeof(TC) / sizeof(TAB);

  // Masks for the row dimension, for both an input and an output.
  svbool_t m_mask_ab = svwhilelt(0, M * dot_factor, TAB{});
  svbool_t m_mask = svwhilelt(0, M, TC{});

  ptrdiff_t n = N;
  while (n >= svl * 4) {
    // (All-true) masks for the column dimension, for both an input and an
    // output.
    svbool_t n_mask_ab = svptrue(TAB{});
    svbool_t n_mask = svptrue(TC{});

    if (C_in) {
      // Load the output to initialize the tile accumulator.
      // TODO: To improve numerical precision and better match the other
      // kernels, it would be best to initialize this to zero (`svzero_za()`)
      // the tile instead of loading the initial accumulator, and add this
      // later.
      for (size_t m = 0; m < M; ++m) {
        svbool_t p = svpsel_lane_b32(n_mask, m_mask, m);
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_vnum_za32(/*tile=*/0, /*slice=*/m, p, C_in_m, 0);
        svld1_hor_vnum_za32(/*tile=*/1, /*slice=*/m, p, C_in_m, 1);
        svld1_hor_vnum_za32(/*tile=*/2, /*slice=*/m, p, C_in_m, 2);
        svld1_hor_vnum_za32(/*tile=*/3, /*slice=*/m, p, C_in_m, 3);
      }
    } else {
      svzero_za();
    }
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        ptrdiff_t k1 = K1;
        while (k1 > 0) {
          auto a = svld1(m_mask_ab, reinterpret_cast<const TAB*>(A_k1));
          auto b_0 =
              svld1_vnum(n_mask_ab, reinterpret_cast<const TAB*>(B_k1), 0);
          auto b_1 =
              svld1_vnum(n_mask_ab, reinterpret_cast<const TAB*>(B_k1), 1);
          auto b_2 =
              svld1_vnum(n_mask_ab, reinterpret_cast<const TAB*>(B_k1), 2);
          auto b_3 =
              svld1_vnum(n_mask_ab, reinterpret_cast<const TAB*>(B_k1), 3);
          svmopa</*tile=*/0>(m_mask_ab, n_mask_ab, a, b_0);
          svmopa</*tile=*/1>(m_mask_ab, n_mask_ab, a, b_1);
          svmopa</*tile=*/2>(m_mask_ab, n_mask_ab, a, b_2);
          svmopa</*tile=*/3>(m_mask_ab, n_mask_ab, a, b_3);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    // Store the accumulated result back to the output.
    for (size_t m = 0; m < M; ++m) {
      svbool_t p = svpsel_lane_b32(n_mask, m_mask, m);
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_vnum_za32(/*tile=*/0, /*slice=*/m, p, C_out_m, 0);
      svst1_hor_vnum_za32(/*tile=*/1, /*slice=*/m, p, C_out_m, 1);
      svst1_hor_vnum_za32(/*tile=*/2, /*slice=*/m, p, C_out_m, 2);
      svst1_hor_vnum_za32(/*tile=*/3, /*slice=*/m, p, C_out_m, 3);
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(TC) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(TC) * 4);
    B = offset_bytes(B, svl * sizeof(TC) * 4);
    n -= svl * 4;
  }
  while (n > 0) {
    // Masks for the column dimension, for both an input and an output.
    svbool_t n_mask_ab = svwhilelt(0, n * dot_factor, TAB{});
    svbool_t n_mask = svwhilelt(0, n, TC{});

    // Load the output to initialize the tile accumulator.
    // TODO: To improve numerical precision and better match the other kernels,
    // it would be best to initialize this to zero (`svzero_za()`) the tile
    // instead of loading the initial accumulator, and add this later.
    if (C_in) {
      for (size_t m = 0; m < M; m += 4) {
        svbool_t p0 = svpsel_lane_b32(n_mask, m_mask, m + 0);
        svbool_t p1 = svpsel_lane_b32(n_mask, m_mask, m + 1);
        svbool_t p2 = svpsel_lane_b32(n_mask, m_mask, m + 2);
        svbool_t p3 = svpsel_lane_b32(n_mask, m_mask, m + 3);
        svld1_hor_za32(
            /*tile=*/0, /*slice=*/m + 0, p0,
            offset_bytes(C_in, (m + 0) * C_in_stride_m));
        svld1_hor_za32(
            /*tile=*/0, /*slice=*/m + 1, p1,
            offset_bytes(C_in, (m + 1) * C_in_stride_m));
        svld1_hor_za32(
            /*tile=*/0, /*slice=*/m + 2, p2,
            offset_bytes(C_in, (m + 2) * C_in_stride_m));
        svld1_hor_za32(
            /*tile=*/0, /*slice=*/m + 3, p3,
            offset_bytes(C_in, (m + 3) * C_in_stride_m));
      }
    } else {
      svzero_za();
    }
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        ptrdiff_t k1 = K1;
        while (k1 > 0) {
          auto a = svld1(m_mask_ab, reinterpret_cast<const TAB*>(A_k1));
          auto b = svld1(n_mask_ab, reinterpret_cast<const TAB*>(B_k1));
          svmopa</*tile=*/0>(m_mask_ab, n_mask_ab, a, b);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    // Store the accumulated result back to the output.
    for (size_t m = 0; m < M; m += 4) {
      svbool_t p0 = svpsel_lane_b32(n_mask, m_mask, m + 0);
      svbool_t p1 = svpsel_lane_b32(n_mask, m_mask, m + 1);
      svbool_t p2 = svpsel_lane_b32(n_mask, m_mask, m + 2);
      svbool_t p3 = svpsel_lane_b32(n_mask, m_mask, m + 3);
      svst1_hor_za32(
          /*tile=*/0, /*slice=*/m + 0, p0,
          offset_bytes(C_out, (m + 0) * C_out_stride_m));
      svst1_hor_za32(
          /*tile=*/0, /*slice=*/m + 1, p1,
          offset_bytes(C_out, (m + 1) * C_out_stride_m));
      svst1_hor_za32(
          /*tile=*/0, /*slice=*/m + 2, p2,
          offset_bytes(C_out, (m + 2) * C_out_stride_m));
      svst1_hor_za32(
          /*tile=*/0, /*slice=*/m + 3, p3,
          offset_bytes(C_out, (m + 3) * C_out_stride_m));
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(TC)) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(TC));
    B = offset_bytes(B, svl * sizeof(TC));
    n -= svl;
  }
}

}  // namespace

void dot_fp32_sme(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                  size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
                  const void* A, size_t B_stride_k3, size_t B_stride_k2,
                  size_t B_stride_k1, const void* B, size_t C_in_stride_m,
                  const void* C_in, size_t C_out_stride_m, void* C_out) {
  dot_impl<float, float>(M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2,
                         A, B_stride_k3, B_stride_k2, B_stride_k1, B,
                         C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_bf16_bf16_fp32_sme(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                            size_t A_stride_m, size_t A_stride_k3,
                            size_t A_stride_k2, const void* A,
                            size_t B_stride_k3, size_t B_stride_k2,
                            size_t B_stride_k1, const void* B,
                            size_t C_in_stride_m, const void* C_in,
                            size_t C_out_stride_m, void* C_out) {
  dot_impl<bfloat16_t, float>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_fp16_fp16_fp32_sme(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                            size_t A_stride_m, size_t A_stride_k3,
                            size_t A_stride_k2, const void* A,
                            size_t B_stride_k3, size_t B_stride_k2,
                            size_t B_stride_k1, const void* B,
                            size_t C_in_stride_m, const void* C_in,
                            size_t C_out_stride_m, void* C_out) {
  dot_impl<float16_t, float>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_int8_int8_int32_sme(size_t M, size_t N, size_t K3, size_t K2,
                             size_t K1, size_t A_stride_m, size_t A_stride_k3,
                             size_t A_stride_k2, const void* A,
                             size_t B_stride_k3, size_t B_stride_k2,
                             size_t B_stride_k1, const void* B,
                             size_t C_in_stride_m, const void* C_in,
                             size_t C_out_stride_m, void* C_out) {
  dot_impl<int8_t, int32_t>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

}  // namespace ynn
