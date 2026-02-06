// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_sme.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/dot/arm64_sme_internal.h"
#include "ynnpack/kernels/dot/dot.h"

#ifndef YNN_DISABLE_SME

namespace ynn {

namespace {

template <typename TAB, typename TC>
__arm_new("za") __arm_locally_streaming void sme2_dot_opt(
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

  constexpr size_t dot_factor = sizeof(TC) / sizeof(TAB);

  svbool_t m_mask_ab = svwhilelt(0, M * dot_factor, TAB{});
  svbool_t m_mask = svwhilelt(0, M, TC{});
  svcount_t m_count_ab = svctrue(TAB{});

  ptrdiff_t n = N;
  while (n >= svl * 4) {
    svbool_t n_mask_ab = svptrue(TAB{});
    svbool_t n_mask = svptrue(TC{});
    svcount_t n_count_ab = svctrue(TAB{});

    if (C_in) {
      for (size_t m = 0; m < M; ++m) {
        svbool_t p = svpsel_lane_b32(n_mask, m_mask, m);
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(0, m, p, offset_bytes(C_in_m, 0 * svl * sizeof(TC)));
        svld1_hor_za32(1, m, p, offset_bytes(C_in_m, 1 * svl * sizeof(TC)));
        svld1_hor_za32(2, m, p, offset_bytes(C_in_m, 2 * svl * sizeof(TC)));
        svld1_hor_za32(3, m, p, offset_bytes(C_in_m, 3 * svl * sizeof(TC)));
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
        
        while (k1 >= 4 * dot_factor) {
            auto a_ptr = reinterpret_cast<const TAB*>(A_k1);
            auto a_x4 = svld1_x4(m_count_ab, a_ptr);
            
            auto b_ptr = reinterpret_cast<const TAB*>(B_k1);
            auto b0 = svld1_x4(n_count_ab, b_ptr);
            b_ptr = offset_bytes(b_ptr, B_stride_k1 * dot_factor);
            auto b1 = svld1_x4(n_count_ab, b_ptr);
            b_ptr = offset_bytes(b_ptr, B_stride_k1 * dot_factor);
            auto b2 = svld1_x4(n_count_ab, b_ptr);
            b_ptr = offset_bytes(b_ptr, B_stride_k1 * dot_factor);
            auto b3 = svld1_x4(n_count_ab, b_ptr);
            B_k1 = b_ptr;

            // K
            svmopa<0>(m_mask_ab, n_mask_ab, svget4(a_x4, 0), svget4(b0, 0));
            svmopa<1>(m_mask_ab, n_mask_ab, svget4(a_x4, 0), svget4(b0, 1));
            svmopa<2>(m_mask_ab, n_mask_ab, svget4(a_x4, 0), svget4(b0, 2));
            svmopa<3>(m_mask_ab, n_mask_ab, svget4(a_x4, 0), svget4(b0, 3));

            // K+1
            svmopa<0>(m_mask_ab, n_mask_ab, svget4(a_x4, 1), svget4(b1, 0));
            svmopa<1>(m_mask_ab, n_mask_ab, svget4(a_x4, 1), svget4(b1, 1));
            svmopa<2>(m_mask_ab, n_mask_ab, svget4(a_x4, 1), svget4(b1, 2));
            svmopa<3>(m_mask_ab, n_mask_ab, svget4(a_x4, 1), svget4(b1, 3));

            // K+2
            svmopa<0>(m_mask_ab, n_mask_ab, svget4(a_x4, 2), svget4(b2, 0));
            svmopa<1>(m_mask_ab, n_mask_ab, svget4(a_x4, 2), svget4(b2, 1));
            svmopa<2>(m_mask_ab, n_mask_ab, svget4(a_x4, 2), svget4(b2, 2));
            svmopa<3>(m_mask_ab, n_mask_ab, svget4(a_x4, 2), svget4(b2, 3));

            // K+3
            svmopa<0>(m_mask_ab, n_mask_ab, svget4(a_x4, 3), svget4(b3, 0));
            svmopa<1>(m_mask_ab, n_mask_ab, svget4(a_x4, 3), svget4(b3, 1));
            svmopa<2>(m_mask_ab, n_mask_ab, svget4(a_x4, 3), svget4(b3, 2));
            svmopa<3>(m_mask_ab, n_mask_ab, svget4(a_x4, 3), svget4(b3, 3));

            k1 -= 4 * dot_factor;
            A_k1 = offset_bytes(A_k1, 4 * A_stride_m);
        }

        while (k1 > 0) {
          auto a = svld1(m_mask_ab, reinterpret_cast<const TAB*>(A_k1));
          auto b = svld1_x4(n_count_ab, reinterpret_cast<const TAB*>(B_k1));
          svmopa<0>(m_mask_ab, n_mask_ab, a, svget4(b, 0));
          svmopa<1>(m_mask_ab, n_mask_ab, a, svget4(b, 1));
          svmopa<2>(m_mask_ab, n_mask_ab, a, svget4(b, 2));
          svmopa<3>(m_mask_ab, n_mask_ab, a, svget4(b, 3));

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

    for (size_t m = 0; m < M; ++m) {
      svbool_t p = svpsel_lane_b32(n_mask, m_mask, m);
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, p, offset_bytes(C_out_m, 0 * svl * sizeof(TC)));
      svst1_hor_za32(1, m, p, offset_bytes(C_out_m, 1 * svl * sizeof(TC)));
      svst1_hor_za32(2, m, p, offset_bytes(C_out_m, 2 * svl * sizeof(TC)));
      svst1_hor_za32(3, m, p, offset_bytes(C_out_m, 3 * svl * sizeof(TC)));
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(TC) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(TC) * 4);
    B = offset_bytes(B, svl * sizeof(TC) * 4);
    n -= svl * 4;
  }
  
  while (n > 0) {
    svbool_t n_mask_ab = svwhilelt(0, n * dot_factor, TAB{});
    svbool_t n_mask = svwhilelt(0, n, TC{});

    if (C_in) {
      for (size_t m = 0; m < M; ++m) {
        svbool_t p = svpsel_lane_b32(n_mask, m_mask, m);
        svld1_hor_za32(0, m, p, offset_bytes(C_in, m * C_in_stride_m));
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
          svmopa<0>(m_mask_ab, n_mask_ab, a, b);

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

    for (size_t m = 0; m < M; ++m) {
      svbool_t p = svpsel_lane_b32(n_mask, m_mask, m);
      svst1_hor_za32(0, m, p, offset_bytes(C_out, m * C_out_stride_m));
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(TC)) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(TC));
    B = offset_bytes(B, svl * sizeof(TC));
    n -= svl;
  }
}

}  // namespace

void dot_fp32_sme2_opt(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                   size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
                   const void* A, size_t B_stride_k3, size_t B_stride_k2,
                   size_t B_stride_k1, const void* B, size_t C_in_stride_m,
                   const void* C_in, size_t C_out_stride_m, void* C_out) {
  sme2_dot_opt<float, float>(M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2,
                         A, B_stride_k3, B_stride_k2, B_stride_k1, B,
                         C_in_stride_m, C_in, C_out_stride_m, C_out);
}

}  // namespace ynn

#endif  // YNN_DISABLE_SME