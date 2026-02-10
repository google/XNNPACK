// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_sve.h>
#include <arm_sme.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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

  A = __builtin_assume_aligned(A, 64);
  B = __builtin_assume_aligned(B, 64);
  if (C_in) C_in = __builtin_assume_aligned(C_in, 64);
  C_out = __builtin_assume_aligned(C_out, 64);

  const size_t svl_tc = svcnt(TC{});
  assert(M <= svl_tc);

  constexpr size_t dot_factor = sizeof(TC) / sizeof(TAB);
  
  // Row masks.
  svbool_t m_mask = svwhilelt(0, (uint32_t)(M * dot_factor), TAB{});
  svbool_t m_mask_c = svwhilelt(0, (uint32_t)M, TC{});

  // Detection of optimal packing for A (Column-major panels)
  const bool a_is_packed = (A_stride_m == svl_tc * sizeof(TC));

  ptrdiff_t n_total = (ptrdiff_t)N;

  // Main loop: Process 4 * svl columns at a time.
  while (n_total >= (ptrdiff_t)(svl_tc * 4)) {
    svbool_t n_mask_all = svptrue(TAB{});
    svbool_t n_mask_c_all = svptrue(TC{});
    svcount_t n_count_all = svctrue(TAB{});

    if (C_in) {
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svbool_t p = svpsel_lane_b32(n_mask_c_all, m_mask_c, m);
        svld1_hor_za32(0, m, p, offset_bytes(C_in_m, 0 * svl_tc * sizeof(TC)));
        svld1_hor_za32(1, m, p, offset_bytes(C_in_m, 1 * svl_tc * sizeof(TC)));
        svld1_hor_za32(2, m, p, offset_bytes(C_in_m, 2 * svl_tc * sizeof(TC)));
        svld1_hor_za32(3, m, p, offset_bytes(C_in_m, 3 * svl_tc * sizeof(TC)));
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
        ptrdiff_t k1 = (ptrdiff_t)K1;

        if (a_is_packed) {
            // Unrolled K-loop (4x unroll) for packed A with better interleaving
            if (k1 >= (ptrdiff_t)(4 * dot_factor)) {
                // Initial preload for pipelining
                auto b0 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_k1));
                auto b1 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(offset_bytes(B_k1, B_stride_k1 * dot_factor)));
                auto b2 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(offset_bytes(B_k1, B_stride_k1 * dot_factor * 2)));
                auto b3 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(offset_bytes(B_k1, B_stride_k1 * dot_factor * 3)));
                
                while (k1 >= (ptrdiff_t)(4 * dot_factor)) {
                    // Load 4 slices of A (Current iteration)
                    auto a = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(A_k1));
                    
                    // Update pointers for NEXT iteration
                    const void* B_next = offset_bytes(B_k1, B_stride_k1 * dot_factor * 4);
                    k1 -= 4 * dot_factor;
                    A_k1 = offset_bytes(A_k1, 4 * A_stride_m);
                    B_k1 = B_next;

                    // Interleaved MOPAs (Current iteration data)
                    svmopa<0>(m_mask, n_mask_all, svget4(a, 0), svget4(b0, 0));
                    svmopa<1>(m_mask, n_mask_all, svget4(a, 1), svget4(b1, 0));
                    svmopa<2>(m_mask, n_mask_all, svget4(a, 2), svget4(b2, 0));
                    svmopa<3>(m_mask, n_mask_all, svget4(a, 3), svget4(b3, 0));

                    if (k1 >= (ptrdiff_t)(4 * dot_factor)) {
                        // Start loading NEXT iteration's B while computing current MOPAs
                        b0 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_next));
                        b1 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(offset_bytes(B_next, B_stride_k1 * dot_factor)));
                    }

                    svmopa<0>(m_mask, n_mask_all, svget4(a, 0), svget4(b0, 1));
                    svmopa<1>(m_mask, n_mask_all, svget4(a, 1), svget4(b1, 1));
                    svmopa<2>(m_mask, n_mask_all, svget4(a, 2), svget4(b2, 1));
                    svmopa<3>(m_mask, n_mask_all, svget4(a, 3), svget4(b3, 1));

                    if (k1 >= (ptrdiff_t)(4 * dot_factor)) {
                        b2 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(offset_bytes(B_next, B_stride_k1 * dot_factor * 2)));
                        b3 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(offset_bytes(B_next, B_stride_k1 * dot_factor * 3)));
                    }

                    svmopa<0>(m_mask, n_mask_all, svget4(a, 0), svget4(b0, 2));
                    svmopa<1>(m_mask, n_mask_all, svget4(a, 1), svget4(b1, 2));
                    svmopa<2>(m_mask, n_mask_all, svget4(a, 2), svget4(b2, 2));
                    svmopa<3>(m_mask, n_mask_all, svget4(a, 3), svget4(b3, 2));

                    svmopa<0>(m_mask, n_mask_all, svget4(a, 0), svget4(b0, 3));
                    svmopa<1>(m_mask, n_mask_all, svget4(a, 1), svget4(b1, 3));
                    svmopa<2>(m_mask, n_mask_all, svget4(a, 2), svget4(b2, 3));
                    svmopa<3>(m_mask, n_mask_all, svget4(a, 3), svget4(b3, 3));
                }
            }
        } else {
            // Unrolled K-loop (4x unroll) for non-packed A
            while (k1 >= (ptrdiff_t)(4 * dot_factor)) {
                // Pre-calc pointers
                const void* A_k1_0 = A_k1;
                const void* A_k1_1 = offset_bytes(A_k1, A_stride_m);
                const void* A_k1_2 = offset_bytes(A_k1_1, A_stride_m);
                const void* A_k1_3 = offset_bytes(A_k1_2, A_stride_m);

                const void* B_k1_0 = B_k1;
                const void* B_k1_1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
                const void* B_k1_2 = offset_bytes(B_k1_1, B_stride_k1 * dot_factor);
                const void* B_k1_3 = offset_bytes(B_k1_2, B_stride_k1 * dot_factor);

                auto a0 = svld1(m_mask, reinterpret_cast<const TAB*>(A_k1_0));
                auto b0 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_k1_0));
                svmopa<0>(m_mask, n_mask_all, a0, svget4(b0, 0));
                svmopa<1>(m_mask, n_mask_all, a0, svget4(b0, 1));
                svmopa<2>(m_mask, n_mask_all, a0, svget4(b0, 2));
                svmopa<3>(m_mask, n_mask_all, a0, svget4(b0, 3));

                auto a1 = svld1(m_mask, reinterpret_cast<const TAB*>(A_k1_1));
                auto b1 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_k1_1));
                svmopa<0>(m_mask, n_mask_all, a1, svget4(b1, 0));
                svmopa<1>(m_mask, n_mask_all, a1, svget4(b1, 1));
                svmopa<2>(m_mask, n_mask_all, a1, svget4(b1, 2));
                svmopa<3>(m_mask, n_mask_all, a1, svget4(b1, 3));

                auto a2 = svld1(m_mask, reinterpret_cast<const TAB*>(A_k1_2));
                auto b2 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_k1_2));
                svmopa<0>(m_mask, n_mask_all, a2, svget4(b2, 0));
                svmopa<1>(m_mask, n_mask_all, a2, svget4(b2, 1));
                svmopa<2>(m_mask, n_mask_all, a2, svget4(b2, 2));
                svmopa<3>(m_mask, n_mask_all, a2, svget4(b2, 3));

                auto a3 = svld1(m_mask, reinterpret_cast<const TAB*>(A_k1_3));
                auto b3 = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_k1_3));
                svmopa<0>(m_mask, n_mask_all, a3, svget4(b3, 0));
                svmopa<1>(m_mask, n_mask_all, a3, svget4(b3, 1));
                svmopa<2>(m_mask, n_mask_all, a3, svget4(b3, 2));
                svmopa<3>(m_mask, n_mask_all, a3, svget4(b3, 3));

                k1 -= 4 * dot_factor;
                B_k1 = offset_bytes(B_k1_3, B_stride_k1 * dot_factor);
                A_k1 = offset_bytes(A_k1_3, A_stride_m);
            }
        }

        // Remainder K-loop
        while (k1 > 0) {
            auto a = svld1(m_mask, reinterpret_cast<const TAB*>(A_k1));
            auto b = svld1_x4_impl(n_count_all, reinterpret_cast<const TAB*>(B_k1));
            
            svmopa<0>(m_mask, n_mask_all, a, svget4(b, 0));
            svmopa<1>(m_mask, n_mask_all, a, svget4(b, 1));
            svmopa<2>(m_mask, n_mask_all, a, svget4(b, 2));
            svmopa<3>(m_mask, n_mask_all, a, svget4(b, 3));

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
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, svpsel_lane_b32(n_mask_c_all, m_mask_c, m), offset_bytes(C_out_m, 0 * svl_tc * sizeof(TC)));
      svst1_hor_za32(1, m, svpsel_lane_b32(n_mask_c_all, m_mask_c, m), offset_bytes(C_out_m, 1 * svl_tc * sizeof(TC)));
      svst1_hor_za32(2, m, svpsel_lane_b32(n_mask_c_all, m_mask_c, m), offset_bytes(C_out_m, 2 * svl_tc * sizeof(TC)));
      svst1_hor_za32(3, m, svpsel_lane_b32(n_mask_c_all, m_mask_c, m), offset_bytes(C_out_m, 3 * svl_tc * sizeof(TC)));
    }
    
    C_in = C_in ? offset_bytes(C_in, svl_tc * sizeof(TC) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl_tc * sizeof(TC) * 4);
    B = offset_bytes(B, svl_tc * sizeof(TC) * 4);
    n_total -= (ptrdiff_t)(svl_tc * 4);
  }

  // Tail loop: Process remaining columns unrolled up to 4 tiles.
  while (n_total > 0) {
    svbool_t n_mask0 = svwhilelt(0 * svl_tc * dot_factor, (uint32_t)(n_total * dot_factor), TAB{});
    svbool_t n_mask1 = svwhilelt(1 * svl_tc * dot_factor, (uint32_t)(n_total * dot_factor), TAB{});
    svbool_t n_mask2 = svwhilelt(2 * svl_tc * dot_factor, (uint32_t)(n_total * dot_factor), TAB{});
    svbool_t n_mask3 = svwhilelt(3 * svl_tc * dot_factor, (uint32_t)(n_total * dot_factor), TAB{});

    svbool_t n_mask_c0 = svwhilelt(0 * svl_tc, (uint32_t)n_total, TC{});
    svbool_t n_mask_c1 = svwhilelt(1 * svl_tc, (uint32_t)n_total, TC{});
    svbool_t n_mask_c2 = svwhilelt(2 * svl_tc, (uint32_t)n_total, TC{});
    svbool_t n_mask_c3 = svwhilelt(3 * svl_tc, (uint32_t)n_total, TC{});

    const bool do_tile_c1 = svany(n_mask_c1);
    const bool do_tile_c2 = svany(n_mask_c2);
    const bool do_tile_c3 = svany(n_mask_c3);

    if (C_in) {
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(0, m, svpsel_lane_b32(n_mask_c0, m_mask_c, m), offset_bytes(C_in_m, 0 * svl_tc * sizeof(TC)));
        if (do_tile_c1) svld1_hor_za32(1, m, svpsel_lane_b32(n_mask_c1, m_mask_c, m), offset_bytes(C_in_m, 1 * svl_tc * sizeof(TC)));
        if (do_tile_c2) svld1_hor_za32(2, m, svpsel_lane_b32(n_mask_c2, m_mask_c, m), offset_bytes(C_in_m, 2 * svl_tc * sizeof(TC)));
        if (do_tile_c3) svld1_hor_za32(3, m, svpsel_lane_b32(n_mask_c3, m_mask_c, m), offset_bytes(C_in_m, 3 * svl_tc * sizeof(TC)));
      }
    } else {
      svzero_za();
    }
    
    const void* B_k3 = B;
    const void* A_k3 = A;
    size_t k3 = K3;
    const bool do_tile1 = svany(n_mask1);
    const bool do_tile2 = svany(n_mask2);
    const bool do_tile3 = svany(n_mask3);
    do {
      const void* B_k2 = B_k3;
      const void* A_k2 = A_k3;
      size_t k2 = K2;
      do {
        const void* B_k1 = B_k2;
        const void* A_k1 = A_k2;
        ptrdiff_t k1 = (ptrdiff_t)K1;
        while (k1 > 0) {
          auto a = svld1(m_mask, reinterpret_cast<const TAB*>(A_k1));
          
          svmopa<0>(m_mask, n_mask0, a, svld1(n_mask0, reinterpret_cast<const TAB*>(B_k1)));
          if (do_tile1) svmopa<1>(m_mask, n_mask1, a, svld1(n_mask1, reinterpret_cast<const TAB*>(offset_bytes(B_k1, 1 * svl_tc * sizeof(TC)))));
          if (do_tile2) svmopa<2>(m_mask, n_mask2, a, svld1(n_mask2, reinterpret_cast<const TAB*>(offset_bytes(B_k1, 2 * svl_tc * sizeof(TC)))));
          if (do_tile3) svmopa<3>(m_mask, n_mask3, a, svld1(n_mask3, reinterpret_cast<const TAB*>(offset_bytes(B_k1, 3 * svl_tc * sizeof(TC)))));

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
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, svpsel_lane_b32(n_mask_c0, m_mask_c, m), offset_bytes(C_out_m, 0 * svl_tc * sizeof(TC)));
      if (do_tile_c1) svst1_hor_za32(1, m, svpsel_lane_b32(n_mask_c1, m_mask_c, m), offset_bytes(C_out_m, 1 * svl_tc * sizeof(TC)));
      if (do_tile_c2) svst1_hor_za32(2, m, svpsel_lane_b32(n_mask_c2, m_mask_c, m), offset_bytes(C_out_m, 2 * svl_tc * sizeof(TC)));
      if (do_tile_c3) svst1_hor_za32(3, m, svpsel_lane_b32(n_mask_c3, m_mask_c, m), offset_bytes(C_out_m, 3 * svl_tc * sizeof(TC)));
    }
    
    C_in = C_in ? offset_bytes(C_in, svl_tc * sizeof(TC) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl_tc * sizeof(TC) * 4);
    B = offset_bytes(B, svl_tc * sizeof(TC) * 4);
    n_total -= (ptrdiff_t)(svl_tc * 4);
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

void dot_bf16_bf16_fp32_sme2_opt(size_t M, size_t N, size_t K3, size_t K2,
                             size_t K1, size_t A_stride_m, size_t A_stride_k3,
                             size_t A_stride_k2, const void* A,
                             size_t B_stride_k3, size_t B_stride_k2,
                             size_t B_stride_k1, const void* B,
                             size_t C_in_stride_m, const void* C_in,
                             size_t C_out_stride_m, void* C_out) {
  sme2_dot_opt<bfloat16_t, float>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_fp16_fp16_fp32_sme2_opt(size_t M, size_t N, size_t K3, size_t K2,
                             size_t K1, size_t A_stride_m, size_t A_stride_k3,
                             size_t A_stride_k2, const void* A,
                             size_t B_stride_k3, size_t B_stride_k2,
                             size_t B_stride_k1, const void* B,
                             size_t C_in_stride_m, const void* C_in,
                             size_t C_out_stride_m, void* C_out) {
  sme2_dot_opt<float16_t, float>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_int8_int8_int32_sme2_opt(size_t M, size_t N, size_t K3, size_t K2,
                              size_t K1, size_t A_stride_m, size_t A_stride_k3,
                              size_t A_stride_k2, const void* A,
                              size_t B_stride_k3, size_t B_stride_k2,
                              size_t B_stride_k1, const void* B,
                              size_t C_in_stride_m, const void* C_in,
                              size_t C_out_stride_m, void* C_out) {
  sme2_dot_opt<int8_t, int32_t>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

}  // namespace ynn

#endif  // YNN_DISABLE_SME