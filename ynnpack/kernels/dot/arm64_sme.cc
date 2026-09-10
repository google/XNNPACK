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

#ifndef YNN_DISABLE_SME

namespace ynn {

size_t sme_vl(float) {
  if (is_arch_supported(arch_flag::sme) || is_arch_supported(arch_flag::sme2)) {
    return svcnts(float{});
  } else {
    return 0;
  }
}

size_t sme_vl(int32_t) {
  if (is_arch_supported(arch_flag::sme) || is_arch_supported(arch_flag::sme2)) {
    return svcnts(int32_t{});
  } else {
    return 0;
  }
}

namespace {

template <typename TAB, typename TC>
__arm_new("za") __arm_locally_streaming void sme_dot(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  const size_t svl = svcnts(TC{});
  assert(M <= svl);

  // This is how many elements of the k dimension are multiplied and accumulated
  // at once.
  constexpr size_t dot_factor = sizeof(TC) / sizeof(TAB);

  // Masks for the row dimension output.
  svbool_t m_mask_ab = svwhilelt(0, M * dot_factor, TAB{});

  ptrdiff_t n = N;
  while (n >= svl * 4) {
    if (C_in) {
      // Load the output to initialize the tile accumulator.
      // TODO: To improve numerical precision and better match the other
      // kernels, it would be best to initialize this to zero (`svzero_za()`)
      // the tile instead of loading the initial accumulator, and add this
      // later.
      svbool_t n_mask = svptrue(TC{});
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(/*tile=*/0, /*slice=*/m, n_mask,
                       offset_bytes(C_in_m, 0 * svl * sizeof(TC)));
        svld1_hor_za32(/*tile=*/1, /*slice=*/m, n_mask,
                       offset_bytes(C_in_m, 1 * svl * sizeof(TC)));
        svld1_hor_za32(/*tile=*/2, /*slice=*/m, n_mask,
                       offset_bytes(C_in_m, 2 * svl * sizeof(TC)));
        svld1_hor_za32(/*tile=*/3, /*slice=*/m, n_mask,
                       offset_bytes(C_in_m, 3 * svl * sizeof(TC)));
      }
    } else {
      svzero_za();
    }

    // (All-true) masks for the column dimension output.
    svbool_t n_mask_ab = svptrue(TAB{});

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
          A_k1 = offset_bytes(A_k1, A_stride_m * dot_factor);
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
    svbool_t n_mask = svptrue(TC{});
    for (size_t m = 0; m < M; ++m) {
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(/*tile=*/0, /*slice=*/m, n_mask,
                     offset_bytes(C_out_m, 0 * svl * sizeof(TC)));
      svst1_hor_za32(/*tile=*/1, /*slice=*/m, n_mask,
                     offset_bytes(C_out_m, 1 * svl * sizeof(TC)));
      svst1_hor_za32(/*tile=*/2, /*slice=*/m, n_mask,
                     offset_bytes(C_out_m, 2 * svl * sizeof(TC)));
      svst1_hor_za32(/*tile=*/3, /*slice=*/m, n_mask,
                     offset_bytes(C_out_m, 3 * svl * sizeof(TC)));
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(TC) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(TC) * 4);
    B = offset_bytes(B, svl * sizeof(TC) * 4);
    n -= svl * 4;
  }
  if (n > 0) {
    if (C_in) {
      // Load the output to initialize the tile accumulator.
      // TODO: To improve numerical precision and better match the other
      // kernels, it would be best to initialize this to zero (`svzero_za()`)
      // the tile instead of loading the initial accumulator, and add this
      // later.
      svbool_t n_mask0 = svwhilelt(0 * svl, n, TC{});
      svbool_t n_mask1 = svwhilelt(1 * svl, n, TC{});
      svbool_t n_mask2 = svwhilelt(2 * svl, n, TC{});
      svbool_t n_mask3 = svwhilelt(3 * svl, n, TC{});
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(/*tile=*/0, /*slice=*/m, n_mask0,
                       offset_bytes(C_in_m, 0 * svl * sizeof(TC)));
        svld1_hor_za32(/*tile=*/1, /*slice=*/m, n_mask1,
                       offset_bytes(C_in_m, 1 * svl * sizeof(TC)));
        svld1_hor_za32(/*tile=*/2, /*slice=*/m, n_mask2,
                       offset_bytes(C_in_m, 2 * svl * sizeof(TC)));
        svld1_hor_za32(/*tile=*/3, /*slice=*/m, n_mask3,
                       offset_bytes(C_in_m, 3 * svl * sizeof(TC)));
      }
    } else {
      svzero_za();
    }

    // Masks for the column dimension output.
    svbool_t n_mask_ab0 =
        svwhilelt(0 * svl * dot_factor, n * dot_factor, TAB{});
    svbool_t n_mask_ab1 =
        svwhilelt(1 * svl * dot_factor, n * dot_factor, TAB{});
    svbool_t n_mask_ab2 =
        svwhilelt(2 * svl * dot_factor, n * dot_factor, TAB{});
    svbool_t n_mask_ab3 =
        svwhilelt(3 * svl * dot_factor, n * dot_factor, TAB{});

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
              svld1_vnum(n_mask_ab0, reinterpret_cast<const TAB*>(B_k1), 0);
          auto b_1 =
              svld1_vnum(n_mask_ab1, reinterpret_cast<const TAB*>(B_k1), 1);
          auto b_2 =
              svld1_vnum(n_mask_ab2, reinterpret_cast<const TAB*>(B_k1), 2);
          auto b_3 =
              svld1_vnum(n_mask_ab3, reinterpret_cast<const TAB*>(B_k1), 3);
          svmopa</*tile=*/0>(m_mask_ab, n_mask_ab0, a, b_0);
          svmopa</*tile=*/1>(m_mask_ab, n_mask_ab1, a, b_1);
          svmopa</*tile=*/2>(m_mask_ab, n_mask_ab2, a, b_2);
          svmopa</*tile=*/3>(m_mask_ab, n_mask_ab3, a, b_3);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m * dot_factor);
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
    svbool_t n_mask0 = svwhilelt(0 * svl, n, TC{});
    svbool_t n_mask1 = svwhilelt(1 * svl, n, TC{});
    svbool_t n_mask2 = svwhilelt(2 * svl, n, TC{});
    svbool_t n_mask3 = svwhilelt(3 * svl, n, TC{});
    for (size_t m = 0; m < M; ++m) {
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(/*tile=*/0, /*slice=*/m, n_mask0,
                     offset_bytes(C_out_m, 0 * svl * sizeof(TC)));
      svst1_hor_za32(/*tile=*/1, /*slice=*/m, n_mask1,
                     offset_bytes(C_out_m, 1 * svl * sizeof(TC)));
      svst1_hor_za32(/*tile=*/2, /*slice=*/m, n_mask2,
                     offset_bytes(C_out_m, 2 * svl * sizeof(TC)));
      svst1_hor_za32(/*tile=*/3, /*slice=*/m, n_mask3,
                     offset_bytes(C_out_m, 3 * svl * sizeof(TC)));
    }
  }
}

YNN_ALWAYS_INLINE void unpack_int4(svint8_t b_raw, svint8_t& zm_0,
                                   svint8_t& zm_1) YNN_SME_HELPER {
  const svbool_t ptrue = svptrue(int8_t{});
  auto lo = svasr_n_s8_x(ptrue, svlsl_n_s8_x(ptrue, b_raw, 4), 4);
  auto hi = svasr_n_s8_x(ptrue, b_raw, 4);
  auto z0 = svzip1_s8(lo, hi);
  auto z1 = svzip2_s8(lo, hi);
  zm_0 = svreinterpret_s8_s32(
      svuzp1_s32(svreinterpret_s32_s8(z0), svreinterpret_s32_s8(z1)));
  zm_1 = svreinterpret_s8_s32(
      svuzp2_s32(svreinterpret_s32_s8(z0), svreinterpret_s32_s8(z1)));
}

YNN_ALWAYS_INLINE void mopa4_int4(svbool_t m_mask, svbool_t n0, svbool_t n1,
                                  svbool_t n2, svbool_t n3, svint8_t a_0,
                                  svint8_t a_1, svint8_t b0, svint8_t b1,
                                  svint8_t b2, svint8_t b3) YNN_SME_HELPER {
  svint8_t b0_0, b0_1;
  unpack_int4(b0, b0_0, b0_1);
  svmopa<0>(m_mask, n0, a_0, b0_0);

  svint8_t b1_0, b1_1;
  unpack_int4(b1, b1_0, b1_1);
  svmopa<1>(m_mask, n1, a_0, b1_0);
  svmopa<0>(m_mask, n0, a_1, b0_1);

  svint8_t b2_0, b2_1;
  unpack_int4(b2, b2_0, b2_1);
  svmopa<2>(m_mask, n2, a_0, b2_0);
  svmopa<1>(m_mask, n1, a_1, b1_1);

  svint8_t b3_0, b3_1;
  unpack_int4(b3, b3_0, b3_1);
  svmopa<3>(m_mask, n3, a_0, b3_0);
  svmopa<2>(m_mask, n2, a_1, b2_1);

  svmopa<3>(m_mask, n3, a_1, b3_1);
}

YNN_ALWAYS_INLINE void mopa4_int4(svbool_t m_mask, svbool_t n, svint8_t a_0,
                                  svint8_t a_1, svint8_t b0, svint8_t b1,
                                  svint8_t b2, svint8_t b3) YNN_SME_HELPER {
  mopa4_int4(m_mask, n, n, n, n, a_0, a_1, b0, b1, b2, b3);
}

__arm_new("za") __arm_locally_streaming void sme_dot_int8_int4_int32(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  const size_t svl = svcnts(int32_t{});
  assert(M <= svl);

  constexpr size_t dot_factor = 8;
  const svbool_t m_mask_s32 = svwhilelt(0, M, int32_t{});
  const svbool_t m_mask_ab =
      svwhilelt(static_cast<int64_t>(0), static_cast<int64_t>(M * 4), int8_t{});
  const svbool_t n_mask_ab = svptrue(int8_t{});

  ptrdiff_t n = N;
  while (n >= static_cast<ptrdiff_t>(svl * 4)) {
    if (C_in) {
      svbool_t n_mask = svptrue(int32_t{});
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(0, m, n_mask,
                       offset_bytes(C_in_m, 0 * svl * sizeof(int32_t)));
        svld1_hor_za32(1, m, n_mask,
                       offset_bytes(C_in_m, 1 * svl * sizeof(int32_t)));
        svld1_hor_za32(2, m, n_mask,
                       offset_bytes(C_in_m, 2 * svl * sizeof(int32_t)));
        svld1_hor_za32(3, m, n_mask,
                       offset_bytes(C_in_m, 3 * svl * sizeof(int32_t)));
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
          auto a_w =
              svld2_s32(m_mask_s32, reinterpret_cast<const int32_t*>(A_k1));
          auto a_0 = svreinterpret_s8_s32(svget2(a_w, 0));
          auto a_1 = svreinterpret_s8_s32(svget2(a_w, 1));

          auto b_raw_0 = svld1_s8(
              n_mask_ab, reinterpret_cast<const int8_t*>(B_k1) + 0 * svl * 4);
          auto b_raw_1 = svld1_s8(
              n_mask_ab, reinterpret_cast<const int8_t*>(B_k1) + 1 * svl * 4);
          auto b_raw_2 = svld1_s8(
              n_mask_ab, reinterpret_cast<const int8_t*>(B_k1) + 2 * svl * 4);
          auto b_raw_3 = svld1_s8(
              n_mask_ab, reinterpret_cast<const int8_t*>(B_k1) + 3 * svl * 4);

          mopa4_int4(m_mask_ab, n_mask_ab, a_0, a_1, b_raw_0, b_raw_1, b_raw_2,
                     b_raw_3);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m * dot_factor);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    svbool_t n_mask = svptrue(int32_t{});
    for (size_t m = 0; m < M; ++m) {
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, n_mask,
                     offset_bytes(C_out_m, 0 * svl * sizeof(int32_t)));
      svst1_hor_za32(1, m, n_mask,
                     offset_bytes(C_out_m, 1 * svl * sizeof(int32_t)));
      svst1_hor_za32(2, m, n_mask,
                     offset_bytes(C_out_m, 2 * svl * sizeof(int32_t)));
      svst1_hor_za32(3, m, n_mask,
                     offset_bytes(C_out_m, 3 * svl * sizeof(int32_t)));
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(int32_t) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(int32_t) * 4);
    B = offset_bytes(B, svl * 16);
    n -= svl * 4;
  }
  if (n > 0) {
    if (C_in) {
      svbool_t n_mask0 = svwhilelt(0 * svl, n, int32_t{});
      svbool_t n_mask1 = svwhilelt(1 * svl, n, int32_t{});
      svbool_t n_mask2 = svwhilelt(2 * svl, n, int32_t{});
      svbool_t n_mask3 = svwhilelt(3 * svl, n, int32_t{});
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(0, m, n_mask0,
                       offset_bytes(C_in_m, 0 * svl * sizeof(int32_t)));
        svld1_hor_za32(1, m, n_mask1,
                       offset_bytes(C_in_m, 1 * svl * sizeof(int32_t)));
        svld1_hor_za32(2, m, n_mask2,
                       offset_bytes(C_in_m, 2 * svl * sizeof(int32_t)));
        svld1_hor_za32(3, m, n_mask3,
                       offset_bytes(C_in_m, 3 * svl * sizeof(int32_t)));
      }
    } else {
      svzero_za();
    }

    svbool_t n_mask_ab0 = svwhilelt(static_cast<int64_t>(0 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_ab1 = svwhilelt(static_cast<int64_t>(1 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_ab2 = svwhilelt(static_cast<int64_t>(2 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_ab3 = svwhilelt(static_cast<int64_t>(3 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});

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
          auto a_w =
              svld2_s32(m_mask_s32, reinterpret_cast<const int32_t*>(A_k1));
          auto a_0 = svreinterpret_s8_s32(svget2(a_w, 0));
          auto a_1 = svreinterpret_s8_s32(svget2(a_w, 1));

          auto b_raw_0 = svld1_s8(
              n_mask_ab0, reinterpret_cast<const int8_t*>(B_k1) + 0 * svl * 4);
          auto b_raw_1 = svld1_s8(
              n_mask_ab1, reinterpret_cast<const int8_t*>(B_k1) + 1 * svl * 4);
          auto b_raw_2 = svld1_s8(
              n_mask_ab2, reinterpret_cast<const int8_t*>(B_k1) + 2 * svl * 4);
          auto b_raw_3 = svld1_s8(
              n_mask_ab3, reinterpret_cast<const int8_t*>(B_k1) + 3 * svl * 4);

          mopa4_int4(m_mask_ab, n_mask_ab0, n_mask_ab1, n_mask_ab2, n_mask_ab3,
                     a_0, a_1, b_raw_0, b_raw_1, b_raw_2, b_raw_3);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m * dot_factor);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    svbool_t n_mask0 = svwhilelt(0 * svl, n, int32_t{});
    svbool_t n_mask1 = svwhilelt(1 * svl, n, int32_t{});
    svbool_t n_mask2 = svwhilelt(2 * svl, n, int32_t{});
    svbool_t n_mask3 = svwhilelt(3 * svl, n, int32_t{});
    for (size_t m = 0; m < M; ++m) {
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, n_mask0,
                     offset_bytes(C_out_m, 0 * svl * sizeof(int32_t)));
      svst1_hor_za32(1, m, n_mask1,
                     offset_bytes(C_out_m, 1 * svl * sizeof(int32_t)));
      svst1_hor_za32(2, m, n_mask2,
                     offset_bytes(C_out_m, 2 * svl * sizeof(int32_t)));
      svst1_hor_za32(3, m, n_mask3,
                     offset_bytes(C_out_m, 3 * svl * sizeof(int32_t)));
    }
  }
}

YNN_ALWAYS_INLINE svint8_t unpack_int2(svint8_t b_val, svuint8_t tbl_idx,
                                       svuint8_t shift_vec) YNN_SME_HELPER {
  const svbool_t ptrue = svptrue(int8_t{});
  return svasr_n_s8_x(
      ptrue, svlsl_s8_x(ptrue, svtbl_s8(b_val, tbl_idx), shift_vec), 6);
}

YNN_ALWAYS_INLINE void mopa4_int2(svbool_t m_mask, svbool_t n0, svbool_t n1,
                                  svbool_t n2, svbool_t n3, svuint8_t shift_vec,
                                  svuint8_t tbl0, svuint8_t tbl1,
                                  svuint8_t tbl2, svuint8_t tbl3, svint8_t a,
                                  svint8_t b) YNN_SME_HELPER {
  svmopa<0>(m_mask, n0, a, unpack_int2(b, tbl0, shift_vec));
  svmopa<1>(m_mask, n1, a, unpack_int2(b, tbl1, shift_vec));
  svmopa<2>(m_mask, n2, a, unpack_int2(b, tbl2, shift_vec));
  svmopa<3>(m_mask, n3, a, unpack_int2(b, tbl3, shift_vec));
}

YNN_ALWAYS_INLINE void mopa4_int2(svbool_t m_mask, svbool_t n,
                                  svuint8_t shift_vec, svuint8_t tbl0,
                                  svuint8_t tbl1, svuint8_t tbl2,
                                  svuint8_t tbl3, svint8_t a,
                                  svint8_t b) YNN_SME_HELPER {
  mopa4_int2(m_mask, n, n, n, n, shift_vec, tbl0, tbl1, tbl2, tbl3, a, b);
}

__arm_new("za") __arm_locally_streaming void sme_dot_int8_int2_int32(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out) {
  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  const size_t svl = svcnts(int32_t{});
  assert(M <= svl);

  constexpr size_t dot_factor = 16;
  const svbool_t m_mask_s32 = svwhilelt(0, M, int32_t{});
  const svbool_t m_mask_ab =
      svwhilelt(static_cast<int64_t>(0), static_cast<int64_t>(M * 4), int8_t{});
  const svbool_t n_mask_ab = svptrue(int8_t{});
  const svuint8_t shift_vec = svreinterpret_u8_u32(svdup_n_u32(0x00020406));
  const svuint8_t tbl0 =
      svreinterpret_u8_u32(svindex_u32(0 * svl * 0x01010101, 0x01010101));
  const svuint8_t tbl1 =
      svreinterpret_u8_u32(svindex_u32(1 * svl * 0x01010101, 0x01010101));
  const svuint8_t tbl2 =
      svreinterpret_u8_u32(svindex_u32(2 * svl * 0x01010101, 0x01010101));
  const svuint8_t tbl3 =
      svreinterpret_u8_u32(svindex_u32(3 * svl * 0x01010101, 0x01010101));

  ptrdiff_t n = N;
  while (n >= static_cast<ptrdiff_t>(svl * 4)) {
    if (C_in) {
      svbool_t n_mask = svptrue(int32_t{});
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(0, m, n_mask,
                       offset_bytes(C_in_m, 0 * svl * sizeof(int32_t)));
        svld1_hor_za32(1, m, n_mask,
                       offset_bytes(C_in_m, 1 * svl * sizeof(int32_t)));
        svld1_hor_za32(2, m, n_mask,
                       offset_bytes(C_in_m, 2 * svl * sizeof(int32_t)));
        svld1_hor_za32(3, m, n_mask,
                       offset_bytes(C_in_m, 3 * svl * sizeof(int32_t)));
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
          auto a_w =
              svld4_s32(m_mask_s32, reinterpret_cast<const int32_t*>(A_k1));
          auto a_0 = svreinterpret_s8_s32(svget4(a_w, 0));
          auto a_1 = svreinterpret_s8_s32(svget4(a_w, 1));
          auto a_2 = svreinterpret_s8_s32(svget4(a_w, 2));
          auto a_3 = svreinterpret_s8_s32(svget4(a_w, 3));

          auto b_bytes =
              svld4_s8(n_mask_ab, reinterpret_cast<const int8_t*>(B_k1));
          auto b_0 = svget4(b_bytes, 0);
          auto b_1 = svget4(b_bytes, 1);
          auto b_2 = svget4(b_bytes, 2);
          auto b_3 = svget4(b_bytes, 3);

          mopa4_int2(m_mask_ab, n_mask_ab, shift_vec, tbl0, tbl1, tbl2, tbl3,
                     a_0, b_0);
          mopa4_int2(m_mask_ab, n_mask_ab, shift_vec, tbl0, tbl1, tbl2, tbl3,
                     a_1, b_1);
          mopa4_int2(m_mask_ab, n_mask_ab, shift_vec, tbl0, tbl1, tbl2, tbl3,
                     a_2, b_2);
          mopa4_int2(m_mask_ab, n_mask_ab, shift_vec, tbl0, tbl1, tbl2, tbl3,
                     a_3, b_3);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m * dot_factor);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    svbool_t n_mask = svptrue(int32_t{});
    for (size_t m = 0; m < M; ++m) {
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, n_mask,
                     offset_bytes(C_out_m, 0 * svl * sizeof(int32_t)));
      svst1_hor_za32(1, m, n_mask,
                     offset_bytes(C_out_m, 1 * svl * sizeof(int32_t)));
      svst1_hor_za32(2, m, n_mask,
                     offset_bytes(C_out_m, 2 * svl * sizeof(int32_t)));
      svst1_hor_za32(3, m, n_mask,
                     offset_bytes(C_out_m, 3 * svl * sizeof(int32_t)));
    }
    C_in = C_in ? offset_bytes(C_in, svl * sizeof(int32_t) * 4) : nullptr;
    C_out = offset_bytes(C_out, svl * sizeof(int32_t) * 4);
    B = offset_bytes(B, svl * 16);
    n -= svl * 4;
  }
  if (n > 0) {
    if (C_in) {
      svbool_t n_mask0 = svwhilelt(0 * svl, n, int32_t{});
      svbool_t n_mask1 = svwhilelt(1 * svl, n, int32_t{});
      svbool_t n_mask2 = svwhilelt(2 * svl, n, int32_t{});
      svbool_t n_mask3 = svwhilelt(3 * svl, n, int32_t{});
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        svld1_hor_za32(0, m, n_mask0,
                       offset_bytes(C_in_m, 0 * svl * sizeof(int32_t)));
        svld1_hor_za32(1, m, n_mask1,
                       offset_bytes(C_in_m, 1 * svl * sizeof(int32_t)));
        svld1_hor_za32(2, m, n_mask2,
                       offset_bytes(C_in_m, 2 * svl * sizeof(int32_t)));
        svld1_hor_za32(3, m, n_mask3,
                       offset_bytes(C_in_m, 3 * svl * sizeof(int32_t)));
      }
    } else {
      svzero_za();
    }

    svbool_t n_mask_ab0 = svwhilelt(static_cast<int64_t>(0 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_ab1 = svwhilelt(static_cast<int64_t>(1 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_ab2 = svwhilelt(static_cast<int64_t>(2 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_ab3 = svwhilelt(static_cast<int64_t>(3 * svl * 4),
                                    static_cast<int64_t>(n * 4), int8_t{});
    svbool_t n_mask_b = svwhilelt(0, n, int8_t{});

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
          auto a_w =
              svld4_s32(m_mask_s32, reinterpret_cast<const int32_t*>(A_k1));
          auto a_0 = svreinterpret_s8_s32(svget4(a_w, 0));
          auto a_1 = svreinterpret_s8_s32(svget4(a_w, 1));
          auto a_2 = svreinterpret_s8_s32(svget4(a_w, 2));
          auto a_3 = svreinterpret_s8_s32(svget4(a_w, 3));

          auto b_bytes =
              svld4_s8(n_mask_b, reinterpret_cast<const int8_t*>(B_k1));
          auto b_0 = svget4(b_bytes, 0);
          auto b_1 = svget4(b_bytes, 1);
          auto b_2 = svget4(b_bytes, 2);
          auto b_3 = svget4(b_bytes, 3);

          mopa4_int2(m_mask_ab, n_mask_ab0, n_mask_ab1, n_mask_ab2, n_mask_ab3,
                     shift_vec, tbl0, tbl1, tbl2, tbl3, a_0, b_0);
          mopa4_int2(m_mask_ab, n_mask_ab0, n_mask_ab1, n_mask_ab2, n_mask_ab3,
                     shift_vec, tbl0, tbl1, tbl2, tbl3, a_1, b_1);
          mopa4_int2(m_mask_ab, n_mask_ab0, n_mask_ab1, n_mask_ab2, n_mask_ab3,
                     shift_vec, tbl0, tbl1, tbl2, tbl3, a_2, b_2);
          mopa4_int2(m_mask_ab, n_mask_ab0, n_mask_ab1, n_mask_ab2, n_mask_ab3,
                     shift_vec, tbl0, tbl1, tbl2, tbl3, a_3, b_3);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_stride_k1 * dot_factor);
          A_k1 = offset_bytes(A_k1, A_stride_m * dot_factor);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    svbool_t n_mask0 = svwhilelt(0 * svl, n, int32_t{});
    svbool_t n_mask1 = svwhilelt(1 * svl, n, int32_t{});
    svbool_t n_mask2 = svwhilelt(2 * svl, n, int32_t{});
    svbool_t n_mask3 = svwhilelt(3 * svl, n, int32_t{});
    for (size_t m = 0; m < M; ++m) {
      void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
      svst1_hor_za32(0, m, n_mask0,
                     offset_bytes(C_out_m, 0 * svl * sizeof(int32_t)));
      svst1_hor_za32(1, m, n_mask1,
                     offset_bytes(C_out_m, 1 * svl * sizeof(int32_t)));
      svst1_hor_za32(2, m, n_mask2,
                     offset_bytes(C_out_m, 2 * svl * sizeof(int32_t)));
      svst1_hor_za32(3, m, n_mask3,
                     offset_bytes(C_out_m, 3 * svl * sizeof(int32_t)));
    }
  }
}

}  // namespace

void dot_fp32_sme(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                  size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
                  const void* A, size_t B_stride_k3, size_t B_stride_k2,
                  size_t B_stride_k1, const void* B, size_t C_in_stride_m,
                  const void* C_in, size_t C_out_stride_m, void* C_out) {
  sme_dot<float, float>(M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2,
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
  sme_dot<bfloat16_t, float>(
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
  sme_dot<float16_t, float>(
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
  sme_dot<int8_t, int32_t>(
      M, N, K3, K2, K1, A_stride_m, A_stride_k3, A_stride_k2, A, B_stride_k3,
      B_stride_k2, B_stride_k1, B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_int8_int4_int32_sme(size_t M, size_t N, size_t K3, size_t K2,
                             size_t K1, size_t A_stride_m, size_t A_stride_k3,
                             size_t A_stride_k2, const void* A,
                             size_t B_stride_k3, size_t B_stride_k2,
                             size_t B_stride_k1, const void* B,
                             size_t C_in_stride_m, const void* C_in,
                             size_t C_out_stride_m, void* C_out) {
  sme_dot_int8_int4_int32(M, N, K3, K2, K1, A_stride_m, A_stride_k3,
                          A_stride_k2, A, B_stride_k3, B_stride_k2, B_stride_k1,
                          B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

void dot_int8_int2_int32_sme(size_t M, size_t N, size_t K3, size_t K2,
                             size_t K1, size_t A_stride_m, size_t A_stride_k3,
                             size_t A_stride_k2, const void* A,
                             size_t B_stride_k3, size_t B_stride_k2,
                             size_t B_stride_k1, const void* B,
                             size_t C_in_stride_m, const void* C_in,
                             size_t C_out_stride_m, void* C_out) {
  sme_dot_int8_int2_int32(M, N, K3, K2, K1, A_stride_m, A_stride_k3,
                          A_stride_k2, A, B_stride_k3, B_stride_k2, B_stride_k1,
                          B, C_in_stride_m, C_in, C_out_stride_m, C_out);
}

}  // namespace ynn

#endif  // YNN_DISABLE_SME
