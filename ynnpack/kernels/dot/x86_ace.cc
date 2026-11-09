// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/x86_amx.h"

namespace ynn {

namespace {

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))

// ACEv1 TOP4BSSD tmmDst, zmm1, zmm2:
//   tmmDst[m, n] += sum_{c=0..3} s8(va[m, c]) * s8(vb[n, c])
// Encoded as EVEX.512.F2.0F38.W0 0x5E /r with ModRM.rm = zmm1 (va) and
// EVEX.vvvv = ~2 -> zmm2 (vb). Marked volatile because tmmDst is unmodeled
// architectural tile state.
YNN_ALWAYS_INLINE void top4bssd_tmm0(__m512i va, __m512i vb) {
  register __m512i z1 __asm__("zmm1") = va;
  register __m512i z2 __asm__("zmm2") = vb;
  __asm__ volatile(".byte 0x62, 0xf2, 0x6f, 0x48, 0x5e, 0xc1"
                   :
                   : "v"(z1), "v"(z2));
}

YNN_ALWAYS_INLINE void top4bssd_tmm1(__m512i va, __m512i vb) {
  register __m512i z1 __asm__("zmm1") = va;
  register __m512i z2 __asm__("zmm2") = vb;
  __asm__ volatile(".byte 0x62, 0xf2, 0x6f, 0x48, 0x5e, 0xc9"
                   :
                   : "v"(z1), "v"(z2));
}

// AMX-AVX512 TILEMOVROW zmm0, tmmSrc, eax:
//   Extracts 64-byte row `row` (16 x int32_t) from tmmSrc directly into zmm0.
// Encoded as EVEX.512.66.0F38.W0 0x4A /r with ModRM.reg = zmm0, ModRM.rm =
// tmmSrc, and EVEX.vvvv = ~0 -> eax. Marked volatile because tmmSrc is
// unmodeled architectural tile state.
YNN_ALWAYS_INLINE __m512i tilemovrow_tmm0(uint32_t row) {
  register __m512i z0 __asm__("zmm0");
  __asm__ volatile(".byte 0x62, 0xf2, 0x7d, 0x48, 0x4a, 0xc0"
                   : "=v"(z0)
                   : "a"(row));
  return z0;
}

YNN_ALWAYS_INLINE __m512i tilemovrow_tmm1(uint32_t row) {
  register __m512i z0 __asm__("zmm0");
  __asm__ volatile(".byte 0x62, 0xf2, 0x7d, 0x48, 0x4a, 0xc1"
                   : "=v"(z0)
                   : "a"(row));
  return z0;
}

#else

YNN_ALWAYS_INLINE void top4bssd_tmm0(__m512i, __m512i) { YNN_UNREACHABLE; }
YNN_ALWAYS_INLINE void top4bssd_tmm1(__m512i, __m512i) { YNN_UNREACHABLE; }
YNN_ALWAYS_INLINE __m512i tilemovrow_tmm0(uint32_t) {
  YNN_UNREACHABLE;
  return _mm512_setzero_si512();
}
YNN_ALWAYS_INLINE __m512i tilemovrow_tmm1(uint32_t) {
  YNN_UNREACHABLE;
  return _mm512_setzero_si512();
}

#endif

}  // namespace

void dot_int8_int8_int32_16x32x4_1x16x4_ace(
    size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
    size_t A_stride_k3, size_t A_stride_k2, const void* A, size_t B_stride_k3,
    size_t B_stride_k2, size_t B_stride_k1, const void* B, size_t C_in_stride_m,
    const void* C_in, size_t C_out_stride_m, void* C_out,
    dot_kernel_state* state) {
  assert(state != nullptr);
  assert(M > 0);
  assert(N > 0);
  assert(K3 > 0);
  assert(K2 > 0);
  assert(K1 > 0);
  assert(M <= 16);

  constexpr size_t dot_factor = 4;
  constexpr size_t tile_cols = 16;
  constexpr size_t tile_row_bytes = tile_cols * sizeof(int32_t);

  // Palette 2 configures 16 architectural 16x16x32b tiles with zeroed row/col
  // metadata fields.
  alignas(64) tile_config config = {0};
  config.palette_id = 2;
  load_if_needed(*state, config);

  const __mmask16 m_mask =
      static_cast<__mmask16>((UINT32_C(1) << M) - UINT32_C(1));
  const size_t B_step_k1 = B_stride_k1 * dot_factor;
  const size_t A_step_k1 = A_stride_m * dot_factor;

  while (N >= 2 * tile_cols) {
    _tile_zero(0);
    _tile_zero(1);

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
          const __m512i va = _mm512_maskz_loadu_epi32(m_mask, A_k1);
          const __m512i vb0 = _mm512_loadu_si512(B_k1);
          const __m512i vb1 =
              _mm512_loadu_si512(offset_bytes(B_k1, tile_row_bytes));
          top4bssd_tmm0(va, vb0);
          top4bssd_tmm1(va, vb1);

          k1 -= dot_factor;
          B_k1 = offset_bytes(B_k1, B_step_k1);
          A_k1 = offset_bytes(A_k1, A_step_k1);
        }
        k2 -= 1;
        B_k2 = offset_bytes(B_k2, B_stride_k2);
        A_k2 = offset_bytes(A_k2, A_stride_k2);
      } while (k2 > 0);
      k3 -= 1;
      B_k3 = offset_bytes(B_k3, B_stride_k3);
      A_k3 = offset_bytes(A_k3, A_stride_k3);
    } while (k3 > 0);

    if (C_in) {
      for (size_t m = 0; m < M; ++m) {
        const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
        void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
        const __m512i vacc0 =
            _mm512_add_epi32(tilemovrow_tmm0(static_cast<uint32_t>(m)),
                             _mm512_loadu_si512(offset_bytes(C_in_m, 0)));
        const __m512i vacc1 = _mm512_add_epi32(
            tilemovrow_tmm1(static_cast<uint32_t>(m)),
            _mm512_loadu_si512(offset_bytes(C_in_m, tile_row_bytes)));
        _mm512_storeu_si512(offset_bytes(C_out_m, 0), vacc0);
        _mm512_storeu_si512(offset_bytes(C_out_m, tile_row_bytes), vacc1);
      }
      C_in = offset_bytes(C_in, 2 * tile_row_bytes);
    } else {
      for (size_t m = 0; m < M; ++m) {
        void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
        _mm512_storeu_si512(offset_bytes(C_out_m, 0),
                            tilemovrow_tmm0(static_cast<uint32_t>(m)));
        _mm512_storeu_si512(offset_bytes(C_out_m, tile_row_bytes),
                            tilemovrow_tmm1(static_cast<uint32_t>(m)));
      }
    }

    C_out = offset_bytes(C_out, 2 * tile_row_bytes);
    B = offset_bytes(B, 2 * tile_row_bytes);
    N -= 2 * tile_cols;
  }

  if (N > 0) {
    if (N > tile_cols) {
      const __mmask16 n_mask1 = static_cast<__mmask16>(
          (UINT32_C(1) << (N - tile_cols)) - UINT32_C(1));
      _tile_zero(0);
      _tile_zero(1);

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
            const __m512i va = _mm512_maskz_loadu_epi32(m_mask, A_k1);
            const __m512i vb0 = _mm512_loadu_si512(B_k1);
            const __m512i vb1 = _mm512_maskz_loadu_epi32(
                n_mask1, offset_bytes(B_k1, tile_row_bytes));
            top4bssd_tmm0(va, vb0);
            top4bssd_tmm1(va, vb1);

            k1 -= dot_factor;
            B_k1 = offset_bytes(B_k1, B_step_k1);
            A_k1 = offset_bytes(A_k1, A_step_k1);
          }
          k2 -= 1;
          B_k2 = offset_bytes(B_k2, B_stride_k2);
          A_k2 = offset_bytes(A_k2, A_stride_k2);
        } while (k2 > 0);
        k3 -= 1;
        B_k3 = offset_bytes(B_k3, B_stride_k3);
        A_k3 = offset_bytes(A_k3, A_stride_k3);
      } while (k3 > 0);

      if (C_in) {
        for (size_t m = 0; m < M; ++m) {
          const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
          void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
          const __m512i vacc0 =
              _mm512_add_epi32(tilemovrow_tmm0(static_cast<uint32_t>(m)),
                               _mm512_loadu_si512(offset_bytes(C_in_m, 0)));
          const __m512i vacc1 = _mm512_add_epi32(
              tilemovrow_tmm1(static_cast<uint32_t>(m)),
              _mm512_maskz_loadu_epi32(n_mask1,
                                       offset_bytes(C_in_m, tile_row_bytes)));
          _mm512_storeu_si512(offset_bytes(C_out_m, 0), vacc0);
          _mm512_mask_storeu_epi32(offset_bytes(C_out_m, tile_row_bytes),
                                   n_mask1, vacc1);
        }
      } else {
        for (size_t m = 0; m < M; ++m) {
          void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
          _mm512_storeu_si512(offset_bytes(C_out_m, 0),
                              tilemovrow_tmm0(static_cast<uint32_t>(m)));
          _mm512_mask_storeu_epi32(offset_bytes(C_out_m, tile_row_bytes),
                                   n_mask1,
                                   tilemovrow_tmm1(static_cast<uint32_t>(m)));
        }
      }
    } else {
      const __mmask16 n_mask0 =
          static_cast<__mmask16>((UINT32_C(1) << N) - UINT32_C(1));
      _tile_zero(0);

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
            const __m512i va = _mm512_maskz_loadu_epi32(m_mask, A_k1);
            const __m512i vb0 = _mm512_maskz_loadu_epi32(n_mask0, B_k1);
            top4bssd_tmm0(va, vb0);

            k1 -= dot_factor;
            B_k1 = offset_bytes(B_k1, B_step_k1);
            A_k1 = offset_bytes(A_k1, A_step_k1);
          }
          k2 -= 1;
          B_k2 = offset_bytes(B_k2, B_stride_k2);
          A_k2 = offset_bytes(A_k2, A_stride_k2);
        } while (k2 > 0);
        k3 -= 1;
        B_k3 = offset_bytes(B_k3, B_stride_k3);
        A_k3 = offset_bytes(A_k3, A_stride_k3);
      } while (k3 > 0);

      if (C_in) {
        for (size_t m = 0; m < M; ++m) {
          const void* C_in_m = offset_bytes(C_in, m * C_in_stride_m);
          void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
          const __m512i vacc0 = _mm512_add_epi32(
              tilemovrow_tmm0(static_cast<uint32_t>(m)),
              _mm512_maskz_loadu_epi32(n_mask0, offset_bytes(C_in_m, 0)));
          _mm512_mask_storeu_epi32(offset_bytes(C_out_m, 0), n_mask0, vacc0);
        }
      } else {
        for (size_t m = 0; m < M; ++m) {
          void* C_out_m = offset_bytes(C_out, m * C_out_stride_m);
          _mm512_mask_storeu_epi32(offset_bytes(C_out_m, 0), n_mask0,
                                   tilemovrow_tmm0(static_cast<uint32_t>(m)));
        }
      }
    }
  }

  set_amx_destructor(*state);
}

}  // namespace ynn
