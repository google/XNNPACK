// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/x16-packw/gio-c2-avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/packw.h"
#include "src/xnnpack/prefetch.h"


void xnn_x16_packw_gemm_gio_ukernel_x32c2__avx512skx_u2_prfm(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t k_stride,
  const uint16_t* weights,
  const uint16_t* bias,
  const void* scale,
  uint16_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 32);   // This kernel is for NR=32
  assert(kr == 2);
  assert(sr == 1);
  assert(k_stride != 0);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  const __m512i vzero = _mm512_setzero_si512();
  const __m512i vperm = _mm512_set_epi64(7, 3, 6, 2, 5, 1, 4, 0);
  const uint16_t* b = bias;
  uint16_t* packed_w = packed_weights;
  do {
    // NC main loop multiple of 32
    const uint16_t* w = weights;
    size_t n = nc;

    for (; n >= 32; n -= 32) {
      if XNN_LIKELY(b != NULL) {
        const __m512i vb0 = _mm512_loadu_si512((const __m512i*) (b + 0));
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vb0);
        b += 32;
      } else {
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vzero);
      }
      packed_w += 32;

      const uint16_t* w_col = w;
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const __m512i v0_0 = _mm512_permutexvar_epi64(vperm, _mm512_loadu_si512((const __m512i*) (w_col + 0)));
        const __m512i v1_0 = _mm512_permutexvar_epi64(vperm, _mm512_loadu_si512((const __m512i*) (w_col + k_stride + 0)));
        xnn_prefetch_to_l1((const int8_t*) (w_col + 2 * k_stride + 0));
        xnn_prefetch_to_l1((const int8_t*) (w_col + 3 * k_stride + 0));
        const __m512i vlo_0 = _mm512_unpacklo_epi16(v0_0, v1_0);
        const __m512i vhi_0 = _mm512_unpackhi_epi16(v0_0, v1_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vlo_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vhi_0);
        w_col += 2 * k_stride;
        packed_w += 64;
      }

      if XNN_UNLIKELY(k != 0) {
        const __m512i v0_0 = _mm512_permutexvar_epi64(vperm, _mm512_loadu_si512((const __m512i*) (w_col + 0)));
        const __m512i vlo_0 = _mm512_unpacklo_epi16(v0_0, vzero);
        const __m512i vhi_0 = _mm512_unpackhi_epi16(v0_0, vzero);
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vlo_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vhi_0);
        packed_w += 64;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
      w += 32;
    }

    // NC remainder (1..31)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1);
      assert(n <= 31);

      const __mmask32 vmask0 = _cvtu32_mask32((uint32_t) ((UINT64_C(1) << n) - UINT64_C(1)));

      if XNN_LIKELY(b != NULL) {
        const __m512i vb0 = _mm512_maskz_loadu_epi16(vmask0, b + 0);
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vb0);
        b += n;
      } else {
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vzero);
      }
      packed_w += 32;

      const uint16_t* w_col = w;
      size_t k = kc;
      for (; k >= 2; k -= 2) {
        const __m512i v0_0 = _mm512_permutexvar_epi64(vperm, _mm512_maskz_loadu_epi16(vmask0, w_col + 0));
        const __m512i v1_0 = _mm512_permutexvar_epi64(vperm, _mm512_maskz_loadu_epi16(vmask0, w_col + k_stride + 0));
        const __m512i vlo_0 = _mm512_unpacklo_epi16(v0_0, v1_0);
        const __m512i vhi_0 = _mm512_unpackhi_epi16(v0_0, v1_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vlo_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vhi_0);
        w_col += 2 * k_stride;
        packed_w += 64;
      }

      if XNN_UNLIKELY(k != 0) {
        const __m512i v0_0 = _mm512_permutexvar_epi64(vperm, _mm512_maskz_loadu_epi16(vmask0, w_col + 0));
        const __m512i vlo_0 = _mm512_unpacklo_epi16(v0_0, vzero);
        const __m512i vhi_0 = _mm512_unpackhi_epi16(v0_0, vzero);
        _mm512_storeu_si512((__m512i*) (packed_w + 0), vlo_0);
        _mm512_storeu_si512((__m512i*) (packed_w + 32), vhi_0);
        packed_w += 64;
      }
      packed_w = (uint16_t*) ((uintptr_t) packed_w + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
