// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/avx512fp16-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_igemm_minmax_ukernel_1x32__avx512fp16_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const void** restrict a,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

#if defined(__AVX512FP16__)
  uint16_t* c0 = (uint16_t*) c;

  do {
    __m512h vacc0x0 = _mm512_load_ph(w);
    w = (const uint16_t*) w + 32;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const __m512h vb0 = _mm512_load_ph(w);
        w = (const uint16_t*) w + 32;

        const __m512h va0 = _mm512_castsi512_ph(_mm512_set1_epi16(*a0));
        vacc0x0 = _mm512_fmadd_ph(va0, vb0, vacc0x0);
        a0 += 1;

        k -= sizeof(uint16_t);
      } while (k != 0);

      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0 = _mm512_max_ph(vmin, vacc0x0);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0 = _mm512_min_ph(vmax, vacc0x0);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ph(c0, vacc0x0);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 32;
    } else {
      assert(nc != 0);
      assert(nc < 32);
      // Prepare mask for valid 16-bit elements (depends on nc).
      const __mmask32 vmask = _cvtu32_mask32((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1));
      _mm512_mask_storeu_epi16(c0, vmask, _mm512_castph_si512(vacc0x0));
      nc = 0;
    }
  } while (nc != 0);
#endif  // defined(__AVX512FP16__)
}
