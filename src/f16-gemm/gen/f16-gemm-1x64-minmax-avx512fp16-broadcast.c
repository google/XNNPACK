// Auto-generated file. Do not edit!
//   Template: src/f16-gemm/avx512fp16-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_gemm_minmax_ukernel_1x64__avx512fp16_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const void* restrict a,
    size_t a_stride,
    const void* restrict w,
    void* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;

  do {
    __m512h vacc0x0 = _mm512_load_ph(w);
    __m512h vacc0x1 = _mm512_load_ph((const uint16_t*) w + 32);
    w = (const uint16_t*) w + 64;

    size_t k = kc;
    do {
      const __m512h vb0 = _mm512_load_ph(w);
      const __m512h vb1 = _mm512_load_ph((const uint16_t*) w + 32);
      w = (const uint16_t*) w + 64;

      const __m512h va0 = _mm512_castsi512_ph(_mm512_set1_epi16(*a0));
      vacc0x0 = _mm512_fmadd_ph(va0, vb0, vacc0x0);
      vacc0x1 = _mm512_fmadd_ph(va0, vb1, vacc0x1);
      a0 += 1;

      k -= sizeof(uint16_t);
    } while (k != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0 = _mm512_max_ph(vmin, vacc0x0);
    vacc0x1 = _mm512_max_ph(vmin, vacc0x1);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0 = _mm512_min_ph(vmax, vacc0x0);
    vacc0x1 = _mm512_min_ph(vmax, vacc0x1);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ph(c0, vacc0x0);
      _mm512_storeu_ph((uint16_t*) c0 + 1, vacc0x1);
      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      nc -= 64;
    } else {
      if (nc & 32) {
        _mm512_storeu_ph(c0, vacc0x0);

        vacc0x0 = vacc0x1;
        c0 += 32;
      }
      if (nc & 31) {
        // Prepare mask for valid 16-bit elements (depends on nc).
        const __mmask32 vmask = _cvtu32_mask32((uint32_t) (UINT32_C(1) << (nc & 31)) - UINT32_C(1));
        _mm512_mask_storeu_epi16(c0, vmask, _mm512_castph_si512(vacc0x0));
      }
      nc = 0;
    }
  } while (nc != 0);
#endif  // defined(__AVX512FP16__)
}
