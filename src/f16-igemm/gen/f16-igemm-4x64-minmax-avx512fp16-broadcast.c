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


void xnn_f16_igemm_minmax_ukernel_4x64__avx512fp16_broadcast(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

#if defined(__AVX512FP16__)
  uint16_t* c0 = (uint16_t*) c;
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    __m512h vacc0x0 = _mm512_load_ph(w);
    __m512h vacc0x1 = _mm512_load_ph((const uint16_t*) w + 32);
    __m512h vacc1x0 = vacc0x0;
    __m512h vacc1x1 = vacc0x1;
    __m512h vacc2x0 = vacc0x0;
    __m512h vacc2x1 = vacc0x1;
    __m512h vacc3x0 = vacc0x0;
    __m512h vacc3x1 = vacc0x1;
    w = (const uint16_t*) w + 64;

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint16_t* restrict a1 = (const uint16_t*) a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint16_t*) ((uintptr_t) a1 + a_offset);
      }
      const uint16_t* restrict a2 = (const uint16_t*) a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const uint16_t*) ((uintptr_t) a2 + a_offset);
      }
      const uint16_t* restrict a3 = (const uint16_t*) a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const uint16_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const __m512h vb0 = _mm512_load_ph(w);
        const __m512h vb1 = _mm512_load_ph((const uint16_t*) w + 32);
        w = (const uint16_t*) w + 64;

        const __m512h va0 = _mm512_castsi512_ph(_mm512_set1_epi16(*a0));
        vacc0x0 = _mm512_fmadd_ph(va0, vb0, vacc0x0);
        vacc0x1 = _mm512_fmadd_ph(va0, vb1, vacc0x1);
        a0 += 1;
        const __m512h va1 = _mm512_castsi512_ph(_mm512_set1_epi16(*a1));
        vacc1x0 = _mm512_fmadd_ph(va1, vb0, vacc1x0);
        vacc1x1 = _mm512_fmadd_ph(va1, vb1, vacc1x1);
        a1 += 1;
        const __m512h va2 = _mm512_castsi512_ph(_mm512_set1_epi16(*a2));
        vacc2x0 = _mm512_fmadd_ph(va2, vb0, vacc2x0);
        vacc2x1 = _mm512_fmadd_ph(va2, vb1, vacc2x1);
        a2 += 1;
        const __m512h va3 = _mm512_castsi512_ph(_mm512_set1_epi16(*a3));
        vacc3x0 = _mm512_fmadd_ph(va3, vb0, vacc3x0);
        vacc3x1 = _mm512_fmadd_ph(va3, vb1, vacc3x1);
        a3 += 1;

        k -= sizeof(uint16_t);
      } while (k != 0);

      p -= 4 * sizeof(void*);
    } while (p != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0 = _mm512_max_ph(vmin, vacc0x0);
    vacc1x0 = _mm512_max_ph(vmin, vacc1x0);
    vacc2x0 = _mm512_max_ph(vmin, vacc2x0);
    vacc3x0 = _mm512_max_ph(vmin, vacc3x0);
    vacc0x1 = _mm512_max_ph(vmin, vacc0x1);
    vacc1x1 = _mm512_max_ph(vmin, vacc1x1);
    vacc2x1 = _mm512_max_ph(vmin, vacc2x1);
    vacc3x1 = _mm512_max_ph(vmin, vacc3x1);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0 = _mm512_min_ph(vmax, vacc0x0);
    vacc1x0 = _mm512_min_ph(vmax, vacc1x0);
    vacc2x0 = _mm512_min_ph(vmax, vacc2x0);
    vacc3x0 = _mm512_min_ph(vmax, vacc3x0);
    vacc0x1 = _mm512_min_ph(vmax, vacc0x1);
    vacc1x1 = _mm512_min_ph(vmax, vacc1x1);
    vacc2x1 = _mm512_min_ph(vmax, vacc2x1);
    vacc3x1 = _mm512_min_ph(vmax, vacc3x1);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ph(c3, vacc3x0);
      _mm512_storeu_ph((uint16_t*) c3 + 1, vacc3x1);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ph(c2, vacc2x0);
      _mm512_storeu_ph((uint16_t*) c2 + 1, vacc2x1);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ph(c1, vacc1x0);
      _mm512_storeu_ph((uint16_t*) c1 + 1, vacc1x1);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ph(c0, vacc0x0);
      _mm512_storeu_ph((uint16_t*) c0 + 1, vacc0x1);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
      nc -= 64;
    } else {
      if (nc & 32) {
        _mm512_storeu_ph(c3, vacc3x0);
        _mm512_storeu_ph(c2, vacc2x0);
        _mm512_storeu_ph(c1, vacc1x0);
        _mm512_storeu_ph(c0, vacc0x0);

        vacc3x0 = vacc3x1;
        c3 += 32;
        vacc2x0 = vacc2x1;
        c2 += 32;
        vacc1x0 = vacc1x1;
        c1 += 32;
        vacc0x0 = vacc0x1;
        c0 += 32;
      }
      if (nc & 31) {
        // Prepare mask for valid 16-bit elements (depends on nc).
        const __mmask32 vmask = _cvtu32_mask32((uint32_t) (UINT32_C(1) << (nc & 31)) - UINT32_C(1));
        _mm512_mask_storeu_epi16(c3, vmask, _mm512_castph_si512(vacc3x0));
        _mm512_mask_storeu_epi16(c2, vmask, _mm512_castph_si512(vacc2x0));
        _mm512_mask_storeu_epi16(c1, vmask, _mm512_castph_si512(vacc1x0));
        _mm512_mask_storeu_epi16(c0, vmask, _mm512_castph_si512(vacc0x0));
      }
      nc = 0;
    }
  } while (nc != 0);
#endif  // defined(__AVX512FP16__)
}
