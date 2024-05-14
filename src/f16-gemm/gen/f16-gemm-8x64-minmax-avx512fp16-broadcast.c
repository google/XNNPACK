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

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f16_gemm_minmax_ukernel_8x64__avx512fp16_broadcast(
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
  assert(mr <= 8);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a0 = (const uint16_t*) a;
  uint16_t* c0 = (uint16_t*) c;
  const uint16_t* a1 = (const uint16_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint16_t* a2 = (const uint16_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint16_t* a3 = (const uint16_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const uint16_t* a4 = (const uint16_t*) ((uintptr_t) a3 + a_stride);
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const uint16_t* a5 = (const uint16_t*) ((uintptr_t) a4 + a_stride);
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const uint16_t* a6 = (const uint16_t*) ((uintptr_t) a5 + a_stride);
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const uint16_t* a7 = (const uint16_t*) ((uintptr_t) a6 + a_stride);
  uint16_t* c7 = (uint16_t*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 8) {
    a7 = a6;
    c7 = c6;
  }

  do {
    __m512h vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_load_ph(w);
    __m512h vacc0xWXYZ0123456789abcdefghijklmnopqr = _mm512_load_ph((const uint16_t*) w + 32);
    __m512h vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc1xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    __m512h vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc2xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    __m512h vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc3xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    __m512h vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc4xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    __m512h vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc5xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    __m512h vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc6xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    __m512h vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV;
    __m512h vacc7xWXYZ0123456789abcdefghijklmnopqr = vacc0xWXYZ0123456789abcdefghijklmnopqr;
    w = (const uint16_t*) w + 64;

    size_t k = kc;
    do {
      const __m512h vb0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_load_ph(w);
      const __m512h vbWXYZ0123456789abcdefghijklmnopqr = _mm512_load_ph((const uint16_t*) w + 32);
      w = (const uint16_t*) w + 64;

      const __m512h va0 = _mm512_castsi512_ph(_mm512_set1_epi16(*a0));
      vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va0, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc0xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va0, vbWXYZ0123456789abcdefghijklmnopqr, vacc0xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va1 = _mm512_castsi512_ph(_mm512_set1_epi16(*a1));
      vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va1, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc1xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va1, vbWXYZ0123456789abcdefghijklmnopqr, vacc1xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va2 = _mm512_castsi512_ph(_mm512_set1_epi16(*a2));
      vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va2, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc2xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va2, vbWXYZ0123456789abcdefghijklmnopqr, vacc2xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va3 = _mm512_castsi512_ph(_mm512_set1_epi16(*a3));
      vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va3, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc3xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va3, vbWXYZ0123456789abcdefghijklmnopqr, vacc3xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va4 = _mm512_castsi512_ph(_mm512_set1_epi16(*a4));
      vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va4, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc4xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va4, vbWXYZ0123456789abcdefghijklmnopqr, vacc4xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va5 = _mm512_castsi512_ph(_mm512_set1_epi16(*a5));
      vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va5, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc5xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va5, vbWXYZ0123456789abcdefghijklmnopqr, vacc5xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va6 = _mm512_castsi512_ph(_mm512_set1_epi16(*a6));
      vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va6, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc6xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va6, vbWXYZ0123456789abcdefghijklmnopqr, vacc6xWXYZ0123456789abcdefghijklmnopqr);
      const __m512h va7 = _mm512_castsi512_ph(_mm512_set1_epi16(*a7));
      vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_fmadd_ph(va7, vb0123456789ABCDEFGHIJKLMNOPQRSTUV, vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      vacc7xWXYZ0123456789abcdefghijklmnopqr = _mm512_fmadd_ph(va7, vbWXYZ0123456789abcdefghijklmnopqr, vacc7xWXYZ0123456789abcdefghijklmnopqr);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;
      a5 += 1;
      a6 += 1;
      a7 += 1;

      k -= sizeof(uint16_t);
    } while (k != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_max_ph(vmin, vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc0xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc0xWXYZ0123456789abcdefghijklmnopqr);
    vacc1xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc1xWXYZ0123456789abcdefghijklmnopqr);
    vacc2xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc2xWXYZ0123456789abcdefghijklmnopqr);
    vacc3xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc3xWXYZ0123456789abcdefghijklmnopqr);
    vacc4xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc4xWXYZ0123456789abcdefghijklmnopqr);
    vacc5xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc5xWXYZ0123456789abcdefghijklmnopqr);
    vacc6xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc6xWXYZ0123456789abcdefghijklmnopqr);
    vacc7xWXYZ0123456789abcdefghijklmnopqr = _mm512_max_ph(vmin, vacc7xWXYZ0123456789abcdefghijklmnopqr);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV = _mm512_min_ph(vmax, vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV);
    vacc0xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc0xWXYZ0123456789abcdefghijklmnopqr);
    vacc1xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc1xWXYZ0123456789abcdefghijklmnopqr);
    vacc2xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc2xWXYZ0123456789abcdefghijklmnopqr);
    vacc3xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc3xWXYZ0123456789abcdefghijklmnopqr);
    vacc4xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc4xWXYZ0123456789abcdefghijklmnopqr);
    vacc5xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc5xWXYZ0123456789abcdefghijklmnopqr);
    vacc6xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc6xWXYZ0123456789abcdefghijklmnopqr);
    vacc7xWXYZ0123456789abcdefghijklmnopqr = _mm512_min_ph(vmax, vacc7xWXYZ0123456789abcdefghijklmnopqr);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ph(c0, vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c0 + 32, vacc0xWXYZ0123456789abcdefghijklmnopqr);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm512_storeu_ph(c1, vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c1 + 32, vacc1xWXYZ0123456789abcdefghijklmnopqr);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ph(c2, vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c2 + 32, vacc2xWXYZ0123456789abcdefghijklmnopqr);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ph(c3, vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c3 + 32, vacc3xWXYZ0123456789abcdefghijklmnopqr);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ph(c4, vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c4 + 32, vacc4xWXYZ0123456789abcdefghijklmnopqr);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ph(c5, vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c5 + 32, vacc5xWXYZ0123456789abcdefghijklmnopqr);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ph(c6, vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c6 + 32, vacc6xWXYZ0123456789abcdefghijklmnopqr);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ph(c7, vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV);
      _mm512_storeu_ph((uint16_t*) c7 + 32, vacc7xWXYZ0123456789abcdefghijklmnopqr);
      c7 = (uint16_t*) ((uintptr_t) c7 + cn_stride);

      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);
      a6 = (const uint16_t*) ((uintptr_t) a6 - kc);
      a7 = (const uint16_t*) ((uintptr_t) a7 - kc);

      nc -= 64;
    } else {
      if (nc & 32) {
        _mm512_storeu_ph(c0, vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c1, vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c2, vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c3, vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c4, vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c5, vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c6, vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV);
        _mm512_storeu_ph(c7, vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV);

        vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc0xWXYZ0123456789abcdefghijklmnopqr;
        vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc1xWXYZ0123456789abcdefghijklmnopqr;
        vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc2xWXYZ0123456789abcdefghijklmnopqr;
        vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc3xWXYZ0123456789abcdefghijklmnopqr;
        vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc4xWXYZ0123456789abcdefghijklmnopqr;
        vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc5xWXYZ0123456789abcdefghijklmnopqr;
        vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc6xWXYZ0123456789abcdefghijklmnopqr;
        vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV = vacc7xWXYZ0123456789abcdefghijklmnopqr;

        c0 += 32;
        c1 += 32;
        c2 += 32;
        c3 += 32;
        c4 += 32;
        c5 += 32;
        c6 += 32;
        c7 += 32;
      }
      if (nc & 31) {
        // Prepare mask for valid 16-bit elements (depends on nc).
        const __mmask32 vmask = _cvtu32_mask32((uint32_t) (UINT32_C(1) << (nc & 31)) - UINT32_C(1));
        _mm512_mask_storeu_epi16(c0, vmask, _mm512_castph_si512(vacc0x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c1, vmask, _mm512_castph_si512(vacc1x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c2, vmask, _mm512_castph_si512(vacc2x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c3, vmask, _mm512_castph_si512(vacc3x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c4, vmask, _mm512_castph_si512(vacc4x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c5, vmask, _mm512_castph_si512(vacc5x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c6, vmask, _mm512_castph_si512(vacc6x0123456789ABCDEFGHIJKLMNOPQRSTUV));
        _mm512_mask_storeu_epi16(c7, vmask, _mm512_castph_si512(vacc7x0123456789ABCDEFGHIJKLMNOPQRSTUV));
      }
      nc = 0;
    }
  } while (nc != 0);
#endif  // defined(__AVX512FP16__)
}
