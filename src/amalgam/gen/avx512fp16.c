// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/gemm.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/reduce.h"
#include "xnnpack/vbinary.h"


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

void xnn_f16_gemm_minmax_ukernel_7x64__avx512fp16_broadcast(
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
  assert(mr <= 7);
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

  do {
    __m512h vacc0x0 = _mm512_load_ph(w);
    __m512h vacc0x1 = _mm512_load_ph((const uint16_t*) w + 32);
    __m512h vacc1x0 = vacc0x0;
    __m512h vacc1x1 = vacc0x1;
    __m512h vacc2x0 = vacc0x0;
    __m512h vacc2x1 = vacc0x1;
    __m512h vacc3x0 = vacc0x0;
    __m512h vacc3x1 = vacc0x1;
    __m512h vacc4x0 = vacc0x0;
    __m512h vacc4x1 = vacc0x1;
    __m512h vacc5x0 = vacc0x0;
    __m512h vacc5x1 = vacc0x1;
    __m512h vacc6x0 = vacc0x0;
    __m512h vacc6x1 = vacc0x1;
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
      const __m512h va4 = _mm512_castsi512_ph(_mm512_set1_epi16(*a4));
      vacc4x0 = _mm512_fmadd_ph(va4, vb0, vacc4x0);
      vacc4x1 = _mm512_fmadd_ph(va4, vb1, vacc4x1);
      a4 += 1;
      const __m512h va5 = _mm512_castsi512_ph(_mm512_set1_epi16(*a5));
      vacc5x0 = _mm512_fmadd_ph(va5, vb0, vacc5x0);
      vacc5x1 = _mm512_fmadd_ph(va5, vb1, vacc5x1);
      a5 += 1;
      const __m512h va6 = _mm512_castsi512_ph(_mm512_set1_epi16(*a6));
      vacc6x0 = _mm512_fmadd_ph(va6, vb0, vacc6x0);
      vacc6x1 = _mm512_fmadd_ph(va6, vb1, vacc6x1);
      a6 += 1;

      k -= sizeof(uint16_t);
    } while (k != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0 = _mm512_max_ph(vmin, vacc0x0);
    vacc1x0 = _mm512_max_ph(vmin, vacc1x0);
    vacc2x0 = _mm512_max_ph(vmin, vacc2x0);
    vacc3x0 = _mm512_max_ph(vmin, vacc3x0);
    vacc4x0 = _mm512_max_ph(vmin, vacc4x0);
    vacc5x0 = _mm512_max_ph(vmin, vacc5x0);
    vacc6x0 = _mm512_max_ph(vmin, vacc6x0);
    vacc0x1 = _mm512_max_ph(vmin, vacc0x1);
    vacc1x1 = _mm512_max_ph(vmin, vacc1x1);
    vacc2x1 = _mm512_max_ph(vmin, vacc2x1);
    vacc3x1 = _mm512_max_ph(vmin, vacc3x1);
    vacc4x1 = _mm512_max_ph(vmin, vacc4x1);
    vacc5x1 = _mm512_max_ph(vmin, vacc5x1);
    vacc6x1 = _mm512_max_ph(vmin, vacc6x1);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0 = _mm512_min_ph(vmax, vacc0x0);
    vacc1x0 = _mm512_min_ph(vmax, vacc1x0);
    vacc2x0 = _mm512_min_ph(vmax, vacc2x0);
    vacc3x0 = _mm512_min_ph(vmax, vacc3x0);
    vacc4x0 = _mm512_min_ph(vmax, vacc4x0);
    vacc5x0 = _mm512_min_ph(vmax, vacc5x0);
    vacc6x0 = _mm512_min_ph(vmax, vacc6x0);
    vacc0x1 = _mm512_min_ph(vmax, vacc0x1);
    vacc1x1 = _mm512_min_ph(vmax, vacc1x1);
    vacc2x1 = _mm512_min_ph(vmax, vacc2x1);
    vacc3x1 = _mm512_min_ph(vmax, vacc3x1);
    vacc4x1 = _mm512_min_ph(vmax, vacc4x1);
    vacc5x1 = _mm512_min_ph(vmax, vacc5x1);
    vacc6x1 = _mm512_min_ph(vmax, vacc6x1);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ph(c0, vacc0x0);
      _mm512_storeu_ph((uint16_t*) c0 + 1, vacc0x1);
      a0 = (const uint16_t*) ((uintptr_t) a0 - kc);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      _mm512_storeu_ph(c1, vacc1x0);
      _mm512_storeu_ph((uint16_t*) c1 + 1, vacc1x1);
      a1 = (const uint16_t*) ((uintptr_t) a1 - kc);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ph(c2, vacc2x0);
      _mm512_storeu_ph((uint16_t*) c2 + 1, vacc2x1);
      a2 = (const uint16_t*) ((uintptr_t) a2 - kc);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ph(c3, vacc3x0);
      _mm512_storeu_ph((uint16_t*) c3 + 1, vacc3x1);
      a3 = (const uint16_t*) ((uintptr_t) a3 - kc);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ph(c4, vacc4x0);
      _mm512_storeu_ph((uint16_t*) c4 + 1, vacc4x1);
      a4 = (const uint16_t*) ((uintptr_t) a4 - kc);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ph(c5, vacc5x0);
      _mm512_storeu_ph((uint16_t*) c5 + 1, vacc5x1);
      a5 = (const uint16_t*) ((uintptr_t) a5 - kc);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ph(c6, vacc6x0);
      _mm512_storeu_ph((uint16_t*) c6 + 1, vacc6x1);
      a6 = (const uint16_t*) ((uintptr_t) a6 - kc);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);

      nc -= 64;
    } else {
      if (nc & 32) {
        _mm512_storeu_ph(c0, vacc0x0);
        _mm512_storeu_ph(c1, vacc1x0);
        _mm512_storeu_ph(c2, vacc2x0);
        _mm512_storeu_ph(c3, vacc3x0);
        _mm512_storeu_ph(c4, vacc4x0);
        _mm512_storeu_ph(c5, vacc5x0);
        _mm512_storeu_ph(c6, vacc6x0);

        vacc0x0 = vacc0x1;
        c0 += 32;
        vacc1x0 = vacc1x1;
        c1 += 32;
        vacc2x0 = vacc2x1;
        c2 += 32;
        vacc3x0 = vacc3x1;
        c3 += 32;
        vacc4x0 = vacc4x1;
        c4 += 32;
        vacc5x0 = vacc5x1;
        c5 += 32;
        vacc6x0 = vacc6x1;
        c6 += 32;
      }
      if (nc & 31) {
        // Prepare mask for valid 16-bit elements (depends on nc).
        const __mmask32 vmask = _cvtu32_mask32((uint32_t) (UINT32_C(1) << (nc & 31)) - UINT32_C(1));
        _mm512_mask_storeu_epi16(c0, vmask, _mm512_castph_si512(vacc0x0));
        _mm512_mask_storeu_epi16(c1, vmask, _mm512_castph_si512(vacc1x0));
        _mm512_mask_storeu_epi16(c2, vmask, _mm512_castph_si512(vacc2x0));
        _mm512_mask_storeu_epi16(c3, vmask, _mm512_castph_si512(vacc3x0));
        _mm512_mask_storeu_epi16(c4, vmask, _mm512_castph_si512(vacc4x0));
        _mm512_mask_storeu_epi16(c5, vmask, _mm512_castph_si512(vacc5x0));
        _mm512_mask_storeu_epi16(c6, vmask, _mm512_castph_si512(vacc6x0));
      }
      nc = 0;
    }
  } while (nc != 0);
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_igemm_minmax_ukernel_1x64__avx512fp16_broadcast(
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
    __m512h vacc0x1 = _mm512_load_ph((const uint16_t*) w + 32);
    w = (const uint16_t*) w + 64;

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
        const __m512h vb1 = _mm512_load_ph((const uint16_t*) w + 32);
        w = (const uint16_t*) w + 64;

        const __m512h va0 = _mm512_castsi512_ph(_mm512_set1_epi16(*a0));
        vacc0x0 = _mm512_fmadd_ph(va0, vb0, vacc0x0);
        vacc0x1 = _mm512_fmadd_ph(va0, vb1, vacc0x1);
        a0 += 1;

        k -= sizeof(uint16_t);
      } while (k != 0);

      p -= 1 * sizeof(void*);
    } while (p != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0 = _mm512_max_ph(vmin, vacc0x0);
    vacc0x1 = _mm512_max_ph(vmin, vacc0x1);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0 = _mm512_min_ph(vmax, vacc0x0);
    vacc0x1 = _mm512_min_ph(vmax, vacc0x1);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ph(c0, vacc0x0);
      _mm512_storeu_ph((uint16_t*) c0 + 1, vacc0x1);
      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);

      a = (const void**restrict) ((uintptr_t) a - ks);
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

void xnn_f16_igemm_minmax_ukernel_7x64__avx512fp16_broadcast(
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
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
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
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  uint16_t* c4 = (uint16_t*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  uint16_t* c5 = (uint16_t*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  uint16_t* c6 = (uint16_t*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
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
    __m512h vacc4x0 = vacc0x0;
    __m512h vacc4x1 = vacc0x1;
    __m512h vacc5x0 = vacc0x0;
    __m512h vacc5x1 = vacc0x1;
    __m512h vacc6x0 = vacc0x0;
    __m512h vacc6x1 = vacc0x1;
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
      const uint16_t* restrict a4 = (const uint16_t*) a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const uint16_t*) ((uintptr_t) a4 + a_offset);
      }
      const uint16_t* restrict a5 = (const uint16_t*) a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const uint16_t*) ((uintptr_t) a5 + a_offset);
      }
      const uint16_t* restrict a6 = (const uint16_t*) a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const uint16_t*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

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
        const __m512h va4 = _mm512_castsi512_ph(_mm512_set1_epi16(*a4));
        vacc4x0 = _mm512_fmadd_ph(va4, vb0, vacc4x0);
        vacc4x1 = _mm512_fmadd_ph(va4, vb1, vacc4x1);
        a4 += 1;
        const __m512h va5 = _mm512_castsi512_ph(_mm512_set1_epi16(*a5));
        vacc5x0 = _mm512_fmadd_ph(va5, vb0, vacc5x0);
        vacc5x1 = _mm512_fmadd_ph(va5, vb1, vacc5x1);
        a5 += 1;
        const __m512h va6 = _mm512_castsi512_ph(_mm512_set1_epi16(*a6));
        vacc6x0 = _mm512_fmadd_ph(va6, vb0, vacc6x0);
        vacc6x1 = _mm512_fmadd_ph(va6, vb1, vacc6x1);
        a6 += 1;

        k -= sizeof(uint16_t);
      } while (k != 0);

      p -= 7 * sizeof(void*);
    } while (p != 0);

    const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
    vacc0x0 = _mm512_max_ph(vmin, vacc0x0);
    vacc1x0 = _mm512_max_ph(vmin, vacc1x0);
    vacc2x0 = _mm512_max_ph(vmin, vacc2x0);
    vacc3x0 = _mm512_max_ph(vmin, vacc3x0);
    vacc4x0 = _mm512_max_ph(vmin, vacc4x0);
    vacc5x0 = _mm512_max_ph(vmin, vacc5x0);
    vacc6x0 = _mm512_max_ph(vmin, vacc6x0);
    vacc0x1 = _mm512_max_ph(vmin, vacc0x1);
    vacc1x1 = _mm512_max_ph(vmin, vacc1x1);
    vacc2x1 = _mm512_max_ph(vmin, vacc2x1);
    vacc3x1 = _mm512_max_ph(vmin, vacc3x1);
    vacc4x1 = _mm512_max_ph(vmin, vacc4x1);
    vacc5x1 = _mm512_max_ph(vmin, vacc5x1);
    vacc6x1 = _mm512_max_ph(vmin, vacc6x1);

    const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
    vacc0x0 = _mm512_min_ph(vmax, vacc0x0);
    vacc1x0 = _mm512_min_ph(vmax, vacc1x0);
    vacc2x0 = _mm512_min_ph(vmax, vacc2x0);
    vacc3x0 = _mm512_min_ph(vmax, vacc3x0);
    vacc4x0 = _mm512_min_ph(vmax, vacc4x0);
    vacc5x0 = _mm512_min_ph(vmax, vacc5x0);
    vacc6x0 = _mm512_min_ph(vmax, vacc6x0);
    vacc0x1 = _mm512_min_ph(vmax, vacc0x1);
    vacc1x1 = _mm512_min_ph(vmax, vacc1x1);
    vacc2x1 = _mm512_min_ph(vmax, vacc2x1);
    vacc3x1 = _mm512_min_ph(vmax, vacc3x1);
    vacc4x1 = _mm512_min_ph(vmax, vacc4x1);
    vacc5x1 = _mm512_min_ph(vmax, vacc5x1);
    vacc6x1 = _mm512_min_ph(vmax, vacc6x1);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ph(c6, vacc6x0);
      _mm512_storeu_ph((uint16_t*) c6 + 1, vacc6x1);
      c6 = (uint16_t*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ph(c5, vacc5x0);
      _mm512_storeu_ph((uint16_t*) c5 + 1, vacc5x1);
      c5 = (uint16_t*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ph(c4, vacc4x0);
      _mm512_storeu_ph((uint16_t*) c4 + 1, vacc4x1);
      c4 = (uint16_t*) ((uintptr_t) c4 + cn_stride);
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
        _mm512_storeu_ph(c6, vacc6x0);
        _mm512_storeu_ph(c5, vacc5x0);
        _mm512_storeu_ph(c4, vacc4x0);
        _mm512_storeu_ph(c3, vacc3x0);
        _mm512_storeu_ph(c2, vacc2x0);
        _mm512_storeu_ph(c1, vacc1x0);
        _mm512_storeu_ph(c0, vacc0x0);

        vacc6x0 = vacc6x1;
        c6 += 32;
        vacc5x0 = vacc5x1;
        c5 += 32;
        vacc4x0 = vacc4x1;
        c4 += 32;
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
        _mm512_mask_storeu_epi16(c6, vmask, _mm512_castph_si512(vacc6x0));
        _mm512_mask_storeu_epi16(c5, vmask, _mm512_castph_si512(vacc5x0));
        _mm512_mask_storeu_epi16(c4, vmask, _mm512_castph_si512(vacc4x0));
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

void xnn_f16_rmax_ukernel__avx512fp16_u128_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* i = (const uint16_t*) input;
  __m512h vmax0 = _mm512_castsi512_ph(_mm512_set1_epi16(*i));
  __m512h vmax1 = vmax0;
  __m512h vmax2 = vmax0;
  __m512h vmax3 = vmax0;
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    const __m512h vt0 = _mm512_loadu_ph(i);
    const __m512h vt1 = _mm512_loadu_ph((i + 32));
    const __m512h vt2 = _mm512_loadu_ph((i + 64));
    const __m512h vt3 = _mm512_loadu_ph((i + 96));
    i += 128;

    vmax0 = _mm512_max_ph(vmax0, vt0);
    vmax1 = _mm512_max_ph(vmax1, vt1);
    vmax2 = _mm512_max_ph(vmax2, vt2);
    vmax3 = _mm512_max_ph(vmax3, vt3);
  }
  vmax0 = _mm512_max_ph(vmax0, vmax1);
  vmax2 = _mm512_max_ph(vmax2, vmax3);
  vmax0 = _mm512_max_ph(vmax0, vmax2);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vmax0 = _mm512_max_ph(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vmax0 = _mm512_mask_max_ph(vmax0, vmask, vmax0, vt);
  }
  __m256h vmax256 = _mm256_max_ph(_mm512_castph512_ph256(vmax0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmax0), 1)));
  __m128h vmax = _mm_max_ph(_mm256_castph256_ph128(vmax256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmax256), 1)));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmax), _mm_castph_ps(vmax))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmax))));
  vmax = _mm_max_sh(vmax, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmax), 16)));

  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmax), 0);
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_rminmax_ukernel__avx512fp16_u128_acc4(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* i = (const uint16_t*) input;
  __m512h vmin0 = _mm512_castsi512_ph(_mm512_set1_epi16(*i));
  __m512h vmax0 = vmin0;
  __m512h vmin1 = vmin0;
  __m512h vmax1 = vmax0;
  __m512h vmin2 = vmin0;
  __m512h vmax2 = vmax0;
  __m512h vmin3 = vmin0;
  __m512h vmax3 = vmax0;
  for (; batch >= 128 * sizeof(uint16_t); batch -= 128 * sizeof(uint16_t)) {
    const __m512h vt0 = _mm512_loadu_ph(i);
    const __m512h vt1 = _mm512_loadu_ph((i + 32));
    const __m512h vt2 = _mm512_loadu_ph((i + 64));
    const __m512h vt3 = _mm512_loadu_ph((i + 96));
    i += 128;

    vmin0 = _mm512_min_ph(vmin0, vt0);
    vmax0 = _mm512_max_ph(vmax0, vt0);
    vmin1 = _mm512_min_ph(vmin1, vt1);
    vmax1 = _mm512_max_ph(vmax1, vt1);
    vmin2 = _mm512_min_ph(vmin2, vt2);
    vmax2 = _mm512_max_ph(vmax2, vt2);
    vmin3 = _mm512_min_ph(vmin3, vt3);
    vmax3 = _mm512_max_ph(vmax3, vt3);
  }
  vmin0 = _mm512_min_ph(vmin0, vmin1);
  vmax0 = _mm512_max_ph(vmax0, vmax1);
  vmin2 = _mm512_min_ph(vmin2, vmin3);
  vmax2 = _mm512_max_ph(vmax2, vmax3);
  vmin0 = _mm512_min_ph(vmin0, vmin2);
  vmax0 = _mm512_max_ph(vmax0, vmax2);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m512h vt = _mm512_loadu_ph(i);
    i += 32;

    vmin0 = _mm512_min_ph(vmin0, vt);
    vmax0 = _mm512_max_ph(vmax0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    const __m512h vt = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i));

    vmin0 = _mm512_mask_min_ph(vmin0, vmask, vmin0, vt);
    vmax0 = _mm512_mask_max_ph(vmax0, vmask, vmax0, vt);
  }
  __m256h vmin256 = _mm256_min_ph(_mm512_castph512_ph256(vmin0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmin0), 1)));
  __m256h vmax256 = _mm256_max_ph(_mm512_castph512_ph256(vmax0), _mm256_castpd_ph(_mm512_extractf64x4_pd(_mm512_castph_pd(vmax0), 1)));
  __m128h vmin = _mm_min_ph(_mm256_castph256_ph128(vmin256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmin256), 1)));
  __m128h vmax = _mm_max_ph(_mm256_castph256_ph128(vmax256), _mm_castps_ph(_mm256_extractf128_ps(_mm256_castph_ps(vmax256), 1)));
  vmin = _mm_min_ph(vmin, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmin), _mm_castph_ps(vmin))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehl_ps(_mm_castph_ps(vmax), _mm_castph_ps(vmax))));
  vmin = _mm_min_ph(vmin, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmin))));
  vmax = _mm_max_ph(vmax, _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(vmax))));
  vmin = _mm_min_sh(vmin, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmin), 16)));
  vmax = _mm_max_sh(vmax, _mm_castsi128_ph(_mm_srli_epi32(_mm_castph_si128(vmax), 16)));

  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmin), 0);
  *((uint16_t*) output + 1) = (uint16_t) _mm_extract_epi16(_mm_castph_si128(vmax), 0);
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vadd_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_add_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_add_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_add_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_add_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vaddc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_add_ph(vacc0, vb);
    vacc1 = _mm512_add_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_add_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_add_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vdiv_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_div_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_div_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_div_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_div_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vdivc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_div_ph(vacc0, vb);
    vacc1 = _mm512_div_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_div_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_div_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_max_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_max_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_max_ph(vacc, _mm512_loadu_ph(b));
    b += 32;


    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_max_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmaxc_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_max_ph(vacc0, vb);
    vacc1 = _mm512_max_ph(vacc1, vb);



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_max_ph(vacc, vb);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_max_ph(vmask, vacc, vb);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmin_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_min_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_min_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_min_ph(vacc, _mm512_loadu_ph(b));
    b += 32;


    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_min_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vminc_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_min_ph(vacc0, vb);
    vacc1 = _mm512_min_ph(vacc1, vb);



    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_min_ph(vacc, vb);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_min_ph(vmask, vacc, vb);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmul_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_mul_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_mul_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_mul_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vmulc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_mul_ph(vacc0, vb);
    vacc1 = _mm512_mul_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_mul_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vrdivc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_div_ph(vb, vacc0);
    vacc1 = _mm512_div_ph(vb, vacc1);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_div_ph(vb, vacc);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_div_ph(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vrsubc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vb, vacc0);
    vacc1 = _mm512_sub_ph(vb, vacc1);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vb, vacc);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vb, vacc);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsqrdiff_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_sub_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;

    vacc0 = _mm512_mul_ph(vacc0, vacc0);
    vacc1 = _mm512_mul_ph(vacc1, vacc1);


    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_mul_ph(vacc, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_mul_ph(vmask, vacc, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsqrdiffc_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, vb);
    vacc1 = _mm512_sub_ph(vacc1, vb);

    vacc0 = _mm512_mul_ph(vacc0, vacc0);
    vacc1 = _mm512_mul_ph(vacc1, vacc1);


    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, vb);
    vacc = _mm512_mul_ph(vacc, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_mul_ph(vmask, vacc, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsub_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, _mm512_loadu_ph(b));
    vacc1 = _mm512_sub_ph(vacc1, _mm512_loadu_ph(b + 32));
    b += 64;


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, _mm512_loadu_ph(b));
    b += 32;

    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, b)));

    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}

void xnn_f16_vsubc_minmax_ukernel__avx512fp16_u64(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

#if defined(__AVX512FP16__)
  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m512h voutput_min = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.min));
  const __m512h voutput_max = _mm512_castsi512_ph(_mm512_set1_epi16(params->fp16arith.max));
  const __m512h vb = _mm512_castsi512_ph(_mm512_set1_epi16(*b));

  for (; batch >= 64 * sizeof(uint16_t); batch -= 64 * sizeof(uint16_t)) {
    __m512h vacc0 = _mm512_loadu_ph(a);
    __m512h vacc1 = _mm512_loadu_ph(a + 32);
    a += 64;

    vacc0 = _mm512_sub_ph(vacc0, vb);
    vacc1 = _mm512_sub_ph(vacc1, vb);


    vacc0 = _mm512_max_ph(voutput_min, vacc0);
    vacc1 = _mm512_max_ph(voutput_min, vacc1);

    vacc0 = _mm512_min_ph(voutput_max, vacc0);
    vacc1 = _mm512_min_ph(voutput_max, vacc1);

    _mm512_storeu_ph(o, vacc0);
    _mm512_storeu_ph(o + 32, vacc1);
    o += 64;
  }
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m512h vacc = _mm512_loadu_ph(a);
    a += 32;

    vacc = _mm512_sub_ph(vacc, vb);
    vacc = _mm512_max_ph(voutput_min, vacc);
    vacc = _mm512_min_ph(voutput_max, vacc);

    _mm512_storeu_ph(o, vacc);
    o += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 31 * sizeof(uint16_t));
    // Prepare mask for valid 16-bit elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_HALF;
    const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512h vacc = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, a));

    vacc = _mm512_maskz_sub_ph(vmask, vacc, vb);
    vacc = _mm512_maskz_max_ph(vmask, voutput_min, vacc);
    vacc = _mm512_maskz_min_ph(vmask, voutput_max, vacc);
    _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc));
  }
#endif  // defined(__AVX512FP16__)
}
