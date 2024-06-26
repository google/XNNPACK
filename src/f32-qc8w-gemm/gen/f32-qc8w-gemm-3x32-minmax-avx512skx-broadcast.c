// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/avx512-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>
#include <smmintrin.h>

#include "xnnpack/gemm.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_3x32__avx512skx_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  do {
    __m512 vacc0x0 = _mm512_loadu_ps(w);
    __m512 vacc0x1 = _mm512_loadu_ps((const float*) w + 16);
    __m512 vacc1x0 = vacc0x0;
    __m512 vacc1x1 = vacc0x1;
    __m512 vacc2x0 = vacc0x0;
    __m512 vacc2x1 = vacc0x1;
    w = (const float*) w + 32;

    size_t k = kc;
    do {
      const __m512i vbi0 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) w));
      const __m512i vbi1  = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) ((const int8_t*) w + 16)));
      const __m512 vb0  = _mm512_cvtepi32_ps(vbi0);
      const __m512 vb1  = _mm512_cvtepi32_ps(vbi1);
      w = (const int8_t*) w + 32;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
      vacc0x1 = _mm512_fmadd_ps(va0, vb1, vacc0x1);
      const __m512 va1 = _mm512_set1_ps(*a1);
      vacc1x0 = _mm512_fmadd_ps(va1, vb0, vacc1x0);
      vacc1x1 = _mm512_fmadd_ps(va1, vb1, vacc1x1);
      const __m512 va2 = _mm512_set1_ps(*a2);
      vacc2x0 = _mm512_fmadd_ps(va2, vb0, vacc2x0);
      vacc2x1 = _mm512_fmadd_ps(va2, vb1, vacc2x1);

      a0 += 1;
      a1 += 1;
      a2 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vscale0 = _mm512_loadu_ps((const float*) w + 0);
    vacc0x0 = _mm512_mul_ps(vacc0x0, vscale0);
    vacc1x0 = _mm512_mul_ps(vacc1x0, vscale0);
    vacc2x0 = _mm512_mul_ps(vacc2x0, vscale0);
    const __m512 vscale1 = _mm512_loadu_ps((const float*) w + 16);
    vacc0x1 = _mm512_mul_ps(vacc0x1, vscale1);
    vacc1x1 = _mm512_mul_ps(vacc1x1, vscale1);
    vacc2x1 = _mm512_mul_ps(vacc2x1, vscale1);
    w = (const float*) w + 32;
    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc1x0 = _mm512_max_ps(vmin, vacc1x0);
    vacc2x0 = _mm512_max_ps(vmin, vacc2x0);
    vacc0x1 = _mm512_max_ps(vmin, vacc0x1);
    vacc1x1 = _mm512_max_ps(vmin, vacc1x1);
    vacc2x1 = _mm512_max_ps(vmin, vacc2x1);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);
    vacc1x0 = _mm512_min_ps(vmax, vacc1x0);
    vacc2x0 = _mm512_min_ps(vmax, vacc2x0);
    vacc0x1 = _mm512_min_ps(vmax, vacc0x1);
    vacc1x1 = _mm512_min_ps(vmax, vacc1x1);
    vacc2x1 = _mm512_min_ps(vmax, vacc2x1);

    if XNN_LIKELY(nc >= 32) {
      _mm512_storeu_ps(c0, vacc0x0);
      _mm512_storeu_ps(c0 + 16, vacc0x1);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0);
      _mm512_storeu_ps(c1 + 16, vacc1x1);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0);
      _mm512_storeu_ps(c2 + 16, vacc2x1);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);

      nc -= 32;
    } else {
      if (nc & 16) {
        _mm512_storeu_ps(c0, vacc0x0);
        _mm512_storeu_ps(c1, vacc1x0);
        _mm512_storeu_ps(c2, vacc2x0);

        vacc0x0 = vacc0x1;
        vacc1x0 = vacc1x1;
        vacc2x0 = vacc2x1;

        c0 += 16;
        c1 += 16;
        c2 += 16;
      }
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) (UINT32_C(1) << (nc & 15)) - UINT32_C(1));
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
