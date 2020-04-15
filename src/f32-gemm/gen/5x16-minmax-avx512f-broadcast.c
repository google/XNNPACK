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

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float*restrict a,
    size_t a_stride,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 5);
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
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    w += 16;

    size_t k = kc;
    do {
      const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
      w += 16;

      vacc0x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a0), vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
      vacc1x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a1), vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
      vacc2x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a2), vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
      vacc3x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a3), vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
      vacc4x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a4), vb0123456789ABCDEF, vacc4x0123456789ABCDEF);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmax = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.max));
    vacc0x0123456789ABCDEF = _mm512_min_ps(vacc0x0123456789ABCDEF, vmax);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vacc1x0123456789ABCDEF, vmax);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vacc2x0123456789ABCDEF, vmax);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vacc3x0123456789ABCDEF, vmax);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vacc4x0123456789ABCDEF, vmax);

    const __m512 vmin = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.min));
    vacc0x0123456789ABCDEF = _mm512_max_ps(vacc0x0123456789ABCDEF, vmin);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vacc1x0123456789ABCDEF, vmin);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vacc2x0123456789ABCDEF, vmin);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vacc3x0123456789ABCDEF, vmin);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vacc4x0123456789ABCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c4, vacc4x0123456789ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0123456789ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0123456789ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0123456789ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0, vacc0x0123456789ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a4 = (const float*) ((uintptr_t) a4 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c4, vmask, vacc4x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c3, vmask, vacc3x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c2, vmask, vacc2x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c1, vmask, vacc1x0123456789ABCDEF);
        _mm512_mask_storeu_ps(c0, vmask, vacc0x0123456789ABCDEF);
      }

      nc = 0;
    }
  } while (nc != 0);
}
