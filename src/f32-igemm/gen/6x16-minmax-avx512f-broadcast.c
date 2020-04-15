// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/avx512-broadcast.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/igemm.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_f32_igemm_minmax_ukernel_6x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float**restrict a,
    const float*restrict w,
    float*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (6 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    c5 = c4;
  }

  do {
    __m512 vacc0x0123456789ABCDEF = _mm512_load_ps(w);
    __m512 vacc1x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc2x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc3x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc4x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    __m512 vacc5x0123456789ABCDEF = vacc0x0123456789ABCDEF;
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      a += 6;

      size_t k = kc;
      do {
        const __m512 vb0123456789ABCDEF = _mm512_load_ps(w);
        w += 16;

        vacc0x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a0), vb0123456789ABCDEF, vacc0x0123456789ABCDEF);
        vacc1x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a1), vb0123456789ABCDEF, vacc1x0123456789ABCDEF);
        vacc2x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a2), vb0123456789ABCDEF, vacc2x0123456789ABCDEF);
        vacc3x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a3), vb0123456789ABCDEF, vacc3x0123456789ABCDEF);
        vacc4x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a4), vb0123456789ABCDEF, vacc4x0123456789ABCDEF);
        vacc5x0123456789ABCDEF = _mm512_fmadd_ps(_mm512_set1_ps(*a5), vb0123456789ABCDEF, vacc5x0123456789ABCDEF);

        a0 += 1;
        a1 += 1;
        a2 += 1;
        a3 += 1;
        a4 += 1;
        a5 += 1;

        k -= sizeof(float);
      } while (k != 0);
      p -= 6 * sizeof(void*);
    } while (p != 0);

    const __m512 vmax = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.max));
    vacc0x0123456789ABCDEF = _mm512_min_ps(vacc0x0123456789ABCDEF, vmax);
    vacc1x0123456789ABCDEF = _mm512_min_ps(vacc1x0123456789ABCDEF, vmax);
    vacc2x0123456789ABCDEF = _mm512_min_ps(vacc2x0123456789ABCDEF, vmax);
    vacc3x0123456789ABCDEF = _mm512_min_ps(vacc3x0123456789ABCDEF, vmax);
    vacc4x0123456789ABCDEF = _mm512_min_ps(vacc4x0123456789ABCDEF, vmax);
    vacc5x0123456789ABCDEF = _mm512_min_ps(vacc5x0123456789ABCDEF, vmax);

    const __m512 vmin = _mm512_broadcast_f32x4(_mm_load_ps(params->sse.min));
    vacc0x0123456789ABCDEF = _mm512_max_ps(vacc0x0123456789ABCDEF, vmin);
    vacc1x0123456789ABCDEF = _mm512_max_ps(vacc1x0123456789ABCDEF, vmin);
    vacc2x0123456789ABCDEF = _mm512_max_ps(vacc2x0123456789ABCDEF, vmin);
    vacc3x0123456789ABCDEF = _mm512_max_ps(vacc3x0123456789ABCDEF, vmin);
    vacc4x0123456789ABCDEF = _mm512_max_ps(vacc4x0123456789ABCDEF, vmin);
    vacc5x0123456789ABCDEF = _mm512_max_ps(vacc5x0123456789ABCDEF, vmin);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c5, vacc5x0123456789ABCDEF);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
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

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 15) {
        // Prepare mask for valid 32-bit elements (depends on nc).
        const __mmask16 vmask = _cvtu32_mask16((uint16_t) ((uint32_t) (UINT32_C(1) << nc) - UINT32_C(1)));

        _mm512_mask_storeu_ps(c5, vmask, vacc5x0123456789ABCDEF);
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
