// clang-format off
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

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/intrinsics-polyfill.h"


void xnn_f32_gemm_minmax_ukernel_16x16__avx512f_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 16);
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
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }
  const float* a7 = (const float*) ((uintptr_t) a6 + a_stride);
  float* c7 = (float*) ((uintptr_t) c6 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 8) {
    a7 = a6;
    c7 = c6;
  }
  const float* a8 = (const float*) ((uintptr_t) a7 + a_stride);
  float* c8 = (float*) ((uintptr_t) c7 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 8) {
    a8 = a7;
    c8 = c7;
  }
  const float* a9 = (const float*) ((uintptr_t) a8 + a_stride);
  float* c9 = (float*) ((uintptr_t) c8 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 10) {
    a9 = a8;
    c9 = c8;
  }
  const float* a10 = (const float*) ((uintptr_t) a9 + a_stride);
  float* c10 = (float*) ((uintptr_t) c9 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 10) {
    a10 = a9;
    c10 = c9;
  }
  const float* a11 = (const float*) ((uintptr_t) a10 + a_stride);
  float* c11 = (float*) ((uintptr_t) c10 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 12) {
    a11 = a10;
    c11 = c10;
  }
  const float* a12 = (const float*) ((uintptr_t) a11 + a_stride);
  float* c12 = (float*) ((uintptr_t) c11 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 12) {
    a12 = a11;
    c12 = c11;
  }
  const float* a13 = (const float*) ((uintptr_t) a12 + a_stride);
  float* c13 = (float*) ((uintptr_t) c12 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 14) {
    a13 = a12;
    c13 = c12;
  }
  const float* a14 = (const float*) ((uintptr_t) a13 + a_stride);
  float* c14 = (float*) ((uintptr_t) c13 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 14) {
    a14 = a13;
    c14 = c13;
  }
  const float* a15 = (const float*) ((uintptr_t) a14 + a_stride);
  float* c15 = (float*) ((uintptr_t) c14 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 16) {
    a15 = a14;
    c15 = c14;
  }
  do {
    __m512 vacc0x0 = _mm512_load_ps(w);
    __m512 vacc1x0 = vacc0x0;
    __m512 vacc2x0 = vacc0x0;
    __m512 vacc3x0 = vacc0x0;
    __m512 vacc4x0 = vacc0x0;
    __m512 vacc5x0 = vacc0x0;
    __m512 vacc6x0 = vacc0x0;
    __m512 vacc7x0 = vacc0x0;
    __m512 vacc8x0 = vacc0x0;
    __m512 vacc9x0 = vacc0x0;
    __m512 vacc10x0 = vacc0x0;
    __m512 vacc11x0 = vacc0x0;
    __m512 vacc12x0 = vacc0x0;
    __m512 vacc13x0 = vacc0x0;
    __m512 vacc14x0 = vacc0x0;
    __m512 vacc15x0 = vacc0x0;
    w += 16;

    size_t k = kc;
    do {
      const __m512 vb0 = _mm512_load_ps(w);
      w += 16;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
      const __m512 va1 = _mm512_set1_ps(*a1);
      vacc1x0 = _mm512_fmadd_ps(va1, vb0, vacc1x0);
      const __m512 va2 = _mm512_set1_ps(*a2);
      vacc2x0 = _mm512_fmadd_ps(va2, vb0, vacc2x0);
      const __m512 va3 = _mm512_set1_ps(*a3);
      vacc3x0 = _mm512_fmadd_ps(va3, vb0, vacc3x0);
      const __m512 va4 = _mm512_set1_ps(*a4);
      vacc4x0 = _mm512_fmadd_ps(va4, vb0, vacc4x0);
      const __m512 va5 = _mm512_set1_ps(*a5);
      vacc5x0 = _mm512_fmadd_ps(va5, vb0, vacc5x0);
      const __m512 va6 = _mm512_set1_ps(*a6);
      vacc6x0 = _mm512_fmadd_ps(va6, vb0, vacc6x0);
      const __m512 va7 = _mm512_set1_ps(*a7);
      vacc7x0 = _mm512_fmadd_ps(va7, vb0, vacc7x0);
      const __m512 va8 = _mm512_set1_ps(*a8);
      vacc8x0 = _mm512_fmadd_ps(va8, vb0, vacc8x0);
      const __m512 va9 = _mm512_set1_ps(*a9);
      vacc9x0 = _mm512_fmadd_ps(va9, vb0, vacc9x0);
      const __m512 va10 = _mm512_set1_ps(*a10);
      vacc10x0 = _mm512_fmadd_ps(va10, vb0, vacc10x0);
      const __m512 va11 = _mm512_set1_ps(*a11);
      vacc11x0 = _mm512_fmadd_ps(va11, vb0, vacc11x0);
      const __m512 va12 = _mm512_set1_ps(*a12);
      vacc12x0 = _mm512_fmadd_ps(va12, vb0, vacc12x0);
      const __m512 va13 = _mm512_set1_ps(*a13);
      vacc13x0 = _mm512_fmadd_ps(va13, vb0, vacc13x0);
      const __m512 va14 = _mm512_set1_ps(*a14);
      vacc14x0 = _mm512_fmadd_ps(va14, vb0, vacc14x0);
      const __m512 va15 = _mm512_set1_ps(*a15);
      vacc15x0 = _mm512_fmadd_ps(va15, vb0, vacc15x0);

      a0 += 1;
      a1 += 1;
      a2 += 1;
      a3 += 1;
      a4 += 1;
      a5 += 1;
      a6 += 1;
      a7 += 1;
      a8 += 1;
      a9 += 1;
      a10 += 1;
      a11 += 1;
      a12 += 1;
      a13 += 1;
      a14 += 1;
      a15 += 1;

      k -= sizeof(float);
    } while (k != 0);

    const __m512 vmin = _mm512_set1_ps(params->scalar.min);
    vacc0x0 = _mm512_max_ps(vmin, vacc0x0);
    vacc1x0 = _mm512_max_ps(vmin, vacc1x0);
    vacc2x0 = _mm512_max_ps(vmin, vacc2x0);
    vacc3x0 = _mm512_max_ps(vmin, vacc3x0);
    vacc4x0 = _mm512_max_ps(vmin, vacc4x0);
    vacc5x0 = _mm512_max_ps(vmin, vacc5x0);
    vacc6x0 = _mm512_max_ps(vmin, vacc6x0);
    vacc7x0 = _mm512_max_ps(vmin, vacc7x0);
    vacc8x0 = _mm512_max_ps(vmin, vacc8x0);
    vacc9x0 = _mm512_max_ps(vmin, vacc9x0);
    vacc10x0 = _mm512_max_ps(vmin, vacc10x0);
    vacc11x0 = _mm512_max_ps(vmin, vacc11x0);
    vacc12x0 = _mm512_max_ps(vmin, vacc12x0);
    vacc13x0 = _mm512_max_ps(vmin, vacc13x0);
    vacc14x0 = _mm512_max_ps(vmin, vacc14x0);
    vacc15x0 = _mm512_max_ps(vmin, vacc15x0);

    const __m512 vmax = _mm512_set1_ps(params->scalar.max);
    vacc0x0 = _mm512_min_ps(vmax, vacc0x0);
    vacc1x0 = _mm512_min_ps(vmax, vacc1x0);
    vacc2x0 = _mm512_min_ps(vmax, vacc2x0);
    vacc3x0 = _mm512_min_ps(vmax, vacc3x0);
    vacc4x0 = _mm512_min_ps(vmax, vacc4x0);
    vacc5x0 = _mm512_min_ps(vmax, vacc5x0);
    vacc6x0 = _mm512_min_ps(vmax, vacc6x0);
    vacc7x0 = _mm512_min_ps(vmax, vacc7x0);
    vacc8x0 = _mm512_min_ps(vmax, vacc8x0);
    vacc9x0 = _mm512_min_ps(vmax, vacc9x0);
    vacc10x0 = _mm512_min_ps(vmax, vacc10x0);
    vacc11x0 = _mm512_min_ps(vmax, vacc11x0);
    vacc12x0 = _mm512_min_ps(vmax, vacc12x0);
    vacc13x0 = _mm512_min_ps(vmax, vacc13x0);
    vacc14x0 = _mm512_min_ps(vmax, vacc14x0);
    vacc15x0 = _mm512_min_ps(vmax, vacc15x0);

    if XNN_LIKELY(nc >= 16) {
      _mm512_storeu_ps(c0, vacc0x0);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c6, vacc6x0);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c7, vacc7x0);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c8, vacc8x0);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c9, vacc9x0);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c10, vacc10x0);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c11, vacc11x0);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);
      _mm512_storeu_ps(c12, vacc12x0);
      c12 = (float*) ((uintptr_t) c12 + cn_stride);
      _mm512_storeu_ps(c13, vacc13x0);
      c13 = (float*) ((uintptr_t) c13 + cn_stride);
      _mm512_storeu_ps(c14, vacc14x0);
      c14 = (float*) ((uintptr_t) c14 + cn_stride);
      _mm512_storeu_ps(c15, vacc15x0);
      c15 = (float*) ((uintptr_t) c15 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);
      a5 = (const float*) ((uintptr_t) a5 - kc);
      a6 = (const float*) ((uintptr_t) a6 - kc);
      a7 = (const float*) ((uintptr_t) a7 - kc);
      a8 = (const float*) ((uintptr_t) a8 - kc);
      a9 = (const float*) ((uintptr_t) a9 - kc);
      a10 = (const float*) ((uintptr_t) a10 - kc);
      a11 = (const float*) ((uintptr_t) a11 - kc);
      a12 = (const float*) ((uintptr_t) a12 - kc);
      a13 = (const float*) ((uintptr_t) a13 - kc);
      a14 = (const float*) ((uintptr_t) a14 - kc);
      a15 = (const float*) ((uintptr_t) a15 - kc);

      nc -= 16;
    } else {
      // NC remainder (1..15)
      assert(nc >= 1);
      assert(nc <= 15);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));

      _mm512_mask_storeu_ps(c0 + 0, vmask0, vacc0x0);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vacc1x0);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vacc2x0);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vacc3x0);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vacc4x0);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vacc5x0);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vacc6x0);
      _mm512_mask_storeu_ps(c7 + 0, vmask0, vacc7x0);
      _mm512_mask_storeu_ps(c8 + 0, vmask0, vacc8x0);
      _mm512_mask_storeu_ps(c9 + 0, vmask0, vacc9x0);
      _mm512_mask_storeu_ps(c10 + 0, vmask0, vacc10x0);
      _mm512_mask_storeu_ps(c11 + 0, vmask0, vacc11x0);
      _mm512_mask_storeu_ps(c12 + 0, vmask0, vacc12x0);
      _mm512_mask_storeu_ps(c13 + 0, vmask0, vacc13x0);
      _mm512_mask_storeu_ps(c14 + 0, vmask0, vacc14x0);
      _mm512_mask_storeu_ps(c15 + 0, vmask0, vacc15x0);
      nc = 0;
    }
  } while (nc != 0);
}
