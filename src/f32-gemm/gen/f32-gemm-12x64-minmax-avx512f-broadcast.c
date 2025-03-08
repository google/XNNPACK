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


void xnn_f32_gemm_minmax_ukernel_12x64__avx512f_broadcast(
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
  assert(mr <= 12);
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
  if XNN_UNPREDICTABLE(mr != 12) {
    a11 = a10;
    c11 = c10;
  }
  do {
    __m512 vacc0x0 = _mm512_load_ps(w);
    __m512 vacc0x1 = _mm512_load_ps(w + 16);
    __m512 vacc0x2 = _mm512_load_ps(w + 32);
    __m512 vacc0x3 = _mm512_load_ps(w + 48);
    __m512 vacc1x0 = vacc0x0;
    __m512 vacc1x1 = vacc0x1;
    __m512 vacc1x2 = vacc0x2;
    __m512 vacc1x3 = vacc0x3;
    __m512 vacc2x0 = vacc0x0;
    __m512 vacc2x1 = vacc0x1;
    __m512 vacc2x2 = vacc0x2;
    __m512 vacc2x3 = vacc0x3;
    __m512 vacc3x0 = vacc0x0;
    __m512 vacc3x1 = vacc0x1;
    __m512 vacc3x2 = vacc0x2;
    __m512 vacc3x3 = vacc0x3;
    __m512 vacc4x0 = vacc0x0;
    __m512 vacc4x1 = vacc0x1;
    __m512 vacc4x2 = vacc0x2;
    __m512 vacc4x3 = vacc0x3;
    __m512 vacc5x0 = vacc0x0;
    __m512 vacc5x1 = vacc0x1;
    __m512 vacc5x2 = vacc0x2;
    __m512 vacc5x3 = vacc0x3;
    __m512 vacc6x0 = vacc0x0;
    __m512 vacc6x1 = vacc0x1;
    __m512 vacc6x2 = vacc0x2;
    __m512 vacc6x3 = vacc0x3;
    __m512 vacc7x0 = vacc0x0;
    __m512 vacc7x1 = vacc0x1;
    __m512 vacc7x2 = vacc0x2;
    __m512 vacc7x3 = vacc0x3;
    __m512 vacc8x0 = vacc0x0;
    __m512 vacc8x1 = vacc0x1;
    __m512 vacc8x2 = vacc0x2;
    __m512 vacc8x3 = vacc0x3;
    __m512 vacc9x0 = vacc0x0;
    __m512 vacc9x1 = vacc0x1;
    __m512 vacc9x2 = vacc0x2;
    __m512 vacc9x3 = vacc0x3;
    __m512 vacc10x0 = vacc0x0;
    __m512 vacc10x1 = vacc0x1;
    __m512 vacc10x2 = vacc0x2;
    __m512 vacc10x3 = vacc0x3;
    __m512 vacc11x0 = vacc0x0;
    __m512 vacc11x1 = vacc0x1;
    __m512 vacc11x2 = vacc0x2;
    __m512 vacc11x3 = vacc0x3;
    w += 64;

    size_t k = kc;
    do {
      const __m512 vb0 = _mm512_load_ps(w);
      const __m512 vb1 = _mm512_loadu_ps(w + 16);
      const __m512 vb2 = _mm512_loadu_ps(w + 32);
      const __m512 vb3 = _mm512_loadu_ps(w + 48);
      w += 64;

      const __m512 va0 = _mm512_set1_ps(*a0);
      vacc0x0 = _mm512_fmadd_ps(va0, vb0, vacc0x0);
      vacc0x1 = _mm512_fmadd_ps(va0, vb1, vacc0x1);
      vacc0x2 = _mm512_fmadd_ps(va0, vb2, vacc0x2);
      vacc0x3 = _mm512_fmadd_ps(va0, vb3, vacc0x3);
      const __m512 va1 = _mm512_set1_ps(*a1);
      vacc1x0 = _mm512_fmadd_ps(va1, vb0, vacc1x0);
      vacc1x1 = _mm512_fmadd_ps(va1, vb1, vacc1x1);
      vacc1x2 = _mm512_fmadd_ps(va1, vb2, vacc1x2);
      vacc1x3 = _mm512_fmadd_ps(va1, vb3, vacc1x3);
      const __m512 va2 = _mm512_set1_ps(*a2);
      vacc2x0 = _mm512_fmadd_ps(va2, vb0, vacc2x0);
      vacc2x1 = _mm512_fmadd_ps(va2, vb1, vacc2x1);
      vacc2x2 = _mm512_fmadd_ps(va2, vb2, vacc2x2);
      vacc2x3 = _mm512_fmadd_ps(va2, vb3, vacc2x3);
      const __m512 va3 = _mm512_set1_ps(*a3);
      vacc3x0 = _mm512_fmadd_ps(va3, vb0, vacc3x0);
      vacc3x1 = _mm512_fmadd_ps(va3, vb1, vacc3x1);
      vacc3x2 = _mm512_fmadd_ps(va3, vb2, vacc3x2);
      vacc3x3 = _mm512_fmadd_ps(va3, vb3, vacc3x3);
      const __m512 va4 = _mm512_set1_ps(*a4);
      vacc4x0 = _mm512_fmadd_ps(va4, vb0, vacc4x0);
      vacc4x1 = _mm512_fmadd_ps(va4, vb1, vacc4x1);
      vacc4x2 = _mm512_fmadd_ps(va4, vb2, vacc4x2);
      vacc4x3 = _mm512_fmadd_ps(va4, vb3, vacc4x3);
      const __m512 va5 = _mm512_set1_ps(*a5);
      vacc5x0 = _mm512_fmadd_ps(va5, vb0, vacc5x0);
      vacc5x1 = _mm512_fmadd_ps(va5, vb1, vacc5x1);
      vacc5x2 = _mm512_fmadd_ps(va5, vb2, vacc5x2);
      vacc5x3 = _mm512_fmadd_ps(va5, vb3, vacc5x3);
      const __m512 va6 = _mm512_set1_ps(*a6);
      vacc6x0 = _mm512_fmadd_ps(va6, vb0, vacc6x0);
      vacc6x1 = _mm512_fmadd_ps(va6, vb1, vacc6x1);
      vacc6x2 = _mm512_fmadd_ps(va6, vb2, vacc6x2);
      vacc6x3 = _mm512_fmadd_ps(va6, vb3, vacc6x3);
      const __m512 va7 = _mm512_set1_ps(*a7);
      vacc7x0 = _mm512_fmadd_ps(va7, vb0, vacc7x0);
      vacc7x1 = _mm512_fmadd_ps(va7, vb1, vacc7x1);
      vacc7x2 = _mm512_fmadd_ps(va7, vb2, vacc7x2);
      vacc7x3 = _mm512_fmadd_ps(va7, vb3, vacc7x3);
      const __m512 va8 = _mm512_set1_ps(*a8);
      vacc8x0 = _mm512_fmadd_ps(va8, vb0, vacc8x0);
      vacc8x1 = _mm512_fmadd_ps(va8, vb1, vacc8x1);
      vacc8x2 = _mm512_fmadd_ps(va8, vb2, vacc8x2);
      vacc8x3 = _mm512_fmadd_ps(va8, vb3, vacc8x3);
      const __m512 va9 = _mm512_set1_ps(*a9);
      vacc9x0 = _mm512_fmadd_ps(va9, vb0, vacc9x0);
      vacc9x1 = _mm512_fmadd_ps(va9, vb1, vacc9x1);
      vacc9x2 = _mm512_fmadd_ps(va9, vb2, vacc9x2);
      vacc9x3 = _mm512_fmadd_ps(va9, vb3, vacc9x3);
      const __m512 va10 = _mm512_set1_ps(*a10);
      vacc10x0 = _mm512_fmadd_ps(va10, vb0, vacc10x0);
      vacc10x1 = _mm512_fmadd_ps(va10, vb1, vacc10x1);
      vacc10x2 = _mm512_fmadd_ps(va10, vb2, vacc10x2);
      vacc10x3 = _mm512_fmadd_ps(va10, vb3, vacc10x3);
      const __m512 va11 = _mm512_set1_ps(*a11);
      vacc11x0 = _mm512_fmadd_ps(va11, vb0, vacc11x0);
      vacc11x1 = _mm512_fmadd_ps(va11, vb1, vacc11x1);
      vacc11x2 = _mm512_fmadd_ps(va11, vb2, vacc11x2);
      vacc11x3 = _mm512_fmadd_ps(va11, vb3, vacc11x3);

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
    vacc0x1 = _mm512_max_ps(vmin, vacc0x1);
    vacc1x1 = _mm512_max_ps(vmin, vacc1x1);
    vacc2x1 = _mm512_max_ps(vmin, vacc2x1);
    vacc3x1 = _mm512_max_ps(vmin, vacc3x1);
    vacc4x1 = _mm512_max_ps(vmin, vacc4x1);
    vacc5x1 = _mm512_max_ps(vmin, vacc5x1);
    vacc6x1 = _mm512_max_ps(vmin, vacc6x1);
    vacc7x1 = _mm512_max_ps(vmin, vacc7x1);
    vacc8x1 = _mm512_max_ps(vmin, vacc8x1);
    vacc9x1 = _mm512_max_ps(vmin, vacc9x1);
    vacc10x1 = _mm512_max_ps(vmin, vacc10x1);
    vacc11x1 = _mm512_max_ps(vmin, vacc11x1);
    vacc0x2 = _mm512_max_ps(vmin, vacc0x2);
    vacc1x2 = _mm512_max_ps(vmin, vacc1x2);
    vacc2x2 = _mm512_max_ps(vmin, vacc2x2);
    vacc3x2 = _mm512_max_ps(vmin, vacc3x2);
    vacc4x2 = _mm512_max_ps(vmin, vacc4x2);
    vacc5x2 = _mm512_max_ps(vmin, vacc5x2);
    vacc6x2 = _mm512_max_ps(vmin, vacc6x2);
    vacc7x2 = _mm512_max_ps(vmin, vacc7x2);
    vacc8x2 = _mm512_max_ps(vmin, vacc8x2);
    vacc9x2 = _mm512_max_ps(vmin, vacc9x2);
    vacc10x2 = _mm512_max_ps(vmin, vacc10x2);
    vacc11x2 = _mm512_max_ps(vmin, vacc11x2);
    vacc0x3 = _mm512_max_ps(vmin, vacc0x3);
    vacc1x3 = _mm512_max_ps(vmin, vacc1x3);
    vacc2x3 = _mm512_max_ps(vmin, vacc2x3);
    vacc3x3 = _mm512_max_ps(vmin, vacc3x3);
    vacc4x3 = _mm512_max_ps(vmin, vacc4x3);
    vacc5x3 = _mm512_max_ps(vmin, vacc5x3);
    vacc6x3 = _mm512_max_ps(vmin, vacc6x3);
    vacc7x3 = _mm512_max_ps(vmin, vacc7x3);
    vacc8x3 = _mm512_max_ps(vmin, vacc8x3);
    vacc9x3 = _mm512_max_ps(vmin, vacc9x3);
    vacc10x3 = _mm512_max_ps(vmin, vacc10x3);
    vacc11x3 = _mm512_max_ps(vmin, vacc11x3);

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
    vacc0x1 = _mm512_min_ps(vmax, vacc0x1);
    vacc1x1 = _mm512_min_ps(vmax, vacc1x1);
    vacc2x1 = _mm512_min_ps(vmax, vacc2x1);
    vacc3x1 = _mm512_min_ps(vmax, vacc3x1);
    vacc4x1 = _mm512_min_ps(vmax, vacc4x1);
    vacc5x1 = _mm512_min_ps(vmax, vacc5x1);
    vacc6x1 = _mm512_min_ps(vmax, vacc6x1);
    vacc7x1 = _mm512_min_ps(vmax, vacc7x1);
    vacc8x1 = _mm512_min_ps(vmax, vacc8x1);
    vacc9x1 = _mm512_min_ps(vmax, vacc9x1);
    vacc10x1 = _mm512_min_ps(vmax, vacc10x1);
    vacc11x1 = _mm512_min_ps(vmax, vacc11x1);
    vacc0x2 = _mm512_min_ps(vmax, vacc0x2);
    vacc1x2 = _mm512_min_ps(vmax, vacc1x2);
    vacc2x2 = _mm512_min_ps(vmax, vacc2x2);
    vacc3x2 = _mm512_min_ps(vmax, vacc3x2);
    vacc4x2 = _mm512_min_ps(vmax, vacc4x2);
    vacc5x2 = _mm512_min_ps(vmax, vacc5x2);
    vacc6x2 = _mm512_min_ps(vmax, vacc6x2);
    vacc7x2 = _mm512_min_ps(vmax, vacc7x2);
    vacc8x2 = _mm512_min_ps(vmax, vacc8x2);
    vacc9x2 = _mm512_min_ps(vmax, vacc9x2);
    vacc10x2 = _mm512_min_ps(vmax, vacc10x2);
    vacc11x2 = _mm512_min_ps(vmax, vacc11x2);
    vacc0x3 = _mm512_min_ps(vmax, vacc0x3);
    vacc1x3 = _mm512_min_ps(vmax, vacc1x3);
    vacc2x3 = _mm512_min_ps(vmax, vacc2x3);
    vacc3x3 = _mm512_min_ps(vmax, vacc3x3);
    vacc4x3 = _mm512_min_ps(vmax, vacc4x3);
    vacc5x3 = _mm512_min_ps(vmax, vacc5x3);
    vacc6x3 = _mm512_min_ps(vmax, vacc6x3);
    vacc7x3 = _mm512_min_ps(vmax, vacc7x3);
    vacc8x3 = _mm512_min_ps(vmax, vacc8x3);
    vacc9x3 = _mm512_min_ps(vmax, vacc9x3);
    vacc10x3 = _mm512_min_ps(vmax, vacc10x3);
    vacc11x3 = _mm512_min_ps(vmax, vacc11x3);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ps(c0, vacc0x0);
      _mm512_storeu_ps(c0 + 16, vacc0x1);
      _mm512_storeu_ps(c0 + 32, vacc0x2);
      _mm512_storeu_ps(c0 + 48, vacc0x3);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm512_storeu_ps(c1, vacc1x0);
      _mm512_storeu_ps(c1 + 16, vacc1x1);
      _mm512_storeu_ps(c1 + 32, vacc1x2);
      _mm512_storeu_ps(c1 + 48, vacc1x3);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c2, vacc2x0);
      _mm512_storeu_ps(c2 + 16, vacc2x1);
      _mm512_storeu_ps(c2 + 32, vacc2x2);
      _mm512_storeu_ps(c2 + 48, vacc2x3);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c3, vacc3x0);
      _mm512_storeu_ps(c3 + 16, vacc3x1);
      _mm512_storeu_ps(c3 + 32, vacc3x2);
      _mm512_storeu_ps(c3 + 48, vacc3x3);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c4, vacc4x0);
      _mm512_storeu_ps(c4 + 16, vacc4x1);
      _mm512_storeu_ps(c4 + 32, vacc4x2);
      _mm512_storeu_ps(c4 + 48, vacc4x3);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm512_storeu_ps(c5, vacc5x0);
      _mm512_storeu_ps(c5 + 16, vacc5x1);
      _mm512_storeu_ps(c5 + 32, vacc5x2);
      _mm512_storeu_ps(c5 + 48, vacc5x3);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);
      _mm512_storeu_ps(c6, vacc6x0);
      _mm512_storeu_ps(c6 + 16, vacc6x1);
      _mm512_storeu_ps(c6 + 32, vacc6x2);
      _mm512_storeu_ps(c6 + 48, vacc6x3);
      c6 = (float*) ((uintptr_t) c6 + cn_stride);
      _mm512_storeu_ps(c7, vacc7x0);
      _mm512_storeu_ps(c7 + 16, vacc7x1);
      _mm512_storeu_ps(c7 + 32, vacc7x2);
      _mm512_storeu_ps(c7 + 48, vacc7x3);
      c7 = (float*) ((uintptr_t) c7 + cn_stride);
      _mm512_storeu_ps(c8, vacc8x0);
      _mm512_storeu_ps(c8 + 16, vacc8x1);
      _mm512_storeu_ps(c8 + 32, vacc8x2);
      _mm512_storeu_ps(c8 + 48, vacc8x3);
      c8 = (float*) ((uintptr_t) c8 + cn_stride);
      _mm512_storeu_ps(c9, vacc9x0);
      _mm512_storeu_ps(c9 + 16, vacc9x1);
      _mm512_storeu_ps(c9 + 32, vacc9x2);
      _mm512_storeu_ps(c9 + 48, vacc9x3);
      c9 = (float*) ((uintptr_t) c9 + cn_stride);
      _mm512_storeu_ps(c10, vacc10x0);
      _mm512_storeu_ps(c10 + 16, vacc10x1);
      _mm512_storeu_ps(c10 + 32, vacc10x2);
      _mm512_storeu_ps(c10 + 48, vacc10x3);
      c10 = (float*) ((uintptr_t) c10 + cn_stride);
      _mm512_storeu_ps(c11, vacc11x0);
      _mm512_storeu_ps(c11 + 16, vacc11x1);
      _mm512_storeu_ps(c11 + 32, vacc11x2);
      _mm512_storeu_ps(c11 + 48, vacc11x3);
      c11 = (float*) ((uintptr_t) c11 + cn_stride);

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

      nc -= 64;
    } else {
      // NC remainder (1..63)
      assert(nc >= 1);
      assert(nc <= 63);
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 0));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 16));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 32));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) (((UINT64_C(1) << nc) - 1) >> 48));

      _mm512_mask_storeu_ps(c0 + 0, vmask0, vacc0x0);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vacc0x1);
      _mm512_mask_storeu_ps(c0 + 32, vmask2, vacc0x2);
      _mm512_mask_storeu_ps(c0 + 48, vmask3, vacc0x3);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vacc1x0);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vacc1x1);
      _mm512_mask_storeu_ps(c1 + 32, vmask2, vacc1x2);
      _mm512_mask_storeu_ps(c1 + 48, vmask3, vacc1x3);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vacc2x0);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vacc2x1);
      _mm512_mask_storeu_ps(c2 + 32, vmask2, vacc2x2);
      _mm512_mask_storeu_ps(c2 + 48, vmask3, vacc2x3);
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vacc3x0);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vacc3x1);
      _mm512_mask_storeu_ps(c3 + 32, vmask2, vacc3x2);
      _mm512_mask_storeu_ps(c3 + 48, vmask3, vacc3x3);
      _mm512_mask_storeu_ps(c4 + 0, vmask0, vacc4x0);
      _mm512_mask_storeu_ps(c4 + 16, vmask1, vacc4x1);
      _mm512_mask_storeu_ps(c4 + 32, vmask2, vacc4x2);
      _mm512_mask_storeu_ps(c4 + 48, vmask3, vacc4x3);
      _mm512_mask_storeu_ps(c5 + 0, vmask0, vacc5x0);
      _mm512_mask_storeu_ps(c5 + 16, vmask1, vacc5x1);
      _mm512_mask_storeu_ps(c5 + 32, vmask2, vacc5x2);
      _mm512_mask_storeu_ps(c5 + 48, vmask3, vacc5x3);
      _mm512_mask_storeu_ps(c6 + 0, vmask0, vacc6x0);
      _mm512_mask_storeu_ps(c6 + 16, vmask1, vacc6x1);
      _mm512_mask_storeu_ps(c6 + 32, vmask2, vacc6x2);
      _mm512_mask_storeu_ps(c6 + 48, vmask3, vacc6x3);
      _mm512_mask_storeu_ps(c7 + 0, vmask0, vacc7x0);
      _mm512_mask_storeu_ps(c7 + 16, vmask1, vacc7x1);
      _mm512_mask_storeu_ps(c7 + 32, vmask2, vacc7x2);
      _mm512_mask_storeu_ps(c7 + 48, vmask3, vacc7x3);
      _mm512_mask_storeu_ps(c8 + 0, vmask0, vacc8x0);
      _mm512_mask_storeu_ps(c8 + 16, vmask1, vacc8x1);
      _mm512_mask_storeu_ps(c8 + 32, vmask2, vacc8x2);
      _mm512_mask_storeu_ps(c8 + 48, vmask3, vacc8x3);
      _mm512_mask_storeu_ps(c9 + 0, vmask0, vacc9x0);
      _mm512_mask_storeu_ps(c9 + 16, vmask1, vacc9x1);
      _mm512_mask_storeu_ps(c9 + 32, vmask2, vacc9x2);
      _mm512_mask_storeu_ps(c9 + 48, vmask3, vacc9x3);
      _mm512_mask_storeu_ps(c10 + 0, vmask0, vacc10x0);
      _mm512_mask_storeu_ps(c10 + 16, vmask1, vacc10x1);
      _mm512_mask_storeu_ps(c10 + 32, vmask2, vacc10x2);
      _mm512_mask_storeu_ps(c10 + 48, vmask3, vacc10x3);
      _mm512_mask_storeu_ps(c11 + 0, vmask0, vacc11x0);
      _mm512_mask_storeu_ps(c11 + 16, vmask1, vacc11x1);
      _mm512_mask_storeu_ps(c11 + 32, vmask2, vacc11x2);
      _mm512_mask_storeu_ps(c11 + 48, vmask3, vacc11x3);
      nc = 0;
    }
  } while (nc != 0);
}
